import os
import json
import random
from tqdm import tqdm
import openai
from collections import defaultdict
from sklearn.metrics import f1_score
import pandas as pd

# ===== 환경 세팅 및 경로 =====
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

GROUND_TRUTH_PATH = "data/patient_psi_testset.json"
RESPONSE_PATH = "response/0.5B_EP4_LR5e-4_test.jsonl"  # 모델 생성 응답 텍스트
RESULT_PATH = "evaluation/accuracy/acc_eval_llmjudge_0.5.jsonl"

ACC_CATEGORIES = [
    "situation",
    "coping_strategies",
    "intermediate_beliefs",
    "automatic_thoughts",
    "core_beliefs_fine",
]
F1_CATEGORIES = [
    "core_beliefs",
    "emotions",
    "behaviors",
]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def gpt41mini_chat(prompt, temperature=0.0):
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=128,
    )
    return response["choices"][0]["message"]["content"].strip()

def extract_str(x):
    # 정답셋 필드에서 string 추출
    if isinstance(x, list):
        return str(x[0]) if x else ""
    return str(x)

def format_acc_prompt(field, choices, patient_response):
    letter_map = ["A", "B", "C", "D"]
    prompt = f"""You are a licensed CBT (Cognitive Behavioral Therapy) supervisor evaluating a simulated patient's response.

Patient response:
\"{patient_response}\"

Based on the content and style of the patient's response, which of the following most accurately reflects the '{field.replace('_',' ')}'? Choose only one option.

Choices:"""
    for i, c in enumerate(choices):
        prompt += f"\n{letter_map[i]}. {c}"
    prompt += "\n\nImportant: Respond with only the letter (A, B, C, or D). Do not include any explanation or extra text."
    return prompt

import re
def extract_acc_letter(response):
    response = response.strip().upper()
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)
    # fallback: 전체 문자열에서 가장 먼저 등장하는 A~D
    for c in "ABCD":
        if c in response:
            return c
    return None

def format_f1_prompt(field, label_pool, patient_response):
    prompt = f"""You are a CBT (Cognitive Behavioral Therapy) supervisor.

Patient response:
\"{patient_response}\"

From the following '{field.replace('_',' ')}' options, list all that apply to the patient (comma-separated):

{', '.join(label_pool)}

List your answer as a comma-separated list of labels.
"""
    return prompt

def extract_f1_labels(response, label_pool):
    found = []
    resp = response.lower()
    for label in label_pool:
        if label.lower() in resp:
            found.append(label)
    return list(set(found))

def main():
    # ----- 데이터 로드 -----
    gt_data = load_json(GROUND_TRUTH_PATH)
    model_resps = load_jsonl(RESPONSE_PATH)

    def build_label_pool(field, data):
        pool = set()
        for d in data:
            labels = d.get(field, [])
            if isinstance(labels, str):
                labels = [labels]
            for label in labels:
                if label:
                    pool.add(label.strip())
        return sorted(pool)

    LABEL_POOLS = {field: build_label_pool(field, gt_data) for field in F1_CATEGORIES}

    # model_resps에는 id가 없으므로 인덱스를 기반으로 매핑
    gt_by_id = {int(d["id"]): d for d in gt_data}
    model_by_id = {i: {"response": d["content"]} for i, d in enumerate(model_resps)}

    eval_ids = sorted(set(gt_by_id) & set(model_by_id))

    # Distractor pool robust하게 string 변환
    def get_field_pool(field):
        vals = []
        for d in gt_data:
            v = d.get(field, "")
            s = extract_str(v)
            if s:
                vals.append(s)
        return vals

    field_pools = {field: get_field_pool(field) for field in ACC_CATEGORIES}

    results = []
    acc_stat = defaultdict(list)
    f1_stat = defaultdict(list)

    for sid in tqdm(eval_ids, desc="Evaluating (LLM-as-Judge)"):
        gt = gt_by_id[sid]
        resp = model_by_id[sid]["response"]

        res_obj = {
            "id": sid,
            "response": resp,
        }

        # ---------- Accuracy (단일선택) ----------
        for field in ACC_CATEGORIES:
            gold = extract_str(gt.get(field, ""))
            pool = [x for x in field_pools[field] if x != gold and x != ""]
            # Distractor robust sampling 개선
            pool = list(set(pool))  # 중복 제거
            pool = [p for p in pool if p.lower() != gold.lower() and p.strip() != ""]
            if len(pool) < 3:
                if len(pool) == 0:
                    distractors = ["(no distractor 1)", "(no distractor 2)", "(no distractor 3)"]
                elif len(pool) == 1:
                    distractors = pool * 3
                elif len(pool) == 2:
                    distractors = pool + [random.choice(pool)]
            else:
                distractors = random.sample(pool, k=3)
            choices = distractors + [gold]
            random.shuffle(choices)
            gold_letter = ["A", "B", "C", "D"][choices.index(gold)]

            acc_prompt = format_acc_prompt(field, choices, resp)
            gpt_response = gpt41mini_chat(acc_prompt)
            pred_letter = extract_acc_letter(gpt_response)
            correct = int(pred_letter == gold_letter)

            res_obj[f"{field}_prompt"] = acc_prompt
            res_obj[f"{field}_choices"] = choices
            res_obj[f"{field}_gpt_response"] = gpt_response
            res_obj[f"{field}_pred_letter"] = pred_letter
            res_obj[f"{field}_gold_letter"] = gold_letter
            res_obj[f"{field}_acc"] = correct
            acc_stat[field].append(correct)

        # ---------- F1 (다중선택) ----------
        for field in F1_CATEGORIES:
            label_pool = LABEL_POOLS[field]
            gold_labels = gt.get(field, [])
            if not isinstance(gold_labels, list):
                gold_labels = [gold_labels]
            f1_prompt = format_f1_prompt(field, label_pool, resp)
            gpt_response = gpt41mini_chat(f1_prompt)
            pred_labels = extract_f1_labels(gpt_response, label_pool)
            y_true = [1 if l in gold_labels else 0 for l in label_pool]
            y_pred = [1 if l in pred_labels else 0 for l in label_pool]
            f1 = f1_score(y_true, y_pred, average="macro") if any(y_true) or any(y_pred) else 1.0

            res_obj[f"{field}_prompt"] = f1_prompt
            res_obj[f"{field}_label_pool"] = label_pool
            res_obj[f"{field}_gold_labels"] = gold_labels
            res_obj[f"{field}_gpt_response"] = gpt_response
            res_obj[f"{field}_pred_labels"] = pred_labels
            res_obj[f"{field}_f1"] = f1
            f1_stat[field].append(f1)

        results.append(res_obj)

    # ----- 결과 저장 -----
    with open(RESULT_PATH, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ----- 집계 결과 출력 -----
    print("\n=== Aggregate Results (LLM-as-Judge) ===")
    print("Accuracy-based categories:")
    for field in ACC_CATEGORIES:
        vals = acc_stat[field]
        if vals:
            acc = sum(vals) / len(vals)
            print(f"  {field:20s}: {acc:.3f}")
    print("F1-based categories:")
    for field in F1_CATEGORIES:
        vals = f1_stat[field]
        if vals:
            f1m = sum(vals) / len(vals)
            print(f"  {field:20s}: Macro F1 = {f1m:.3f}")

    # ----- 결과 테이블 저장 -----
    acc_summary = {field: sum(acc_stat[field]) / len(acc_stat[field]) if acc_stat[field] else 0.0 for field in ACC_CATEGORIES}
    f1_summary = {field: sum(f1_stat[field]) / len(f1_stat[field]) if f1_stat[field] else 0.0 for field in F1_CATEGORIES}
    acc_rows = [{"Text-based": k, "Acc.": round(v, 3), "Categorization": "", "F1": ""} for k, v in acc_summary.items()]
    f1_rows = [{"Text-based": "", "Acc.": "", "Categorization": k, "F1": round(v, 3)} for k, v in f1_summary.items()]
    df = pd.DataFrame(acc_rows + f1_rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2 + len(df) * 0.4))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    plt.savefig("evaluation/accuracy/acc_eval_summary_table_0.5.png", bbox_inches="tight")
    print("\nSaved table to evaluation/accuracy/acc_eval_summary_table_0.5.png")

if __name__ == "__main__":
    main()