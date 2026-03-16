import json
from sklearn.metrics import f1_score
from tqdm import tqdm
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

def evaluate_with_gpt(response: str, category: str) -> str:
    prompt = f"""
    아래는 환자 시뮬레이션 응답입니다. 이 응답에서 가장 적절한 '{category}' 항목 값을 한 줄로 출력하세요.
    다른 설명이나 서술은 포함하지 마세요. 정답처럼 간결히 출력하세요.

    응답:
    {response}
    """
    chat = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return chat.choices[0].message['content'].strip().lower()

# 파일 경로 설정
GT_PATH = "data/patient_psi_testset.json"
PRED_PATH = "response/0.5B_EP4_LR5e-4_test.jsonl"

# 평가 항목 정의
ACC_CATEGORIES = ["situation", "coping_strategies", "intermediate_beliefs", "automatic_thoughts", "behaviors"]
F1_CATEGORIES = ["core_beliefs", "emotions", "fine_grained_core_beliefs"]

import sys
import time

start_time = time.time()
total_tasks = len(ACC_CATEGORIES) + len(F1_CATEGORIES)
current_task = 0

print("🕒 Evaluation started...")
print(f"⏳ Estimated total time: Please wait... Evaluating {total_tasks} fields using GPT-4.")
sys.stdout.flush()

# GT 로드
with open(GT_PATH, "r") as f:
    gt_data = {str(sample["id"]): sample for sample in json.load(f)}

# 예측 로드 (ChatML 응답)
predictions = {}
with open(PRED_PATH, "r") as f:
    for idx, line in enumerate(f):
        obj = json.loads(line)
        id_ = str(idx)
        predictions[id_] = obj["content"]  # ← 'response' 대신 'content' 사용

# Accuracy 평가
print("=== Accuracy-based Evaluation ===")
for category in ACC_CATEGORIES:
    correct = 0
    total = 0
    for id_, gt in gt_data.items():
        if id_ not in predictions:
            continue
        response = predictions[id_]
        pred = evaluate_with_gpt(response, category)
        answer = gt.get(category, "").strip().lower()
        if pred == answer:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    print(f"{category:<25}: {acc:.3f}")

# F1 평가
print("\n=== F1-based Evaluation ===")
FINE_GRAINED_MAP = {
    "i am incompetent": "i am incompetent",
    "i am helpless": "i am helpless",
    "i am powerless": "i am powerless, weak, vulnerable",
    "i am weak": "i am powerless, weak, vulnerable",
    "i am vulnerable": "i am powerless, weak, vulnerable",
    "i am a victim": "i am a victim",
    "i am needy": "i am needy",
    "i am trapped": "i am trapped",
    "i am out of control": "i am out of control",
    "i am a failure": "i am a failure, loser",
    "i am a loser": "i am a failure, loser",
    "i am defective": "i am defective",
    "i am unlovable": "i am unlovable",
    "i am unattractive": "i am unattractive",
    "i am undesirable": "i am undesirable, unwanted",
    "i am unwanted": "i am undesirable, unwanted",
    "i am bound to be rejected": "i am bound to be rejected",
    "i am bound to be abandoned": "i am bound to be abandoned",
    "i am bound to be alone": "i am bound to be alone",
    "i am worthless": "i am worthless, waste",
    "i am waste": "i am worthless, waste",
    "i am immoral": "i am immoral",
    "i am bad": "i am bad - dangerous, toxic, evil",
    "i am dangerous": "i am bad - dangerous, toxic, evil",
    "i am toxic": "i am bad - dangerous, toxic, evil",
    "i am evil": "i am bad - dangerous, toxic, evil",
    "i don't deserve to live": "i don’t deserve to live"
}

for category in F1_CATEGORIES:
    y_true = []
    y_pred = []
    for id_, gt in gt_data.items():
        if id_ not in predictions:
            continue
        response = predictions[id_]
        pred_str = evaluate_with_gpt(response, category)
        if category == "fine_grained_core_beliefs":
            # Normalize ground truth
            raw_gt = {x.strip().lower() for x in gt.get(category, "").split(".") if x.strip()}
            gt_set = {FINE_GRAINED_MAP.get(gt, gt) for gt in raw_gt}
            # Normalize prediction
            raw_preds = {x.strip().lower() for x in pred_str.split(".") if x.strip()}
            pred_set = {FINE_GRAINED_MAP.get(pred, pred) for pred in raw_preds}
        else:
            gt_set = {x.strip().lower() for x in gt.get(category, "").split(".") if x.strip()}
            pred_set = {x.strip().lower() for x in pred_str.split(".") if x.strip()}
        all_labels = sorted(list(gt_set.union(pred_set)))
        y_true.append([1 if label in gt_set else 0 for label in all_labels])
        y_pred.append([1 if label in pred_set else 0 for label in all_labels])
    macro_f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
    print(f"{category:<25}: Macro F1 = {macro_f1:.3f}")