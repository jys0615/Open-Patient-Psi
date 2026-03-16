from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from dotenv import load_dotenv
import os
import json
from pathlib import Path
import openai
import time
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 경로 설정
file_path = Path(__file__).parent.parent.parent / "data/patient_psi_testset.json"

# 평가할 필드 목록
fields_to_evaluate = [
    "situation",
    "automatic_thoughts",
    "intermediate_beliefs",
    "coping_strategies",
    "behaviors"
]

with open(file_path, "r") as f:
    data = json.load(f)

start_time = time.time()
total_fields = sum(1 for item in data for field in fields_to_evaluate if item.get(field, "").strip())
processed_fields = 0

correct_counts = {field: 0 for field in fields_to_evaluate}
total_counts = {field: 0 for field in fields_to_evaluate}

for item in data:
    for field in fields_to_evaluate:
        gt = item.get(field, "").strip()
        if not gt:
            continue

        # distractor 생성 예시 (여기서는 샘플로 단순 반복 제거)
        distractors = [ex.get(field, "") for ex in data if ex.get("id") != item.get("id") and ex.get(field)]
        distractors = list(set(distractors))[:3]
        options = [gt] + distractors
        options = options[:4]  # 최대 4개

        prompt = f"""
                 다음 문장은 환자의 '{field}' 항목입니다.

                 \"{gt}\"

                 아래 보기 중 가장 유사한 항목을 선택하세요:
                 """ + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "당신은 정신건강 전문가로, CBT에 기반한 환자의 사고를 분류하는 역할입니다."},
                    {"role": "user", "content": prompt}
                ]
            )
            choice_text = completion["choices"][0]["message"]["content"]
            chosen_idx = ord(choice_text.strip()[0].upper()) - 65
            if 0 <= chosen_idx < len(options) and options[chosen_idx] == gt:
                correct_counts[field] += 1
            total_counts[field] += 1
            processed_fields += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_fields if processed_fields > 0 else 0
            remaining = total_fields - processed_fields
            est_remaining = remaining * avg_time
            print(f"[{field}] ID {item.get('id')}: {processed_fields}/{total_fields} processed, est. {est_remaining:.1f} sec remaining")
        except Exception as e:
            print(f"Error on field {field}, id {item.get('id')}: {e}")
            continue

print("Accuracy per field:")
for field in fields_to_evaluate:
    total = total_counts[field]
    correct = correct_counts[field]
    accuracy = correct / total if total > 0 else 0.0
    print(f"{field}: {accuracy:.2f}")


print("\n--- Categorization F1 Evaluation ---")

categorization_fields = {
    "core_beliefs": [
        "I am unlovable", "I am undesirable", "I am trapped", "I am out of control", "I am incompetent",
        "I am powerless", "I am immoral", "I am helpless", "I am a failure", "I am bound to be alone",
        "I am bound to be rejected", "I am a victim", "I am worthless", "I am weak", "I am waste",
        "I am not good enough", "I am rejected", "I am insignificant", "I am inadequate"
    ],
    "emotions": [
        "anger", "anxiety", "shame", "sadness", "fear", "despair", "guilt", "rejection", "insecurity", "loneliness"
    ]
}

categorization_preds = {field: [] for field in categorization_fields}
categorization_gts = {field: [] for field in categorization_fields}

for item in data:
    for field, label_space in categorization_fields.items():
        gt_text = item.get(field, "")
        if not gt_text:
            continue
        gt_labels = [label.strip() for label in label_space if label in gt_text.lower()]
        if not gt_labels:
            continue

        prompt = f"""
다음은 환자의 응답입니다. 여기에 해당하는 {field.replace('_', ' ')} 카테고리를 모두 선택하세요.
선택지: {', '.join(label_space)}

\"{gt_text}\"
"""

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "당신은 CBT 전문가입니다. 주어진 텍스트에 따라 모든 관련 카테고리를 선택하세요."},
                    {"role": "user", "content": prompt}
                ]
            )
            response = completion["choices"][0]["message"]["content"]
            pred_labels = [label.strip() for label in label_space if label.lower() in response.lower()]
            categorization_gts[field].append(gt_labels)
            categorization_preds[field].append(pred_labels)
        except Exception as e:
            print(f"Categorization error ({field}) at ID {item.get('id')}: {e}")
            continue

for field in categorization_fields:
    mlb = MultiLabelBinarizer(classes=categorization_fields[field])
    y_true = mlb.fit_transform(categorization_gts[field])
    y_pred = mlb.transform(categorization_preds[field])
    macro_f1 = f1_score(y_true, y_pred, average="macro") if y_true.any() and y_pred.any() else 0.0
    print(f"{field} F1: {macro_f1:.2f}")