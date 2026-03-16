import json
import random
from sklearn.metrics import accuracy_score, f1_score
import os 
from dotenv import load_dotenv
load_dotenv()
import openai
import pandas as pd

RESPONSE_PATH = "response/0.5B_EP4_LR5e-4_test.jsonl"
GROUND_TRUTH_PATH = "data/patient_psi_testset.json"

openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. Define evaluation categories
TEXT_BASED_FIELDS = ["situation", "coping_strategies", "intermediate_beliefs", "automatic_thoughts", "behaviors"]
CATEGORIZATION_FIELDS = ["core_beliefs", "emotions"]
FINE_GRAINED_FIELD = "core_beliefs"

# 2. Load response and ground truth
with open(RESPONSE_PATH, "r") as f:
    responses = [json.loads(line) for line in f]

with open(GROUND_TRUTH_PATH, "r") as f:
    ground_truth = {item["id"]: item for item in json.load(f)}

# 3. GPT-4 API call
def query_gpt4(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 정확하게 심리학 개념을 판단하는 평가자야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message['content'].strip()[0]

# 4. Prompt generation function
def generate_prompt(field, response_text, choices):
    return (
        f"응답: {response_text}\n\n심리 요소를 선택하세요:\n"
        + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        + f"\n\n정답은? (A-{chr(64+len(choices))})"
    )

# 5. Run evaluation
results = {field: {"preds": [], "labels": []} for field in TEXT_BASED_FIELDS + CATEGORIZATION_FIELDS + [FINE_GRAINED_FIELD]}

for i, sample in enumerate(responses):
    response_text = sample["content"]
    rid = str(i + 1)
    gt = ground_truth.get(rid) or ground_truth.get(int(rid)) if rid.isdigit() else None
    if gt is None:
        continue

    # Evaluate text-based fields
    for field in TEXT_BASED_FIELDS:
        label = gt[field].strip()
        pool = list(set([x[field] for x in ground_truth.values() if str(x["id"]) != rid]))
        if label in pool:
            pool.remove(label)
        distractors = random.sample(pool, min(3, len(pool)))
        choices = random.sample([label] + distractors, len([label] + distractors))
        prompt = generate_prompt(field, response_text, choices)
        pred = query_gpt4(prompt)
        index = ord(pred.upper()) - 65
        pred_label = choices[index] if 0 <= index < len(choices) else random.choice(choices)
        results[field]["preds"].append(pred_label)
        results[field]["labels"].append(label)

    # Evaluate categorization fields and fine-grained field
    for field in CATEGORIZATION_FIELDS + [FINE_GRAINED_FIELD]:
        label = gt[field].strip()
        pool = list(set([x[field] for x in ground_truth.values() if str(x["id"]) != rid]))
        if label in pool:
            pool.remove(label)
        distractors = random.sample(pool, min(3, len(pool)))
        choices = random.sample([label] + distractors, len([label] + distractors))
        prompt = generate_prompt(field, response_text, choices)
        pred = query_gpt4(prompt)
        index = ord(pred.upper()) - 65
        pred_label = choices[index] if 0 <= index < len(choices) else random.choice(choices)
        results[field]["preds"].append(pred_label)
        results[field]["labels"].append(label)

# 6. Compute scores
print("==== TEXT-BASED ACCURACY ====")
for field in TEXT_BASED_FIELDS:
    acc = accuracy_score(results[field]["labels"], results[field]["preds"])
    print(f"{field}: Accuracy = {acc:.3f}")

print("\n==== CATEGORIZATION F1 ====")
for field in CATEGORIZATION_FIELDS + [FINE_GRAINED_FIELD]:
    f1 = f1_score(results[field]["labels"], results[field]["preds"], average="macro")
    print(f"{field}: F1 = {f1:.3f}")

records = []
for field in TEXT_BASED_FIELDS:
    acc = accuracy_score(results[field]["labels"], results[field]["preds"])
    records.append({"Field": field, "Metric": "Accuracy", "Score": round(acc, 3)})

for field in CATEGORIZATION_FIELDS + [FINE_GRAINED_FIELD]:
    f1 = f1_score(results[field]["labels"], results[field]["preds"], average="macro")
    records.append({"Field": field, "Metric": "F1", "Score": round(f1, 3)})

df = pd.DataFrame(records)
df.to_csv("evaluation_results.csv", index=False)