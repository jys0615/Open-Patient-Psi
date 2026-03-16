import os
import json
import time
from dotenv import load_dotenv
import openai
from sklearn.metrics import f1_score

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CATEGORIES = {
    "core_beliefs": [
        "unlovable", "worthless", "inadequate", "helpless", "defective", "bad", "inferior",
        "unwanted", "incompetent", "guilty", "powerless", "weak", "vulnerable", "rejected",
        "abandoned", "unimportant", "hopeless", "ashamed", "alone"
    ],
    "emotions": [
        "angry", "anxious", "sad", "happy", "fearful", "disgusted", "surprised", "guilty", "ashamed"
    ],
    "behaviors": [
        "avoidance", "withdrawal", "aggression", "procrastination", "perfectionism", "self-harm",
        "substance_use", "overeating", "isolation"
    ],
    "coping_strategies": [
        "problem_solving", "seeking_support", "positive_reappraisal", "denial", "distraction",
        "rumination", "acceptance", "suppression", "avoidance"
    ],
    "intermediate_beliefs": [
        "if_then_rules", "conditional_rules", "attitudes", "assumptions", "rules_for_living",
        "expectations", "standards", "beliefs_about_self", "beliefs_about_others"
    ]
}

def parse_response(text):
    # Split by comma and strip whitespace, convert to lowercase
    return [label.strip().lower() for label in text.split(",") if label.strip()]

def evaluate_category(category, labels, data):
    y_true = []
    y_pred = []
    predictions = []

    start_time = time.time()
    print(f"Evaluating category '{category}' with {len(data)} samples...")

    for idx, sample in enumerate(data):
        content = sample["content"]
        true_labels = [label.lower() for label in sample.get(category, [])]

        prompt = (
            f"다음 대화 내용에서 {category}에 해당하는 항목을 모두 선택하세요. 가능한 항목: {', '.join(labels)}. "
            f"응답은 쉼표로 구분된 영어 단어 리스트로 작성하세요. 예시: sad, angry\n"
            f"대화 내용: \"{content}\""
        )

        try:
            response = openai.api_key.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            output = response.choices[0].message.content
            pred_labels = parse_response(output)
        except Exception as e:
            print(f"OpenAI API call failed at sample {idx} for category '{category}': {e}")
            pred_labels = []

        # For exact match accuracy, sort and compare sets
        y_true.append(sorted(true_labels))
        y_pred.append(sorted(pred_labels))

        predictions.append({
            "id": sample["id"],
            "content": content,
            "true_labels": true_labels,
            "pred_labels": pred_labels
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(data):
            elapsed = time.time() - start_time
            print(f"  Processed {idx+1}/{len(data)} samples in {elapsed:.1f}s")

    # Compute exact match accuracy
    exact_matches = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
    accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0

    # For F1, create multilabel binary indicator matrix
    all_labels = labels
    def to_binary_matrix(label_lists):
        matrix = []
        for label_list in label_lists:
            row = [1 if label in label_list else 0 for label in all_labels]
            matrix.append(row)
        return matrix

    y_true_bin = to_binary_matrix(y_true)
    y_pred_bin = to_binary_matrix(y_pred)

    f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)

    return accuracy, f1, predictions

def main():
    data_path = os.path.join("evaluation", "accuracy", "gpt4mini_f1_eval.jsonl")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = {}
    all_predictions = []

    for category, labels in CATEGORIES.items():
        accuracy, f1, preds = evaluate_category(category, labels, data)
        results[category] = {
            "accuracy": accuracy,
            "f1_macro": f1
        }
        all_predictions.extend([{"category": category, **p} for p in preds])

    output_path = os.path.join("evaluation", "accuracy", "gpt4mini_f1_eval_results.jsonl")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for pred in all_predictions:
            f_out.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print("\nSummary of evaluation:")
    print(f"{'Category':20s} {'Accuracy':>10s} {'Macro F1':>10s}")
    print("-" * 44)
    for category, scores in results.items():
        print(f"{category:20s} {scores['accuracy']*100:9.2f}% {scores['f1_macro']*100:9.2f}%")

if __name__ == "__main__":
    main()
