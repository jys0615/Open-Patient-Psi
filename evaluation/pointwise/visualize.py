import json
from collections import Counter
import matplotlib.pyplot as plt

# 결과 파일 경로
RESULT_FILE = "evaluation/pointwise/pointwise_gpt4nano_vs_0.5B.jsonl"

# 결과 불러오기
with open(RESULT_FILE, "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]

# 모델별 선택 횟수 집계
winner_counts = Counter(result["winner"] for result in results)

# 시각화
plt.figure(figsize=(12, 10), dpi=600)

plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 20,
    'ytick.labelsize': 12
})
bars = plt.bar(winner_counts.keys(), winner_counts.values(), color='steelblue')

# 막대 위에 수치 표시
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{int(height)}',
             ha='center', va='bottom', fontsize=14)

plt.title("Pointwise Preference Evaluation")
plt.xlabel("Model")
plt.ylabel("Selected Count")
plt.ylim(0, max(winner_counts.values()) * 1.15)  # 여유 공간 확보
plt.tight_layout()
plt.savefig("evaluation/pointwise/pointwise_3B_nano_highres.pdf", dpi=300)
plt.show()