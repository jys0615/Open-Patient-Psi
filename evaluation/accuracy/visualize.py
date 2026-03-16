import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Load the accuracy evaluation results
file_path = "evaluation/accuracy/acc_eval_3B.jsonl"  # ← 경로를 상황에 맞게 바꾸세요
with open(file_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

# 정의된 평가 항목
acc_fields = ["situation", "coping_strategies", "intermediate_beliefs", "automatic_thoughts", "behaviors"]
f1_fields = ["core_beliefs", "emotions", "behaviors"]
acc_scores = defaultdict(list)
f1_scores = defaultdict(list)

# 점수 집계
for record in records:
    for field in acc_fields:
        acc_scores[field].append(record.get(f"{field}_acc", 0))
    for field in f1_fields:
        f1_scores[field].append(float(record.get(f"{field}_f1", 0.0)))

# 평균 계산
acc_means = {field: sum(vals)/len(vals) for field, vals in acc_scores.items()}
f1_means = {field: sum(vals)/len(vals) for field, vals in f1_scores.items()}

# 논문 형식 테이블 구성
table_data = {
    "Text-based": ["Situation", "Coping strategies", "Intermediate beliefs", "Automatic thoughts", "Behaviors"],
    "Acc.": [round(acc_means[f], 2) for f in acc_fields],
    "Categorization": ["Core beliefs", "Emotions", "Core beliefs(fine-grained)", "", ""],
    "F1": [
        round(f1_means.get("core_beliefs", 0), 2),
        round(f1_means.get("emotions", 0), 2),
        round(f1_means.get("core_beliefs", 0), 2),
        "", ""
    ]
}
df = pd.DataFrame(table_data)

# 시각화
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis('off')
tbl = Table(ax, bbox=[0, 0, 1, 1])
n_rows, n_cols = df.shape
col_labels = df.columns.tolist()
widths = [0.25, 0.15, 0.35, 0.15]
height = 1.0 / (n_rows + 1)

# 헤더 셀
for i, label in enumerate(col_labels):
    cell = tbl.add_cell(0, i, widths[i], height, text=label, loc='center', facecolor='#40466e')
    cell.get_text().set_color('white')
    cell.get_text().set_weight('bold')

# 데이터 셀
for row in range(n_rows):
    for col in range(n_cols):
        val = df.iloc[row, col]
        tbl.add_cell(row + 1, col, widths[col], height, text=str(val), loc='center', facecolor='white')

tbl.set_fontsize(12)
ax.add_table(tbl)
plt.tight_layout()
plt.savefig("acc_eval_summary_table_3b.png", dpi=2000)  # 또는 plt.show()