import json
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np

# Load evaluation results
result_path = Path("evaluation/fidelity/outputs/gpt4mini_results.jsonl")
with open(result_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Extract fidelity scores
def extract_score(text):
    if not isinstance(text, str):
        return None
    match = re.search(r"\b([1-5])\b", text)
    return int(match.group(1)) if match else None

scores = {"05b": [], "nano": []}
for entry in data:
    model_key = "05b" if "05b" in entry["input_id"] else "nano"
    score = extract_score(entry["response"])
    if score:
        scores[model_key].append(score)

# Compute averages and standard deviations
labels = ["OpenPatientΨ-0.5B", "GPT-4.1-nano"]
averages = [np.mean(scores["05b"]), np.mean(scores["nano"])]
std_devs = [np.std(scores["05b"]), np.std(scores["nano"])]

# Prepare data for line plot
x_pos = [0, 1]
labels_x = ["GPT-4.1-nano", "OpenPatientΨ-0.5B"]

# Extract means and stds in paper order: nano -> 0.5B
means = [averages[1], averages[0]]
stds = [std_devs[1], std_devs[0]]

# Plot lines with improved design
plt.figure(figsize=(5.5, 4.5))
plt.errorbar(x_pos, means, yerr=stds, fmt='o-', color="#1f77b4", linewidth=2.5, capsize=6, markersize=8, label="GPT-4.1-mini eval")

# Annotate points
for x, y, std in zip(x_pos, means, stds):
    plt.text(x, y + 0.5, f"{y:.2f} ± {std:.2f}", ha='center', va='bottom', fontsize=11)

plt.xticks(x_pos, labels_x, fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(3.5, 6) # 그래프 세로 범위
plt.ylabel("Average Fidelity Score", fontsize=12)
plt.title("Overall Fidelity: GPT-4.1-mini Evaluation", fontsize=13, pad=15)

plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.7)
plt.legend(frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig("evaluation/fidelity/outputs/fidelity_lineplot_0.5b.pdf", dpi=300)
plt.show()