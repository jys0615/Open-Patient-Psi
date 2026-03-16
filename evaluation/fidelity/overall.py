import json
from pathlib import Path

# 파일 경로
path_05b = Path("response/0.5B_EP4_LR5e-4_test.jsonl")
nano_path = Path("response/gpt4nano_response.jsonl")
output_path = Path("evaluation/fidelity/inputs/overall_0.5b.jsonl")

# JSONL 로드 함수
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# 파일 읽기
model_05b_data = load_jsonl(path_05b)
nano_data = load_jsonl(nano_path)

# Fidelity 평가 입력 생성
fidelity_inputs = []
for i, (model_05b, nano) in enumerate(zip(model_05b_data, nano_data)):
    conversation = model_05b.get("conversation") or nano.get("conversation")
    fid_05b = {
        "input_id": f"ex_{i:03}_0.5b",
        "conversation": conversation,
        "model_response": model_05b["content"],
        "dimension": "overall"
    }
    fid_nano = {
        "input_id": f"ex_{i:03}_nano",
        "conversation": conversation,
        "model_response": nano["content"],
        "dimension": "overall"
    }
    fidelity_inputs.extend([fid_05b, fid_nano])

# JSONL로 저장
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for item in fidelity_inputs:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"Saved to: {output_path}")