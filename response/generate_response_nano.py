import json
from transformers import AutoTokenizer
import torch
import os
import time
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure key is set in environment

# Load fine-tuned model and tokenizer
base_model = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Load validation dataset
with open("./data/patient_psi_testml.jsonl", "r") as f:
    lines = [json.loads(line) for line in f]

output_path = "./response/gpt4nano_response.jsonl"
with open(output_path, "w") as out_f:
    max_responses = 200  # 원하는 개수로 조정
    for i, sample in enumerate(lines):
        if i >= max_responses:
            break
        # testml.jsonl의 messages는 [user, assistant] 쌍인데 assistant는
        # 데이터셋의 정답 응답 슬롯이라 API에 그대로 보내면 안 됨(빈 assistant
        # turn으로 끝나는 비정상 대화가 됨). user 턴만 넘겨서 생성을 요청한다.
        messages = [m for m in sample["messages"] if m["role"] != "assistant"]

        decoded = ""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4.1-nano",
                messages=messages,
                temperature=0.7,
                max_tokens=384
            )
            decoded = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            continue

        if not decoded or decoded in ['""', "''"]:
            continue
        if all(c in ['\n', '\r', ' '] for c in decoded):
            continue
        if '"role":' in decoded:
            continue

        result = {"role": "assistant", "content": decoded}
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
print(f"Saved responses to {output_path}")