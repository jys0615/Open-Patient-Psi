import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import os
import time

# Load fine-tuned model and tokenizer
base_model = "Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS = [4]
LEARNING_RATES = [5e-4]
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Load validation dataset
with open("/data/yoonsuh0615/repos/patientv3/data/patient_psi_testml.jsonl", "r") as f:
    lines = [json.loads(line) for line in f]

total_runs = len(EPOCHS) * len(LEARNING_RATES)
run_count = 0
start_time = time.time()

# --- 전체 예상 실행 시간 및 총 조합 수 출력 ---
print(f"🔄 총 실행 조합 수: {total_runs}개 (EPOCHS x LEARNING_RATES)")

for epoch in EPOCHS:
    for lr in LEARNING_RATES:
        est_per_response_sec = 3.0  # 예시: 하나의 응답 생성에 걸리는 평균 시간(초)
        est_total_responses = total_runs * 200  # 200개 응답씩 x 조합 수
        est_total_time_min = (est_per_response_sec * est_total_responses) / 60
        print(f"🕒 총 예상 실행 시간: 약 {est_total_time_min:.1f}분 (조합당 200개 응답 기준)")
        lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        model_root = f"/data/yoonsuh0615/repos/patientv3/model/0.5B/model/0.5B_EP{epoch}_LR{lr_str}"
        final_path = os.path.join(model_root, "final")
        if os.path.isdir(final_path):
            finetuned_model_path = final_path
        else:
            checkpoints = [d for d in os.listdir(model_root) if d.startswith("checkpoint-")]
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            finetuned_model_path = os.path.join(model_root, latest_checkpoint)

        model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

        output_path = f"/data/yoonsuh0615/repos/patientv3/response/hparam/0.5B_EP{epoch}_LR{lr_str}_valid.jsonl"
        with open(output_path, "w") as out_f:
            max_responses = 200  # 원하는 개수로 조정
            for i, sample in enumerate(lines):
                if i >= max_responses:
                    break
                messages = sample["messages"]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=384,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    generated_ids = output[0][inputs.input_ids.shape[1]:]
                    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

                decoded = decoded.strip()

                if not decoded or decoded in ['""', "''"]:
                    continue
                if all(c in ['\n', '\r', ' '] for c in decoded):
                    continue
                if '"role":' in decoded:
                    continue

                result = {"role": "assistant", "content": decoded}
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        run_count += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / run_count
        remaining = avg_time * (total_runs - run_count)
        print(f"Saved responses to {output_path} ({run_count}/{total_runs}) - approx {remaining/60:.2f} minutes remaining")