import json
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 설정
EVAL_MODEL = "gpt-4.1-mini"  # GPT-4.1-mini
NANO_RESP_FILE = "response/gpt4nano_response.jsonl"
SFT_RESP_FILE = "response/0.5B_EP4_LR5e-4_test.jsonl"
OUTPUT_FILE = "evaluation/pointwise/pointwise_gpt4nano_vs_0.5B.jsonl"

# 평가 프롬프트 템플릿
SYSTEM_PROMPT = "You are a careful evaluator comparing two model responses to the same therapeutic prompt. Choose the more helpful, empathetic, and realistic patient response."

EVAL_PROMPT = """\
            Compare the two responses from different models to the same therapeutic instruction.
            Your task is to judge which response is better in terms of the following criteria:
            - Emotional realism
            - Alignment with the patient's background and psychological profile
            - Appropriateness in a CBT therapy context

            You must reply with the better response: "A" or "B".
            Do not explain your reasoning unless asked. Just return "A" or "B".

            # Context (therapist prompt):
            {prompt}

            # Response A:
            {resp_a}

            # Response B:
            {resp_b}

            Which is better?
            """

# 파일 불러오기
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

nano_data = load_jsonl(NANO_RESP_FILE)
sft_data = load_jsonl(SFT_RESP_FILE)

assert len(nano_data) == len(sft_data), "❌ 응답 수가 일치하지 않음"

TEST_FILE = "data/patient_psi_testml.jsonl"
test_data = load_jsonl(TEST_FILE)

results = []

for i in tqdm(range(len(test_data)), desc="Evaluating responses"):
    prompt = test_data[i]["messages"][-2]["content"]
    nano_resp = nano_data[i]["content"]
    sft_resp = sft_data[i]["content"]

    eval_input = EVAL_PROMPT.format(prompt=prompt, resp_a=nano_resp, resp_b=sft_resp)

    response = openai.ChatCompletion.create(
        model=EVAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": eval_input}
        ],
        temperature=0,
    )

    choice = response["choices"][0]["message"]["content"].strip()
    results.append({
        "response_A": nano_resp,
        "response_B": sft_resp,
        "winner": "GPT-4.1-nano" if choice == "A" else "OpenPatientΨ-0.5B"
    })

# 저장
with open(OUTPUT_FILE, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ 평가 완료: {OUTPUT_FILE}")