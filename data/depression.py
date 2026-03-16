import json
import openai
import openai
import os
from dotenv import load_dotenv
import time
start_time = time.time()
# 파일 경로
input_path = 'data/patient_psi_testset.json'
output_path = 'patient_psi_trainset_with_depressiontext555111331.json'

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# JSON 로딩 및 필드 추가
with open(input_path, 'r', encoding='utf-8') as f:
    obj = json.load(f)
    # If the loaded object is a dict (single item), wrap it in a list
    if isinstance(obj, dict):
        data = [obj]
    elif isinstance(obj, list):
        data = obj
    else:
        raise ValueError("Unsupported JSON format: Expected object or list.")
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        data = [json.loads(item) for item in data]

# Ensure all items are dicts (in case some are still strings)
cleaned_data = []
for item in data:
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except json.JSONDecodeError:
            continue  # skip if item cannot be parsed
    cleaned_data.append(item)
data = cleaned_data
print("✅ 데이터 로드 완료. 총 항목 수:", len(data))
print("샘플 항목:", data[0] if data else "없음")

for item in data:
    prompt = (
        "You are a clinical assistant trained in cognitive behavioral therapy.\n"
        "Your task is to transform the patient's intermediate beliefs into two depressive-style beliefs.\n\n"
        "These two beliefs should each reflect the following:\n"
        "1. The idea that avoiding challenges or risks provides temporary safety, but trying leads to failure, embarrassment, or distress.\n"
        "2. The idea that hiding one's weaknesses helps maintain self-worth or avoid rejection, but revealing them leads to being seen as incompetent, flawed, or unworthy.\n\n"
        "Original Intermediate Beliefs:\n"
        f"{item.get('intermediate_beliefs', '')}\n\n"
        "Return exactly two depressive-style beliefs (one per condition), written naturally as standalone sentences without numbering or labels. Focus on the underlying cognitive distortion and emotional impact."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that rewrites cognitive beliefs in a depressive context."},
            {"role": "user", "content": prompt}
        ]
    )
    raw = response["choices"][0]["message"]["content"].strip()
    cleaned = raw.replace("Depressive Beliefs:", "").replace("During Depression:", "")
    cleaned = cleaned.replace("\n\n", " ").replace("\n", " ").replace("1.", "").replace("2.", "")
    item["intermediate_depression"] = cleaned.strip()
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"⏱️ 소요 시간: {minutes}분 {seconds}초")

# 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

elapsed = time.time() - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"⏱️ 소요 시간: {minutes}분 {seconds}초")

print(f"✅ 완료: '{output_path}' 파일에 intermediate_depression 필드를 추가했습니다.")