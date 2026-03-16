import json

# 파일 경로
input_path = 'data/patient_psi_testset.json'
output_path = 'patient_psi_trainset_with_depression22333222.json'

# intermediate_depression의 기본 값 (원하는 초기값으로 수정 가능)
default_value = ""

# JSON 로딩 및 필드 추가
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    item["intermediate_depression"] = default_value

# 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 완료: '{output_path}' 파일에 intermediate_depression 필드를 추가했습니다.")