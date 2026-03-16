# Open Patient-Ψ

**오픈소스 LLM 기반 정신건강 상담 환자 시뮬레이션 시스템**

> KCC 2025 발표 논문 기반 구현
> Authors: 정윤서, 한승헌, 성무진 (경희대학교 컴퓨터공학부)

---

## 개요

[PATIENT-Ψ (Wang et al., EMNLP 2024)](https://aclanthology.org/) 논문의 구조를 오픈소스 LLM으로 재현·확장한 프로젝트입니다.

기존 Patient-Ψ 시스템은 GPT-4 기반이라 높은 비용, 폐쇄성, 프라이버시 문제가 있었습니다. 이 프로젝트는 **Qwen2.5 (0.5B / 3B)** 모델을 **QLoRA**로 파인튜닝하여 GPT-4.1-nano에 근접한 환자 시뮬레이션 응답 품질을 달성합니다.

---

## 프로젝트 구조

```
Open-Patient-Psi/
├── data/                          # 데이터셋 생성 및 전처리
│   ├── data_generation.py         # GPT-4-Turbo 기반 CCD 데이터 생성
│   ├── chatml_generation.py       # ChatML 형식 변환
│   ├── testml_generation.py       # 테스트셋 변환
│   ├── response.py                # 응답 생성 스크립트
│   ├── patient_psi_trainset.json  # 학습 데이터 (1000개)
│   ├── patient_psi_validset.json  # 검증 데이터 (100개)
│   ├── patient_psi_testset.json   # 테스트 데이터 (200개)
│   ├── patient_psi_chatml.jsonl   # 학습용 ChatML 형식
│   ├── patient_psi_validml.jsonl  # 검증용 ChatML 형식
│   └── patient_psi_testml.jsonl   # 테스트용 ChatML 형식
│
├── model/                         # 모델 학습
│   ├── 0.5B/
│   │   ├── train_0.5b.py          # Qwen2.5-0.5B QLoRA 학습
│   │   └── run_psi0.5b.sh         # 학습 실행 스크립트
│   └── 3B/
│       ├── train_3b.py            # Qwen2.5-3B QLoRA 학습
│       ├── run_psi3b.sh           # 학습 실행 스크립트
│       └── model/                 # 학습된 체크포인트 저장 디렉토리
│
├── response/                      # 모델 응답 생성
│   ├── generate_response.py       # Qwen 모델 응답 생성
│   ├── generate_response_nano.py  # GPT-4.1-nano 응답 생성
│   ├── response_hp.sh             # 하이퍼파라미터 탐색용 응답 생성 스크립트
│   └── hparam/                    # 하이퍼파라미터별 검증 응답 (25개 조합)
│
├── evaluation/                    # 평가
│   ├── accuracy/                  # 정확도 평가 (텍스트 정확도 + F1)
│   ├── fidelity/                  # 충실도 평가 (GPT-4.1-mini 기반)
│   └── pointwise/                 # 포인트와이즈 비교 평가
│
└── [KCC25]Open Patient-Ψ.pdf     # KCC 2025 발표 논문
```

---

## 방법론

### 데이터셋 구축

Patient-Ψ 논문의 **Cognitive Conceptualization Diagram (CCD)** 구조를 따라 총 1300개 샘플 생성:

| 항목 | 내용 |
|------|------|
| 생성 도구 | GPT-4-Turbo |
| 상황 범주 | family dynamics, workplace pressure, relationship dynamics 등 7가지 |
| 핵심 신념 | helpless, unlovable, worthless (CBT 기반) |
| 발화 스타일 | upset, reserved, plain, verbose, tangent, pleasing |
| 데이터 분할 | 학습 1000개 / 검증 100개 / 테스트 200개 |

### 모델 학습 (QLoRA Fine-tuning)

| 설정 | 값 |
|------|-----|
| 베이스 모델 | Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct |
| 양자화 | 4-bit NF4 (BitsAndBytes) |
| LoRA rank | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| 최대 시퀀스 길이 | 1024 |

### 하이퍼파라미터 탐색

학습률 5가지 × 에폭 수 5가지 = **총 25개 조합** 탐색, BERTScore (roberta-large) 기반으로 자동 선정:

| 모델 | 최적 에폭 | 최적 학습률 |
|------|-----------|-------------|
| 0.5B | 2 | 4e-4 |
| 3B | 6 | 5e-4 |

---

## 평가 결과

### Pointwise 비교 평가 (vs. GPT-4.1-nano)

평가 기준: 정서적 현실감 / 심리 프로파일 정합성 / CBT 맥락 적절성

**두 모델 모두 GPT-4.1-nano 대비 더 높은 선택률 기록**

### 정확도 평가 (텍스트 기반 정확도 / F1)

| 항목 | 0.5B | 3B |
|------|------|----|
| 상황 (텍스트 정확도) | 0.18 | 0.22 |
| 대처 전략 | 0.225 | 0.245 |
| 중간 신념 | 0.26 | 0.225 |
| 자동적 사고 | 0.22 | 0.215 |
| 행동 | 0.24 | 0.22 |
| 핵심 신념 (F1) | 0.398 | 0.391 |
| 감정 (F1) | 0.342 | 0.342 |
| 핵심 신념 세분화 (F1) | 0.377 | 0.375 |

### 충실도 평가 (Fidelity, 1–6점)

GPT-4.1-nano가 근소하게 우위 (GPT 계열 모델 간 평가 편향으로 해석)

---

## 환경 설정

### 의존성 설치

```bash
pip install torch transformers datasets peft bitsandbytes
pip install openai python-dotenv bert-score
```

### 환경 변수

`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```
OPENAI_API_KEY=your_api_key_here
```

---

## 실행 방법

### 1. 데이터 생성

```bash
# CCD 기반 환자 시뮬레이션 데이터 생성
python data/data_generation.py

# ChatML 형식으로 변환
python data/chatml_generation.py
```

### 2. 모델 학습

```bash
# 0.5B 모델 학습
bash model/0.5B/run_psi0.5b.sh

# 3B 모델 학습
bash model/3B/run_psi3b.sh
```

### 3. 응답 생성

```bash
# 파인튜닝 모델 응답 생성
python response/generate_response.py

# GPT-4.1-nano 응답 생성
python response/generate_response_nano.py
```

### 4. 평가

```bash
# 정확도 평가
python evaluation/accuracy/run_accuracy_eval.py

# 충실도 평가
python evaluation/fidelity/evaluate_fidelity_gpt4mini.py

# 포인트와이즈 비교 평가
python evaluation/pointwise/pointwise_evaluation.py
```

---

## 향후 연구

- **RLAIF**: Self-Rater 기반 AI 피드백을 활용한 강화학습 적용으로 정합성 향상
- **멀티 참조 평가**: 평가 편향 보정을 위한 다중 참조 구조 도입
- **모델 확장**: 더 큰 오픈소스 모델 (7B, 14B 등) 적용

---

## 참고 문헌

```
[1] Wang, R., et al., "PATIENT-Ψ: Using Large Language Models to Simulate Patients
    for Training Mental Health Professionals", EMNLP, 2024
[2] Zhang, T., et al., "BERTScore: Evaluating Text Generation with BERT", ICLR, 2020
[3] Dettmers, T., et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS, 2023
[4] Yang A., et al., "Qwen2.5 Technical Report", arXiv:2412.15115, 2024
```

---

## 지원

본 연구는 과학기술정보통신부 및 정보통신기획평가원의 2025년도 SW중심대학사업 (2023-0-00042) 지원을 받아 수행되었습니다.
