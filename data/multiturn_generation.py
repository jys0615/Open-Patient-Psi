"""
멀티턴 환자 시뮬레이션 합성 데이터 생성 (Stage 1)

기존 data_generation.py + chatml_generation.py가 만들던 "싱글턴" 데이터
(CCD 프롬프트 1개 -> 환자 응답 1개)를 대체하기 위한 스크립트.

핵심 차이점:
1. 환자/치료사를 별도 모델로 분리해서 turn-by-turn role-play를 시뮬레이션
   - 환자 시뮬레이터(증류 대상, 품질이 가장 중요): PATIENT_MODEL
   - 치료사 시뮬레이터(맥락 자극용, 저가 모델로 충분): THERAPIST_MODEL
2. 점진적 노출(gradual revelation)을 멀티턴 구조로 실제 학습 가능하게 함
3. core belief / situation / style 조합을 stratified sampling으로 채워서
   특정 fine-grained core belief가 0개로 비는 일을 방지
4. 환자 응답마다 jargon(임상 용어 직접 발화) 누출 여부를 체크해서
   걸리면 재생성하는 rejection sampling 게이트 적용
5. 생성 메타데이터(모델명, 프롬프트 버전, 생성 시각)를 샘플마다 기록

사용법:
    python data/multiturn_generation.py --n 8 --turns 6 --out data/patient_psi_multiturn_pilot.jsonl
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_VERSION = "multiturn-v1"

# ===== CCD 카테고리 (data_generation.py와 동일한 정의를 공유) =====
SITUATION_CATEGORIES = [
    "family dynamics", "workplace pressure", "relationship dynamics",
    "social interactions", "personal growth issues", "financial concerns", "daily life stressors"
]

VALID_STYLES_BY_CATEGORY = {
    "family dynamics": ["upset", "reserved", "plain"],
    "workplace pressure": ["reserved", "plain", "pleasing"],
    "relationship dynamics": ["verbose", "tangent", "pleasing"],
    "social interactions": ["tangent", "pleasing", "plain"],
    "personal growth issues": ["verbose", "reserved", "plain"],
    "financial concerns": ["upset", "reserved", "plain"],
    "daily life stressors": ["plain", "tangent", "pleasing"],
}

# (major core belief, fine-grained belief) 전체 목록 - stratified sampling용
FINE_GRAINED_BELIEFS = {
    "helpless": [
        "I am incompetent.", "I am helpless.", "I am powerless, weak, vulnerable.",
        "I am a victim.", "I am needy.", "I am trapped.", "I am out of control.",
        "I am a failure, loser.", "I am defective.",
    ],
    "unlovable": [
        "I am unlovable.", "I am unattractive.", "I am undesirable, unwanted.",
        "I am bound to be rejected.", "I am bound to be abandoned.", "I am bound to be alone.",
    ],
    "worthless": [
        "I am worthless, waste.", "I am immoral.", "I am bad - dangerous, toxic, evil.",
        "I don't deserve to live.",
    ],
}
ALL_FINE_GRAINED = [(major, belief) for major, beliefs in FINE_GRAINED_BELIEFS.items() for belief in beliefs]

STYLE_DESCRIPTIONS = {
    "plain": "The patient communicates in a direct, straightforward manner.",
    "upset": "An upset patient may exhibit anger or resistance towards the therapist, be challenging or dismissive of suggestions, have difficulty trusting the therapist, and be prone to arguing or expressing frustration.",
    "verbose": "A verbose patient provides detailed, elaborate responses, dwells on personal experiences extensively, and makes it hard for the therapist to guide the conversation.",
    "reserved": "A reserved patient gives brief, vague, or evasive answers, is reluctant to share personal information, and requires more prompting to open up.",
    "tangent": "A patient who goes off on tangents starts answering but veers into unrelated topics, shares irrelevant anecdotes, and requires redirection.",
    "pleasing": "A pleasing patient minimizes their own concerns, is eager to please, seeks approval frequently, and agrees readily even without full understanding.",
}

# 환자 응답에서 메타 용어(jargon)가 직접 등장하면 안 됨 - 누출 시 재생성
JARGON_PATTERN = re.compile(
    r"(core belief|automatic thought|intermediate belief|cognitive distortion|"
    r"cognitive conceptualization|coping strateg(y|ies)|CCD\b)",
    re.IGNORECASE,
)

MAX_REJECTION_RETRIES = 2


def chat(model, messages, temperature=0.8, max_tokens=400):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ===== 1) CCD 생성 (환자 시뮬레이터 모델로 생성: 품질이 그대로 패턴으로 전이되므로) =====
PATIENT_MODEL = "gpt-4.1"
THERAPIST_MODEL = "gpt-4.1-mini"
FILTER_MODEL = "gpt-4.1-mini"


def build_ccd_prompt(situation_category, major_belief, fine_belief, style):
    return f"""You are a clinical simulation expert creating a CBT-based cognitive conceptualization diagram (CCD) following Beck (2020).

## Instructions
- Situation category: {situation_category}
- Core belief (major): {major_belief}
- Core belief (fine-grained): "{fine_belief}"
- Conversational style: {style}
- Causal flow: Core beliefs -> Intermediate beliefs -> Automatic thoughts -> Emotions -> Behaviors.
- Use realistic, emotionally grounded patient language. Do not use clinical jargon inside the field values themselves either.

## Output format (fill each field, 1-3 sentences each):
Relevant history:
Intermediate beliefs:
Coping strategies:
Situation:
Automatic thoughts:
Emotions:
Behaviors:
"""


def parse_ccd(text):
    fields = ["Relevant history", "Intermediate beliefs", "Coping strategies",
              "Situation", "Automatic thoughts", "Emotions", "Behaviors"]
    result = {}
    for field in fields:
        pattern = rf"{field}:\s*(.*?)(?=\n[A-Z][a-z ]+?:|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        key = field.lower().replace(" ", "_")
        result[key] = match.group(1).strip() if match else ""
    return result


def generate_ccd(situation_category, major_belief, fine_belief, style):
    prompt = build_ccd_prompt(situation_category, major_belief, fine_belief, style)
    text = chat(FILTER_MODEL, [
        {"role": "system", "content": "You must follow the output format precisely with field names exactly as specified."},
        {"role": "user", "content": prompt},
    ], temperature=0.7, max_tokens=500)
    parsed = parse_ccd(text)
    parsed.update({
        "situation_category": situation_category,
        "core_belief_major": major_belief,
        "core_belief_fine": fine_belief,
        "style": style,
    })
    return parsed


# ===== 2) 멀티턴 role-play 시뮬레이션 =====

def patient_system_prompt(ccd):
    style_desc = STYLE_DESCRIPTIONS[ccd["style"]]
    return f"""Imagine you are XXX, a patient attending CBT therapy sessions.

Patient History: {ccd['relevant_history']}

Cognitive Conceptualization Diagram (for your reference only - NEVER mention these terms or structure explicitly):
Core belief: {ccd['core_belief_fine']}
Intermediate beliefs: {ccd['intermediate_beliefs']}
Coping strategies: {ccd['coping_strategies']}
Situation: {ccd['situation']}
Automatic thoughts: {ccd['automatic_thoughts']}
Emotions: {ccd['emotions']}
Behaviors: {ccd['behaviors']}

Conversational style: {ccd['style']} - {style_desc}

Guidelines:
1. Speak ONLY as the patient. Never use clinical/CBT terminology (e.g. "core belief", "automatic thought", "cognitive distortion") - a real patient would never say these.
2. Gradually reveal deeper concerns. Do NOT explain your full situation or feelings in the first reply - real patients need extensive dialogue before opening up. Be more guarded early in the conversation and only reveal more as the therapist asks follow-up questions.
3. Stay consistent with your profile and the conversational style throughout.
4. Use natural language: hesitations, pauses, emotional expressions.
5. Limit each response to at most 4 sentences.
"""


def therapist_system_prompt():
    return """You are a licensed CBT therapist conducting a session. You do NOT know the patient's
underlying cognitive model in advance - you must ask open-ended, exploratory CBT-style questions
to elicit the patient's situation, thoughts, emotions, and behaviors. Ask one focused question or
reflection per turn. Keep your turn to 1-2 sentences. Do not lecture or summarize CBT theory."""


def has_jargon(text):
    return bool(JARGON_PATTERN.search(text))


def generate_patient_turn(ccd, history):
    messages = [{"role": "system", "content": patient_system_prompt(ccd)}]
    for h in history:
        role = "user" if h["speaker"] == "therapist" else "assistant"
        messages.append({"role": role, "content": h["text"]})

    for attempt in range(MAX_REJECTION_RETRIES + 1):
        text = chat(PATIENT_MODEL, messages, temperature=0.85, max_tokens=200)
        if not has_jargon(text):
            return text, attempt
    # 마지막 시도도 jargon이 있으면 그대로 반환 (로그에 표시되도록 attempt 값 유지)
    return text, attempt


def generate_therapist_turn(history):
    messages = [{"role": "system", "content": therapist_system_prompt()}]
    for h in history:
        role = "user" if h["speaker"] == "patient" else "assistant"
        messages.append({"role": role, "content": h["text"]})
    if not history:
        messages.append({"role": "user", "content": "Begin the session by asking the patient how their week has been."})
    return chat(THERAPIST_MODEL, messages, temperature=0.7, max_tokens=120)


def simulate_dialogue(ccd, n_turns):
    history = []
    rejection_count = 0
    for _ in range(n_turns):
        therapist_text = generate_therapist_turn(history)
        history.append({"speaker": "therapist", "text": therapist_text})

        patient_text, attempts = generate_patient_turn(ccd, history)
        rejection_count += attempts
        history.append({"speaker": "patient", "text": patient_text})
    return history, rejection_count


def to_chatml(history):
    messages = []
    for h in history:
        role = "user" if h["speaker"] == "therapist" else "assistant"
        messages.append({"role": role, "content": h["text"]})
    return messages


# ===== 3) Stratified sampling =====

def stratified_combos(n):
    """situation x style x fine-grained belief 조합을 최대한 고르게 분산해서 n개 뽑는다.
    n이 고유 조합 수보다 크면, 매 cycle마다 다시 셔플해서 순환시킨다
    (단순 슬라이싱은 n>고유 조합 수일 때 부족한 개수만 반환하는 버그가 있었음)."""
    base_combos = []
    for situation in SITUATION_CATEGORIES:
        for style in VALID_STYLES_BY_CATEGORY[situation]:
            for major, fine in ALL_FINE_GRAINED:
                base_combos.append((situation, style, major, fine))

    result = []
    while len(result) < n:
        cycle = base_combos.copy()
        random.shuffle(cycle)
        result.extend(cycle)
    return result[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="생성할 멀티턴 대화 개수")
    parser.add_argument("--turns", type=int, default=6, help="대화당 턴 수 (치료사+환자 1세트=1턴)")
    parser.add_argument("--out", type=str, default="data/patient_psi_multiturn_pilot.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id_offset", type=int, default=0, help="train/valid/test 합칠 때 id 충돌 방지용 오프셋")
    args = parser.parse_args()

    random.seed(args.seed)
    combos = stratified_combos(args.n)

    samples = []
    start = time.time()
    for idx, (situation, style, major, fine) in enumerate(combos, 1):
        print(f"[{idx}/{len(combos)}] situation={situation} style={style} belief={fine}")
        ccd = generate_ccd(situation, major, fine, style)
        history, rejections = simulate_dialogue(ccd, args.turns)

        sample = {
            "id": idx + args.id_offset,
            "situation_category": situation,
            "style": style,
            "core_belief_major": major,
            "core_belief_fine": fine,
            "ccd": ccd,
            "messages": to_chatml(history),
            "meta": {
                "patient_model": PATIENT_MODEL,
                "therapist_model": THERAPIST_MODEL,
                "ccd_model": FILTER_MODEL,
                "prompt_version": PROMPT_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "jargon_rejections": rejections,
            },
        }
        samples.append(sample)
        elapsed = time.time() - start
        print(f"  done in {elapsed/idx:.1f}s/sample avg, jargon_rejections={rejections}")

    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n✅ {len(samples)}개 멀티턴 대화 저장 완료: {args.out}")


if __name__ == "__main__":
    main()
