import openai
import json
import sys
import time
from typing import Optional

# 입력 및 출력 경로
# 사용법: python data/testml_generation.py        -> testset 변환 (기본값)
#         python data/testml_generation.py valid  -> validset 변환
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
if SPLIT == "valid":
    INPUT_PATH = "data/patient_psi_validset.json"
    OUTPUT_PATH = "data/patient_psi_validml.jsonl"
else:
    INPUT_PATH = "data/patient_psi_testset.json"
    OUTPUT_PATH = "data/patient_psi_testml.jsonl"
MAX_COUNT = 1000  # e.g., set to 100 to only process first 100 samples

# 리스트 또는 문자열을 안전하게 문자열로 변환
def stringify(value):
    if isinstance(value, list):
        return ", ".join(value)
    return value.strip() if isinstance(value, str) else str(value)

def generate_assistant_response(prompt, model="gpt-4.1-mini", temperature=0.7):
    messages = [
        {"role": "system", "content": "You are a simulated patient in a CBT session."},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response["choices"][0]["message"]["content"].strip()

def evaluate_response(prompt: str, model_response: str, reference: Optional[str] = None):
    # TODO: Replace with GPT-4.1-mini or other evaluator API
    # For now, just print for debugging
    print(f"[EVAL] Prompt preview: {prompt[:80]}...")
    print(f"[EVAL] Response preview: {model_response[:80]}...")
    if reference:
        print(f"[EVAL] Reference preview: {reference[:80]}...")
    # Return dummy score for now
    return {"score": 0.0, "reason": "Evaluation not yet implemented"}

# ChatML 형식으로 변환
def convert_to_chatml(sample):
    history = stringify(sample.get("relevant_history", ""))
    core_beliefs = stringify(sample.get("core_beliefs", []))
    intermediate_beliefs = stringify(sample.get("intermediate_beliefs", ""))
    intermediate_depression = stringify(sample.get("intermediate_beliefs_depressed", ""))
    coping_strategies = stringify(sample.get("coping_strategies", ""))
    situation = stringify(sample.get("situation", ""))
    automatic_thoughts = stringify(sample.get("automatic_thoughts", []))
    emotions = stringify(sample.get("emotions", []))
    behaviors = stringify(sample.get("behaviors", []))
    style_list = sample.get("conversational_styles", [])
    STYLE_DESCRIPTION_MAP = {
        "plain": "The patient communicates in a direct, straightforward manner.",
        "upset": "An upset patient may 1) exhibit anger or resistance towards the therapist or the therapeutic process, 2) may be challenging or dismissive of the therapist’s suggestions and interventions, 3) have difficulty trusting the therapist and forming a therapeutic alliance, and 4) be prone to arguing, criticizing, or expressing frustration during therapy sessions.",
        "verbose": "A verbose patient may 1) provide detailed responses to questions, even if directly relevant, 2) elaborate on personal experiences, thoughts, and feelings extensively, and 3) demonstrate difficulty in allowing the therapist to guide the conversation.",
        "reserved": "A reserved patient may 1) provide brief, vague, or evasive answers to questions, 2) demonstrate reluctance to share personal information or feelings, 3) require more prompting and encouragement to open up, and 4) express distrust or skepticism towards the therapist.",
        "tangent": "A patient who goes off on tangent may 1) start answering a question but quickly veer off into unrelated topics, 2) share personal anecdotes or experiences that are not relevant to the question asked, 3) demonstrate difficulty staying focused on the topic at hand, and 4) require redirection to bring the conversation back to the relevant points.",
        "pleasing": "A pleasing patient may 1) minimize or downplay your own concerns or symptoms to maintain a positive image, 2) demonstrate eager-to-please behavior and avoid expressing disagreement or dissatisfaction, 3) seek approval or validation from the therapist frequently, and 4) agree with the therapist’s statements or suggestions readily, even if they may not fully understand or agree."
    }
    style_descriptions = [STYLE_DESCRIPTION_MAP.get(s, "") for s in style_list if s in STYLE_DESCRIPTION_MAP]
    style_description = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(style_descriptions))
    response = stringify(sample.get("response", ""))

    prompt = fprompt = f"""Imagine you are XXX, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as XXX would during a cognitive behavioral therapy (CBT) session. Align your responses with XXX’s background information provided in the ‘Relevant history’ section. Your thought process should be guided by the cognitive conceptualization diagram in the ‘Cognitive Conceptualization Diagram’ section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms.

Patient History: {sample['relevant_history']}

Cognitive Conceptualization Diagram:
Core Beliefs: {sample['core_beliefs']}
Intermediate Beliefs: {sample['intermediate_beliefs']}
Intermediate Beliefs during Depression: {sample['intermediate_depression']}
Coping Strategies: {sample['coping_strategies']}

You will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.

Situation: {sample['situation']}
Automatic thoughts: {sample['automatic_thoughts']}
Emotions: {sample['emotions']}
Behaviors: {sample['behaviors']}

In the upcoming conversation, you will simulate XXX during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:
1. {sample['style']} style
2. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.
3. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient’s true thoughts and emotions.
4. Maintain consistency with XXX’s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.
5. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to XXX’s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.

You are now XXX. Only generate XXX’s responses, without including the therapist’s words or questions. Limit your responses to a maximum of 5 sentences."""


    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }

# 메인 변환 로직
def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    start_time = time.time()
    total = len(data) if MAX_COUNT is None else min(len(data), MAX_COUNT)
    with open(OUTPUT_PATH, "w") as out_f:
        for idx, sample in enumerate(data):
            if MAX_COUNT is not None and idx >= MAX_COUNT:
                break
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (total - idx - 1)
            print(f"🧠 {idx + 1}/{total} 테스트셋 변환 중... ⏱️ 약 {remaining:.1f}초 남음")
            prompt_sample = convert_to_chatml(sample)
            # Auto-generate assistant reply if missing or empty (train과 동일하게)
            if not sample.get("response", "").strip():
                prompt = prompt_sample["messages"][0]["content"]
                sample["response"] = generate_assistant_response(prompt)
                prompt_sample = convert_to_chatml(sample)
            out_f.write(json.dumps(prompt_sample, ensure_ascii=False) + "\n")

    print(f"✅ 테스트셋 ChatML 변환 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

# TODO: Implement evaluation loop using GPT-4.1-mini or other evaluator
# def evaluate_response(prompt, model_response):
#     ...