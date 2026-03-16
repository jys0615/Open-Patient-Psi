import openai
import random
import os
import re
from dotenv import load_dotenv
import time
from datetime import timedelta

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 논문 기반 고정 목록
situation_categories = [
    "family dynamics", "workplace pressure", "relationship dynamics",
    "social interactions", "personal growth issues", "financial concerns", "daily life stressors"
]
core_beliefs_all = ["helpless", "unlovable", "worthless"]
emotions = ["anxious", "sad", "angry", "hurt", "disappointed", "ashamed", "guilty", "suspicious", "jealous"]

# fine-grained core beliefs from Table 7 (with frequencies)
fine_grained_beliefs = {
    "helpless": [
        ("I am incompetent.", 40),
        ("I am helpless.", 47),
        ("I am powerless, weak, vulnerable.", 48),
        ("I am a victim.", 9),
        ("I am needy.", 10),
        ("I am trapped.", 39),
        ("I am out of control.", 34),
        ("I am a failure, loser.", 26),
        ("I am defective.", 8)
    ],
    "unlovable": [
        ("I am unlovable.", 59),
        ("I am unattractive.", 0),
        ("I am undesirable, unwanted.", 31),
        ("I am bound to be rejected.", 21),
        ("I am bound to be abandoned.", 32),
        ("I am bound to be alone.", 30)
    ],
    "worthless": [
        ("I am worthless, waste.", 13),
        ("I am immoral.", 4),
        ("I am bad - dangerous, toxic, evil.", 2),
        ("I don’t deserve to live.", 0)
    ]
}

def sample_core_beliefs():
    selected_major = random.choices(list(fine_grained_beliefs.keys()), weights=[94, 71, 15], k=random.randint(1, 2))
    result = []
    for major in selected_major:
        choices, weights = zip(*fine_grained_beliefs[major])
        result.append(random.choices(choices, weights=weights, k=1)[0])
    return result

# GPT 프롬프트 빌더
def build_prompt(situation_category, core_beliefs, style):
    return f"""
You are a clinical simulation expert tasked with creating a CBT-based cognitive model of a patient.

## Task
Simulate a single patient’s CCD (Cognitive Conceptualization Diagram) following Beck (2020)'s structure.
You must generate all components below in natural English, reflecting realistic psychological conditions.

## Instructions:
- Situation category: {situation_category}
- Core beliefs: {', '.join(core_beliefs)}
- Conversational style: {style}
- The cognitive model must follow a logical and causal psychological flow:
  Core beliefs → Intermediate beliefs → Automatic thoughts → Emotions → Behaviors.
- Each component should naturally and logically stem from the one preceding it.
- Each field should use realistic patient language, emotionally rich and grounded in real-life cognitive patterns.
- Ensure all components reflect the patient's subjective experience in line with CBT principles.
- The model must simulate naturalistic thought processes and emotional responses.
- All fields must follow the exact CCD structure.

## Output format (fill in each field):
Relevant history:
Core beliefs:
Intermediate beliefs:
Coping strategies:
Situation:
Automatic thoughts:  # Write 1–2 full sentences. Avoid using quotes or lists.
Emotions:
Behaviors:
"""

# 정규표현식 기반 파싱 함수
def parse_cognitive_model(text):
    fields = [
        "Relevant history", "Core beliefs", "Intermediate beliefs",
        "Coping strategies", "Situation", "Automatic Thoughts",
        "Emotions", "Behaviors"
    ]
    result = {}
    for field in fields:
        pattern = rf"{field}:\s*(.*?)(?=\n[A-Z][a-z ]+?:|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        key = field.lower().replace(" ", "_")
        value = match.group(1).strip() if match else ""
        # Clean up wrongly escaped characters including smart quotes
        value = value.encode("utf-8", "ignore").decode("unicode_escape", "ignore")
        value = value.replace('â', "'").replace('â', "'").replace('â', '-').replace('â', '-')
        value = value.replace('\\"', '"').replace("\\'", "'")
        # Additionally strip trailing or leading double quotes
        value = value.strip().strip('"').strip("'")
        result[key] = value
    return result

# GPT 호출 함수
def generate_patient_psi_sample(id_num):
    situation_category = random.choices(
        situation_categories, weights=[25, 20, 19, 18, 8, 8, 8], k=1
    )[0]

    valid_styles_by_category = {
        "family dynamics": ["upset", "reserved", "plain"],
        "workplace pressure": ["reserved", "plain", "pleasing"],
        "relationship dynamics": ["verbose", "tangent", "pleasing"],
        "social interactions": ["tangent", "pleasing", "plain"],
        "personal growth issues": ["verbose", "reserved", "plain"],
        "financial concerns": ["upset", "reserved", "plain"],
        "daily life stressors": ["plain", "tangent", "pleasing"]
    }
    valid_styles = valid_styles_by_category[situation_category]

    core_beliefs = sample_core_beliefs()
    style = random.choice(valid_styles)

    prompt = build_prompt(situation_category, core_beliefs, style)

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You must follow the output format precisely with field names exactly as specified."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    text = response['choices'][0]['message']['content']

    parsed = parse_cognitive_model(text)

    result = {
        "id": id_num,
        "situation_category": situation_category,
        "core_beliefs": core_beliefs,
        "style": style,
        **parsed,
        "response": ""
    }
    return result

import json


all_samples = []
start_time = time.time()
for i in range(1, 201):
    sample = generate_patient_psi_sample(i)
    elapsed = time.time() - start_time
    avg_time_per_sample = elapsed / i
    remaining = avg_time_per_sample * (200 - i)
    print(f"⏳ Estimated time remaining: {remaining / 60:.2f} minutes")
    all_samples.append(sample)

with open("data/patient_psi_testset.json", "w") as f:
    json.dump(all_samples, f, indent=2, ensure_ascii=False)