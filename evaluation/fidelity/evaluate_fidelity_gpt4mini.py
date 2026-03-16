import json
from pathlib import Path
from prompter import generate_fidelity_prompt
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_PATH = Path("evaluation/fidelity/inputs/overall.jsonl")
OUTPUT_PATH = Path("evaluation/fidelity/outputs/gpt4mini_results.jsonl")

def load_inputs(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_outputs(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

def evaluate():
    inputs = load_inputs(INPUT_PATH)
    results = []
    for ex in inputs:
        prompt = generate_fidelity_prompt(
            conversation=ex["conversation"],
            response=ex["model_response"],
            dimension=ex["dimension"]
        )
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of simulated patient realism."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            gpt_response = completion.choices[0].message["content"].strip()
        except Exception as e:
            gpt_response = f"ERROR: {e}"

        results.append({
            "input_id": ex["input_id"],
            "prompt": prompt,
            "model": "gpt-4.1-mini",
            "response": gpt_response
        })
    save_outputs(results, OUTPUT_PATH)

if __name__ == "__main__":
    evaluate()