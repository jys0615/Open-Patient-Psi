import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed
)
from pathlib import Path
# QLoRA-specific setup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from itertools import product

set_seed(42)

# Load dataset from JSONL (full set, no split)
dataset = load_dataset("json", data_files="/data/yoonsuh0615/repos/patientv3/data/patient_psi_chatml.jsonl", split="train")

# Load tokenizer and base model
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

# Tokenize the dataset (apply chat template)
def tokenize(example):
    full_prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    split_idx = full_prompt.find("<|im_start|>assistant")
    if split_idx == -1:
        raise ValueError("assistant 응답이 포함되지 않은 메시지입니다.")

    user_prompt = full_prompt[:split_idx]
    assistant_response = full_prompt[split_idx:]

    # 통합 텍스트
    full_text = user_prompt + assistant_response

    # 토크나이즈 (fixed length)
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors=None
    )

    # user length
    user_ids = tokenizer(user_prompt, add_special_tokens=False)["input_ids"]
    user_len = len(user_ids)
    input_len = len(tokenized["input_ids"])

    # labels 설정
    labels = [-100] * user_len + tokenized["input_ids"][user_len:]
    labels = labels[:1024]  # 혹시 모를 초과 방지
    labels += [-100] * (1024 - len(labels))  # 부족하면 패딩

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

tokenized_dataset = dataset.map(tokenize, remove_columns=["messages"], batched=False)

# Load base model with 4-bit QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# LoRA config: Qwen2 구조는 q_proj/k_proj/v_proj/o_proj(attn) +
# gate_proj/up_proj/down_proj(MLP)이며 c_attn(GPT-2 계열)은 존재하지 않는다.
# 기존 코드의 "c_attn"은 매칭되는 레이어가 없어 무시되고, k_proj/o_proj/MLP에는
# LoRA가 전혀 적용되지 않은 채 학습되고 있었다.
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

for ep, lr in product([8, 10], [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]):
    set_seed(42)  # 조합마다 동일한 초기화 지점에서 시작하도록 고정
    run_id = f"3B_EP{ep}_LR{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    run_output_dir = Path(__file__).parent / "model" / run_id

    # 각 (epoch, lr) 조합마다 base model을 새로 로드하고 LoRA 어댑터를 새로
    # 초기화한다. 기존 코드는 model을 for문 밖에서 한 번만 만들어서, 두 번째
    # 조합부터는 이전 조합 학습 결과 위에 이어서 학습하는 버그가 있었다
    # (25개 하이퍼파라미터 조합이 서로 독립적이지 않았음).
    run_base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    run_base_model.gradient_checkpointing_enable()
    run_base_model = prepare_model_for_kbit_training(run_base_model)
    model = get_peft_model(run_base_model, lora_config)

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=ep,
        learning_rate=lr,
        logging_dir=f"{run_output_dir}/logs",
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    model.save_pretrained(f"{run_output_dir}/final")
    tokenizer.save_pretrained(f"{run_output_dir}/final")