import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from pathlib import Path
# QLoRA-specific setup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from itertools import product

# Load dataset from JSONL (full set, no split)
dataset = load_dataset("json", data_files="/data/yoonsuh0615/repos/patientv3/data/patient_psi_chatml.jsonl", split="train")

# Load tokenizer and base model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
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

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for LoRA training
base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["c_attn", "q_proj", "v_proj"],  # Qwen 구조에 맞게 설정
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)

for ep, lr in product([10], [3e-4, 4e-4, 5e-4]):
    run_id = f"0.5B_EP{ep}_LR{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    run_output_dir = Path(__file__).parent / "model" / run_id

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