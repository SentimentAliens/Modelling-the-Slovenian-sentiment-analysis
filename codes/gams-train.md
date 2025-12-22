```
# ------------------------------------------------------------
# 1️⃣  Imports & helper utilities
# ------------------------------------------------------------


import os
import torch
from unsloth import FastLanguageModel, LoRAConfig, train
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
import json
from accelerate import Accelerator

# ------------------------------------------------------------
# 2️⃣  Load the GaMS‑2B‑Instruct model & tokenizer
# ------------------------------------------------------------

model_name = "cjvt/GaMS-2B-Instruct"

# FastLanguageModel handles 8‑bit, bfloat16 & device_map for you
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_8bit=True,          # 8‑bit quantization
    device_map="auto",          # automatic placement on all GPUs
    dtype=torch.bfloat16,       # bfloat16 for better FP8 support
    max_seq_length=4096         # adjust if you want longer contexts
)

# ------------------------------------------------------------
# 3️⃣  Prepare LoRA adapters
# ------------------------------------------------------------

lora_config = LoRAConfig(
    r=16,                      # rank of the LoRA matrices
    lora_alpha=32,             # scaling hyper‑parameter
    lora_dropout=0.1,          # dropout on LoRA weights
    target_modules=["q_proj", "v_proj"],  # the most effective modules
    modules_to_save=["lm_head"]          # keep the output head
)

# Wrap the model with LoRA (this rewires the forward pass)
model = FastLanguageModel.get_peft_model(
    model, lora_config=lora_config
)

# ------------------------------------------------------------
# 4️⃣  Load & tokenise your training data
# ------------------------------------------------------------
# Example: a local JSONL file with {"prompt": "...", "completion": "..."}
dataset_path = "train.jsonl"

# If you don't have one, here's a quick dummy example you can write:
if not os.path.exists(dataset_path):
    dummy = [
        {"prompt": "Kako se zove ta film?", "completion": "To je \"The Matrix\"."},
        {"prompt": "Katera je najljubša knjiga?", "completion": "Moja najljubša knjiga je \"1984\"."}
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        for rec in dummy:
            f.write(json.dumps(rec) + "\n")

# Load using HuggingFace Datasets
raw_ds = load_dataset("json", data_files=dataset_path, split="train")

# Tokeniser expects a single string of "prompt + completion"
def make_prompt(example):
    return example["prompt"] + tokenizer.eos_token + example["completion"]

tokenized_ds = raw_ds.map(
    lambda x: tokenizer(
        make_prompt(x),
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding="max_length",
    ),
    batched=True,
    remove_columns=raw_ds.column_names,
)

# ------------------------------------------------------------
# 5️⃣  Define training arguments (unsloth style)
# ------------------------------------------------------------
training_args = {
    "output_dir": "./gams-2b-finetuned",
    "num_epochs": 3,
    "per_device_train_batch_size": 1,          # GPU‑friendly batch
    "gradient_accumulation_steps": 4,          # effective batch = 4
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "fp16": False,        # we are already using bfloat16
    "logging_steps": 10,
    "save_steps": 500,
    "evaluation_strategy": "no",
    "save_total_limit": 3,
}

# ------------------------------------------------------------
# 6️⃣  Train with UnsLoth
# ------------------------------------------------------------
# UnsLoth handles everything: DataLoader, optimizer, gradient accumulation, etc.
train(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    **training_args
)

# ------------------------------------------------------------
# 7️⃣  Save the fine‑tuned LoRA adapters
# ------------------------------------------------------------
# The base model stays the same; only the LoRA weights are new.
os.makedirs(training_args["output_dir"], exist_ok=True)
model.save_pretrained(training_args["output_dir"])
tokenizer.save_pretrained(training_args["output_dir"])

print("\n✅  Fine‑tuning finished!")
print(f"Model & tokenizer saved to {training_args['output_dir']}")

```
