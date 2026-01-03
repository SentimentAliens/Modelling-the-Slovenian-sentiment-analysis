import os
import torch
import psutil
import builtins
builtins.psutil = psutil
from unsloth import FastLanguageModel
from datasets import Dataset
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from trl import SFTTrainer
from transformers import TrainingArguments

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
VERSION = "V0" # Or V1
DATASET_PATH = "kks_v0_original.csv"
OUTPUT_DIR = f"./gams-2b-finetuned-kks-{VERSION}"
MISCLASSIFIED_PATH = f"misclassified_instances_{VERSION}.csv"
RESULTS_JSON = f"evaluation_results_{VERSION}.json"

# --------------------------------------------------------------------
# Load Dataset
# --------------------------------------------------------------------
print(f"Loading {VERSION} dataset from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

# Use 'sentiment' column for labels
texts = df['text'].tolist()
labels = df['sentiment'].tolist()

# --------------------------------------------------------------------
# Train–test split (fixed, stratified, random_state=42)
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=labels
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ------------------------------------------------------------
# Load the GaMS‑2B‑Instruct model & tokenizer
# ------------------------------------------------------------
model_name = "cjvt/GaMS-2B-Instruct"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None, 
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def create_sentiment_prompt(text):
    return f"""Analiziraj sentiment naslednjega besedila in odgovori z eno besedo: 'positive', 'negative' ali 'neutral'.

Besedilo: {text}

Sentiment:"""

def text_to_label(text):
    text = text.lower().strip()
    if "negative" in text or "negativen" in text or "negativno" in text:
        return "negative"
    elif "positive" in text or "pozitiven" in text or "pozitivno" in text:
        return "positive"
    elif "neutral" in text or "nevtralen" in text or "nevtralno" in text:
        return "neutral"
    else:
        return None

# ------------------------------------------------------------
# Prepare training dataset
# ------------------------------------------------------------
train_data = []
for text, label in zip(X_train, y_train):
    prompt = create_sentiment_prompt(text)
    train_data.append({"prompt": prompt, "completion": label})

train_dataset = Dataset.from_list(train_data)

def formatting_prompts_func(examples):
    texts = []
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        text = f"{prompt} {completion}{tokenizer.eos_token}"
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

# ------------------------------------------------------------
# Training arguments
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    optim="adamw_8bit",
    seed=42,
)

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
print(f"\nStarting training for {VERSION}...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)
trainer.train()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------
print(f"\nEvaluating {VERSION} on test set...")
FastLanguageModel.for_inference(model)

predictions = []
misclassified = []

for i, (text, true_label) in enumerate(zip(X_test, y_test)):
    prompt = create_sentiment_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    pred_label = text_to_label(response)
    
    if pred_label is None:
        pred_label = "neutral" # Default
    
    predictions.append(pred_label)
    
    if pred_label != true_label:
        misclassified.append({
            "text": text,
            "true_label": true_label,
            "predicted_label": pred_label,
            "model_response": response
        })

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
f1_macro = f1_score(y_test, predictions, average='macro')

print(f"\n{VERSION} RESULTS:")
print(f"GaMS Accuracy (IFT original KKS dataset): {accuracy:.4f}")
print(f"GaMS Macro-F1 (IFT original KKS dataset): {f1_macro:.4f}")

# Save misclassified
if misclassified:
    pd.DataFrame(misclassified).to_csv(MISCLASSIFIED_PATH, index=False, encoding="utf-8")
    print(f"Misclassified instances saved to {MISCLASSIFIED_PATH}")
    