```
import os
import torch
import pandas as pd
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 1. Configuration
model_name = "cjvt/GaMS-2B-Instruct"
max_seq_length = 1024
dtype = None # Auto detection (Float16/Bfloat16)
load_in_4bit = True # Use 4bit to save VRAM for faster iteration

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Data Preparation
# Note: SentiNews 1.0 (11356/1115) usually contains 'SentiNews-sentence.csv'
# Labels: 1(Very Neg), 2(Neg), 3(Neu), 4(Pos), 5(Very Pos) OR simplified 3-class.
# We will map them to text labels for the instruction model.

def load_sentinews():
    # Replace with the path to your downloaded SentiNews-sentence.csv
    # You can download it from: https://www.clarin.si/repository/xmlui/handle/11356/1115
    df = pd.read_csv("SentiNews-sentence.csv") 
    
    # Map numerical labels to Slovenian text labels
    label_map = {1: "zelo negativno", 2: "negativno", 3: "nevtralno", 4: "pozitivno", 5: "zelo pozitivno"}
    df['sentiment_text'] = df['sentiment'].map(label_map)
    
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    return train_df, test_df

train_df, test_df = load_sentinews()

# Formatting for GaMS-2B-Instruct (Gemma 2 Prompt Template)
prompt_style = """<start_of_turn>user
Določi sentiment naslednjega slovenskega besedila:
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""

def formatting_prompts_func(examples):
    texts = []
    for content, label in zip(examples["content"], examples["sentiment_text"]):
        text = prompt_style.format(content, label)
        texts.append(text)
    return { "text" : texts }

train_dataset = Dataset.from_pandas(train_df).map(formatting_prompts_func, batched = True)

# 4. Training
trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        max_steps = 500, # Adjust based on dataset size
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 250,
    ),
)

trainer.train()

# 5. Save the Best Model
model.save_pretrained("gams_sentinews_model")
tokenizer.save_pretrained("gams_sentinews_model")

# 6. Evaluation on Test Split
print("\nStarting Evaluation...")
FastLanguageModel.for_inference(model)

predictions = []
true_labels = test_df['sentiment_text'].tolist()
mismatches = []

for index, row in test_df.iterrows():
    inputs = tokenizer(
        [prompt_style.format(row['content'], "")], 
        return_tensors = "pt"
    ).to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens = 10)
    # Extract only the generated part
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1].strip()
    
    predictions.append(pred_text)
    
    if pred_text != row['sentiment_text']:
        mismatches.append({
            "text": row['content'],
            "original_label": row['sentiment_text'],
            "predicted_label": pred_text
        })

# 7. Compute Metrics
report = classification_report(true_labels, predictions)
print(report)

# 8. Save Mismatches
mismatches_df = pd.DataFrame(mismatches)
mismatches_df.to_csv("test_mismatches.csv", index=False, encoding='utf-8')
print("Mismatched instances saved to 'test_mismatches.csv'.")

```
