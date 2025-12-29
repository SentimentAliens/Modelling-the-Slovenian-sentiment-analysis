import xml.etree.ElementTree as ET
import re
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------
# Load XML dataset
# --------------------------------------------------------------------
path = "klxSAcorpus_20160224_1001/klxSAcorpus_20160224_1001.xml"

tree = ET.parse(path)
root = tree.getroot()

ns = {"ns": "http://klxsentiment.azurewebsites.net/annotation/annotateddata.xsd"}

texts = []
labels = []

for item in root.findall(".//ns:AnnotatedItem", ns):
    content = item.find("ns:Content", ns)
    score = item.find("ns:Score", ns)

    if content is None or score is None:
        continue

    txt = content.text.strip()
    lab = int(score.text.strip())

    texts.append(txt)
    labels.append(lab + 1)  # map {-1,0,1} → {0,1,2}

# --------------------------------------------------------------------
# Train–test split (fixed, stratified)
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=labels
)

# --------------------------------------------------------------------
# Load GaMS model
# --------------------------------------------------------------------
model_name = "cjvt/GaMS-2B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --------------------------------------------------------------------
# Label normalization with comprehensive regex patterns
# --------------------------------------------------------------------
LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

# Output file for logging all raw predictions
RAW_OUTPUT_FILE = "gams_raw_outputs.txt"

def normalize_gams_output(text, log_file=None):
    """
    Extract the first valid sentiment label from model output.
    Handles various formats:
    - Case variations: POSITIVE, Positive, positive
    - With punctuation: positive., positive,
    - With quotes: "positive", 'positive'
    - Common typos and variations
    - Slovenian equivalents: pozitiven, negativen, nevtralen
    """
    original_text = text
    text_lower = text.lower().strip()
    
    # Log raw output if file handle provided
    if log_file:
        log_file.write(f"RAW: {repr(original_text)}\n")
    
    # Pattern 1: English labels (case-insensitive, with optional punctuation/quotes)
    english_pattern = r"""
        [\"\']?                           # Optional leading quote
        \b(positive|negative|neutral)\b   # Core sentiment word
        [\"\']?                           # Optional trailing quote
        [.,!?]?                           # Optional punctuation
    """
    match = re.search(english_pattern, text_lower, re.VERBOSE)
    if match:
        return LABEL_MAP[match.group(1)]
    
    # Pattern 2: Slovenian labels (pozitiven/pozitivno, negativen/negativno, nevtralen/nevtralno)
    slovenian_map = {
        "pozitiv": "positive",    # matches pozitiven, pozitivno, pozitivna
        "negativ": "negative",    # matches negativen, negativno, negativna
        "nevtral": "neutral",     # matches nevtralen, nevtralno, nevtralna
    }
    for slo_prefix, eng_label in slovenian_map.items():
        if re.search(rf"\b{slo_prefix}\w*\b", text_lower):
            return LABEL_MAP[eng_label]
    
    # Pattern 3: Common variations and abbreviations
    variations = {
        r"\bpos\b": "positive",
        r"\bneg\b": "negative",
        r"\bneut\b": "neutral",
        r"\b\+\b": "positive",
        r"\b-\b": "negative",
        r"\b0\b": "neutral",
        r"\b1\b": "positive",      # Sometimes 1 = positive
        r"\b2\b": "negative",      # Sometimes 2 = negative (varies by dataset)
    }
    for pattern, label in variations.items():
        if re.search(pattern, text_lower):
            return LABEL_MAP[label]
    
    # Pattern 4: Sentiment as number with context
    number_patterns = [
        (r"sentiment[:\s]+1", "positive"),
        (r"sentiment[:\s]+0", "neutral"),
        (r"sentiment[:\s]+-1", "negative"),
        (r"sentiment[:\s]+2", "positive"),
    ]
    for pattern, label in number_patterns:
        if re.search(pattern, text_lower):
            return LABEL_MAP[label]
    
    # Fallback: neutral (log as unrecognized)
    if log_file:
        log_file.write(f"UNRECOGNIZED: {repr(original_text)}\n")
    
    return LABEL_MAP["neutral"]

# --------------------------------------------------------------------
# Classification function
# --------------------------------------------------------------------
def classify_gams(text, log_file=None):
    prompt = (
        "You are a sentiment analysis assistant.\n"
        "Classify the sentiment of the following text as positive, negative, or neutral.\n"
        "Respond with ONLY one word: positive, negative, or neutral.\n\n"
        f"Text: {text}\n"
        "Sentiment:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    if "Sentiment:" in decoded:
        generated_part = decoded.split("Sentiment:")[-1].strip()
    else:
        generated_part = decoded
    
    return normalize_gams_output(generated_part, log_file)

# --------------------------------------------------------------------
# Run zero-shot inference with logging
# --------------------------------------------------------------------
print(f"Logging raw outputs to: {RAW_OUTPUT_FILE}")

with open(RAW_OUTPUT_FILE, "w", encoding="utf-8") as log_file:
    log_file.write("=" * 60 + "\n")
    log_file.write("GaMS Raw Output Log\n")
    log_file.write("=" * 60 + "\n\n")
    
    preds = []
    for i, t in enumerate(X_test):
        log_file.write(f"\n--- Sample {i+1}/{len(X_test)} ---\n")
        log_file.write(f"TEXT: {t[:100]}...\n" if len(t) > 100 else f"TEXT: {t}\n")
        pred = classify_gams(t, log_file)
        preds.append(pred)
        log_file.write(f"NORMALIZED: {list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(pred)]}\n")

print(f"Logged {len(preds)} predictions to {RAW_OUTPUT_FILE}")

# --------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------
accuracy = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average="macro")

print("GaMS accuracy:", accuracy)
print("GaMS macro-F1:", macro_f1)
