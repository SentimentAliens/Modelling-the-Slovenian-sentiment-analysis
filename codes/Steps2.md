Below is a **complete, end-to-end working solution** containing:

1. **A GPU-accelerated PyTorch + HuggingFace classifier wrapped with cleanlab**
2. **A “full classifier” version (no TF-IDF; real Transformer model)**
3. **Automatic label fixing using cleanlab quality scores**
4. **A single unified script you can run on CPU or GPU**

This is the template you want when running cleanlab with modern NLP models.

---

# ✅ **GPU-accelerated PyTorch + cleanlab + HuggingFace classifier**

This uses:

* **HuggingFace Transformers** (e.g. `"EMBEDDIA/sloberta-base"` — good for Slovene)
* **PyTorch**
* **cleanlab** for label-error detection
* **cross-validated out-of-sample predictions**
* **automatic label correction**

---

# 🚀 **Full Working Code**

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.model_selection import StratifiedKFold
import numpy as np

#############################################
# 1. Load Your Dataset
#############################################

df = pd.read_csv("kks_opinion_corpus.csv")  # You must convert KKS XML → CSV first
texts = df["text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()

# Encode labels → integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(labels)
num_labels = len(le.classes_)

#############################################
# 2. HuggingFace Model + Tokenizer
#############################################

MODEL_NAME = "EMBEDDIA/sloberta-base"   # Strong Slovene model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#############################################
# 3. PyTorch Dataset + DataLoader
#############################################

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(self.labels[idx])
        return enc


#############################################
# 4. Function to train a model and return
#    out-of-sample predicted probabilities
#############################################

def train_fold(train_idx, valid_idx):

    train_ds = TextDataset([texts[i] for i in train_idx], y[train_idx])
    valid_ds = TextDataset([texts[i] for i in valid_idx], y[valid_idx])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(device)

    optim = AdamW(model.parameters(), lr=2e-5)

    # Train a few epochs (you can increase)
    model.train()
    for epoch in range(1):
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optim.step()

    # Predict on validation fold
    model.eval()
    probs = []

    with torch.no_grad():
        for batch in valid_loader:
            labels_ = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)

    probs = np.concatenate(probs, axis=0)
    return probs


#############################################
# 5. Cross-validated out-of-sample probabilities
#############################################

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_probs = np.zeros((len(texts), num_labels))

for train_idx, valid_idx in skf.split(texts, y):
    fold_probs = train_fold(train_idx, valid_idx)
    all_probs[valid_idx] = fold_probs


#############################################
# 6. cleanlab: Detect Label Issues
#############################################

label_quality = get_label_quality_scores(labels=y, pred_probs=all_probs)

issues = find_label_issues(labels=y, pred_probs=all_probs, return_indices_ranked_by="self_confidence")

df_results = pd.DataFrame({
    "text": texts,
    "original_label": le.inverse_transform(y),
    "predicted_label": le.inverse_transform(all_probs.argmax(axis=1)),
    "label_quality": label_quality,
    "is_issue": np.isin(np.arange(len(texts)), issues)
})

print(df_results[df_results.is_issue].head(20))

#############################################
# 7. AUTO-FIX LABELS (optional)
#############################################

# Rule: if predicted label probability > threshold and quality score low → auto-fix
THRESHOLD = 0.8  # adjust

auto_fixed_labels = y.copy()

for i in range(len(texts)):
    pred = all_probs[i].argmax()
    pred_prob = all_probs[i][pred]
    if pred_prob > THRESHOLD and label_quality[i] < 0.4:  # low quality
        auto_fixed_labels[i] = pred

df_results["auto_fixed_label"] = le.inverse_transform(auto_fixed_labels)

print("\nExamples where label was automatically changed:\n")
print(df_results[df_results.original_label != df_results.auto_fixed_label].head(20))

#############################################
# 8. Save cleaned dataset
#############################################

df_clean = pd.DataFrame({
    "text": texts,
    "label": le.inverse_transform(auto_fixed_labels)
})
df_clean.to_csv("kks_opinion_corpus_cleaned.csv", index=False)

print("\nSaved cleaned dataset → kks_opinion_corpus_cleaned.csv")
```

---

# ✔ What this script does

### **1. Loads Slovene text + labels**

You just convert the KKS XML to a simple CSV (`text`, `label`).

### **2. Builds a GPU-accelerated Slovene classifier**

Using **Slovene RoBERTa (SloBERTa)**.

### **3. Runs cross-validated out-of-sample predictions**

This is REQUIRED by cleanlab.

### **4. Computes cleanlab label-quality scores**

Low scores = likely labeling errors.

### **5. Detects label issues**

`find_label_issues(...)` gives ranked suspicious samples.

### **6. Auto-fixes labels**

Based on:

* high predicted confidence
* low cleanlab label quality

### **7. Saves the cleaned dataset**

(You can retrain a model on this later.)

---

