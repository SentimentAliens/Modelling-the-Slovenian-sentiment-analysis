import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

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
    labels.append(lab)

# --------------------------------------------------------------------
# Fixed train–test split
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
# Zero-shot model 1: classla/xlm-r-parlasent
# --------------------------------------------------------------------
clf1 = pipeline(
    "text-classification",
    model="classla/xlm-r-parlasent",
    tokenizer="classla/xlm-r-parlasent",
    device=0 if torch.cuda.is_available() else -1
)

# parlasent model labels vary by task; ensure mapping to integer sentiment
# Assume labels: "negative", "neutral", "positive"
label_map = {
    "negative": -1,
    "neutral": 0,
    "positive": 1
}

preds_1 = []
for t in X_test:
    out = clf1(t, truncation=True)[0]["label"].lower()
    preds_1.append(label_map.get(out, 0))

acc_1 = accuracy_score(y_test, preds_1)
f1_1 = f1_score(y_test, preds_1, average="macro")

print("Parlasent: accuracy:", acc_1)
print("Parlasent: macro-F1:", f1_1)
