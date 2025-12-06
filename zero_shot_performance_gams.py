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
# Zero-shot model 2: cjvt/GaMS-2B-Instruct
# --------------------------------------------------------------------
# GaMS is a causal LM; prompt it explicitly
tok2 = AutoTokenizer.from_pretrained("cjvt/GaMS-2B-Instruct")
mod2 = AutoModelForCausalLM.from_pretrained(
    "cjvt/GaMS-2B-Instruct",
    dtype=torch.float32,
    low_cpu_mem_usage=False
)

mod2.to("cpu")

def classify_gams(text):
    prompt = (
        "You are a sentiment classifier. "
        "Return only one of: -1, 0, 1.\n"
        f"Text: {text}\n"
        "Sentiment:"
    )
    inputs = tok2(prompt, return_tensors="pt").to(mod2.device)
    with torch.no_grad():
        out = mod2.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False
        )
    decoded = tok2.decode(out[0], skip_special_tokens=True)
    # extract the last token(s)
    ans = decoded.split("Sentiment:")[-1].strip()
    # clean
    ans = ans.split()[0].strip()
    try:
        return int(ans)
    except:
        return 0

preds_2 = [classify_gams(t) for t in X_test]

acc_2 = accuracy_score(y_test, preds_2)
f1_2 = f1_score(y_test, preds_2, average="macro")

print("GaMS: accuracy:", acc_2)
print("GaMS: macro-F1:", f1_2)
