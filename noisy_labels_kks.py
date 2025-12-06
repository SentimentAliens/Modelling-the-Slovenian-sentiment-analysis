import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from collections import Counter
from scipy.stats import entropy
import re
import os
import warnings
warnings.filterwarnings('ignore')

print("=== KKS 1.001 Noisy Label Detector ===")

def extract_and_parse(zip_path='klxSAcorpus_20160224_1001.zip', xml_path='klxSAcorpus_20160224_1001.xml'):
    print(f"Checking {zip_path}...")

    if os.path.exists(zip_path):
        print("Extracting ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Extracted!")

    possible_xmls = ['klxSAcorpus_20160224_1001.xml', 'klxSAcorpus_20160224_1001bal.xml', xml_path]
    tree = None
    for xml_file in possible_xmls:
        if os.path.exists(xml_file):
            print(f"Found XML: {xml_file}")
            tree = ET.parse(xml_file)
            break

    if tree is None:
        print("No XML found!")
        return pd.DataFrame()

    root = tree.getroot()

    # Namespace FIX
    ns = {'ns': 'http://klxsentiment.azurewebsites.net/annotation/annotateddata.xsd'}

    data = []
    print("Parsing AnnotatedItems...")

    for item in root.findall('.//ns:AnnotatedItem', ns):

        score_elem = item.find('ns:Score', ns)
        content_elem = item.find('ns:Content', ns)

        if score_elem is None or content_elem is None:
            continue

        try:
            score = int(score_elem.text.strip())
        except:
            continue

        label_map = {1: 'pos', -1: 'neg', 0: 'neu'}
        if score not in label_map:
            continue
        label = label_map[score]

        text = re.sub(r'\s+', ' ', content_elem.text.strip())
        if len(text) > 5:
            data.append({'text': text, 'label': label, 'score': score})

    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} instances")
    return df

# RUN ANALYSIS
df = extract_and_parse()

if len(df) == 0:
    print("NO DATA. Verify ZIP contains klxSAcorpus_20160224_1001.xml")
    print("Files in folder:", os.listdir('.'))
    exit()

# ML Setup
label_id_map = {'pos': 0, 'neg': 1, 'neu': 2}
df['label_id'] = df['label'].map(label_id_map)
df = df.dropna(subset=['label_id'])
print(f"Ready: {len(df)} instances")

# Train model
print("\n Training TF-IDF...")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), lowercase=True)),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])
pipeline.fit(train_df['text'], train_df['label_id'])

# Predictions + confidence
probas = pipeline.predict_proba(df['text'])
df['pred_label'] = np.argmax(probas, axis=1)
df['max_prob'] = np.max(probas, axis=1)
df['entropy'] = np.array([entropy(p) for p in probas])

print("Test performance:")
print(classification_report(test_df['label_id'], pipeline.predict(test_df['text']), 
                           target_names=['POS','NEG','NEU']))

# Noisy label detection
low_conf = df['max_prob'] < 0.75
high_unc = df['entropy'] > 0.65
mismatch = df['label_id'] != df['pred_label']
df['noisy_score'] = np.mean([low_conf.astype(int), high_unc.astype(int), mismatch.astype(int)], axis=0)
df['is_noisy'] = df['noisy_score'] >= 2

noisy_df = df[df['is_noisy']]
print(f"\n {len(noisy_df)} NOISY LABELS ({100*len(noisy_df)/len(df):.1f}%)")

# Show results
print("\n TOP NOISY:")
pred_names = {0:'POS',1:'NEG',2:'NEU'}
for _, row in noisy_df.nlargest(5, 'noisy_score')[['label','pred_label','max_prob','noisy_score','text']].iterrows():
    print(f"{row['label']}→{pred_names[row['pred_label']]} | P:{row['max_prob']:.2f} | Score:{row['noisy_score']:.1f}")
    print(f"   [translate:{row['text'][:120]}...]\n")

# Save
df.to_csv('kks_full_analysis.csv', index=False)
noisy_df.to_csv('kks_noisy_only.csv', index=False)
print("SAVED: kks_full_analysis.csv + kks_noisy_only.csv")
print("COMPLETE!")
