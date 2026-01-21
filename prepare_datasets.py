import pandas as pd
import xml.etree.ElementTree as ET
import re
import os

def prepare_datasets():
    # 1. Load original data from XML
    xml_path = "klxSAcorpus_20160224_1001/klxSAcorpus_20160224_1001.xml"
    if not os.path.exists(xml_path):
        print("XML not found.")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"ns": "http://klxsentiment.azurewebsites.net/annotation/annotateddata.xsd"}

    data = []
    for item in root.findall(".//ns:AnnotatedItem", ns):
        content = item.find("ns:Content", ns)
        score = item.find("ns:Score", ns)
        if content is None or score is None:
            continue
        
        text = content.text.strip()
        label = int(score.text.strip())
        data.append({"text": text, "original_label": label})

    df_v0 = pd.DataFrame(data)
    # Map numeric labels to sentiment strings
    mapping = {-1: "negative", 0: "neutral", 1: "positive"}
    df_v0['sentiment'] = df_v0['original_label'].map(mapping)
    
    # Save V0
    df_v0.to_csv("kks_v0_original.csv", index=False, encoding="utf-8")
    print(f"V0 dataset saved: {len(df_v0)} instances")

    # 2. Load corrections from ODS
    try:
        s_ods = pd.read_excel('sloberta_results_corrected_labels.ods', engine='odf')
        c_ods = pd.read_excel('crosloengual-bert_results_corrected_labels.ods', engine='odf')
        
        # Normalize labels in ODS
        label_map = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'poisitve': 'positive'
        }
        
        for df in [s_ods, c_ods]:
            df['human_corrected_label'] = df['human_corrected_label'].map(lambda x: label_map.get(str(x).strip().lower(), x) if pd.notna(x) else x)
            df['original_label_str'] = df['original_label'].map(lambda x: label_map.get(str(x).strip().lower(), x))

        # Merge corrections
        # We assume the order matches df_v0 based on my previous check
        df_v1 = df_v0.copy()
        
        # Add correction columns
        df_v1['correction_s'] = s_ods['human_corrected_label']
        df_v1['correction_c'] = c_ods['human_corrected_label']
        
        # Best label logic: prioritize human correction, else original
        def get_best_label(row):
            if pd.notna(row['correction_s']):
                return row['correction_s']
            if pd.notna(row['correction_c']):
                return row['correction_c']
            return row['sentiment']

        df_v1['sentiment'] = df_v1.apply(get_best_label, axis=1)
        
        # Save V1
        df_v1.to_csv("kks_v1_corrected.csv", index=False, encoding="utf-8")
        num_corrected = (df_v1['sentiment'] != df_v0['sentiment']).sum()
        print(f"\nV1 dataset saved: {len(df_v1)} instances (Corrected {num_corrected} labels)")

        # 3. Print Label Distribution Comparison
        print("\n" + "="*40)
        print("LABEL DISTRIBUTION: V0 vs V1")
        print("="*40)
        v0_dist = df_v0['sentiment'].value_counts().sort_index()
        v1_dist = df_v1['sentiment'].value_counts().sort_index()
        
        comparison = pd.DataFrame({
            'V0 (Original)': v0_dist,
            'V1 (Corrected)': v1_dist
        }).fillna(0).astype(int)
        
        comparison['Net Change'] = comparison['V1 (Corrected)'] - comparison['V0 (Original)']
        print(comparison)
        print("="*40)

    except Exception as e:
        print(f"Error preparing V1: {e}")

if __name__ == "__main__":
    prepare_datasets()
