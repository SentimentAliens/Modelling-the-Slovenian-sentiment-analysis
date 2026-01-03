import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import os

def analyze_results(file_path, model_name):
    print(f"--- Analysis for {model_name} ---")
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.\n")
        return

    # Load data
    df = pd.read_excel(file_path, engine='odf')
    
    # Define label normalization to fix typos found in data
    label_map = {
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        'poisitve': 'positive' # fix typo
    }
    
    df['original_label'] = df['original_label'].map(lambda x: label_map.get(str(x).strip().lower(), x))
    df['predicted_label'] = df['predicted_label'].map(lambda x: label_map.get(str(x).strip().lower(), x))
    df['human_corrected_label'] = df['human_corrected_label'].map(lambda x: label_map.get(str(x).strip().lower(), x) if pd.notna(x) else x)

    # 1. Compute % of real error in detected noisy labelled instances
    detected_noisy = df[df['is_issue'] == True]
    num_detected = len(detected_noisy)
    
    # A real error is where human_corrected_label is different from original_label
    # (and not NaN)
    real_errors = detected_noisy[detected_noisy['human_corrected_label'].notna() & 
                                (detected_noisy['human_corrected_label'] != detected_noisy['original_label'])]
    num_real_errors = len(real_errors)
    
    perc_real_error = (num_real_errors / num_detected * 100) if num_detected > 0 else 0
    print(f"Detected noisy instances: {num_detected}")
    print(f"Real errors found: {num_real_errors}")
    print(f"Percentage of real error in detected noisy instances: {perc_real_error:.2f}%")
    
    # 2. Label distribution of the errored (original) and corrected labels
    if num_real_errors > 0:
        print("\nLabel distribution of real errors:")
        orig_dist = real_errors['original_label'].value_counts()
        corr_dist = real_errors['human_corrected_label'].value_counts()
        
        dist_df = pd.DataFrame({
            'Original (Errored)': orig_dist,
            'Corrected': corr_dist
        }).fillna(0).astype(int)
        
        print(dist_df)
    else:
        print("\nNo real errors found to show distribution.")
    
    # 3. Compute accuracy and macro-F1 score
    ground_truth = df['human_corrected_label'].combine_first(df['original_label'])
    predictions = df['predicted_label']
    
    valid_indices = predictions.notna() & ground_truth.notna()
    y_true = ground_truth[valid_indices]
    y_pred = predictions[valid_indices]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\nPerformance against ground truth (corrected labels):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    files = {
        "SloBERTa": "sloberta_results_corrected_labels.ods",
        "CroSloEngual-BERT": "crosloengual-bert_results_corrected_labels.ods"
    }
    
    for model, path in files.items():
        analyze_results(path, model)
