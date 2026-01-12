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

def load_and_normalize(file_path):
    """Load ODS file and normalize labels."""
    df = pd.read_excel(file_path, engine='odf')
    
    label_map = {
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        'poisitve': 'positive'
    }
    
    df['original_label'] = df['original_label'].map(lambda x: label_map.get(str(x).strip().lower(), x))
    df['predicted_label'] = df['predicted_label'].map(lambda x: label_map.get(str(x).strip().lower(), x))
    df['human_corrected_label'] = df['human_corrected_label'].map(
        lambda x: label_map.get(str(x).strip().lower(), x) if pd.notna(x) else x
    )
    return df


def get_real_errors(df):
    """
    Get rows that are real errors:
    - Has human_corrected_label (not NaN/empty)
    - human_corrected_label differs from original_label
    """
    has_correction = df['human_corrected_label'].notna() & (df['human_corrected_label'] != '')
    is_real_error = df['human_corrected_label'] != df['original_label']
    return df[has_correction & is_real_error]


def get_noisy_instances(df):
    """Get rows flagged as noisy (is_issue = True)."""
    return df[df['is_issue'] == True]


def analyze_intersection(df_crosloe, df_sloberta):
    """Analyze the intersection of real errors between both models."""
    
    # Get noisy instances (is_issue=True)
    crosloe_noisy = get_noisy_instances(df_crosloe)
    sloberta_noisy = get_noisy_instances(df_sloberta)
    
    # Get real errors (human_corrected_label != original_label)
    crosloe_errors = get_real_errors(df_crosloe)
    sloberta_errors = get_real_errors(df_sloberta)
    
    # Get sets of texts with real errors
    crosloe_error_texts = set(crosloe_errors['text'].tolist())
    sloberta_error_texts = set(sloberta_errors['text'].tolist())
    
    # Calculate set operations for real errors
    intersection = crosloe_error_texts & sloberta_error_texts
    only_crosloe = crosloe_error_texts - sloberta_error_texts
    only_sloberta = sloberta_error_texts - crosloe_error_texts
    
    # Also get noisy instance texts for intersection
    crosloe_noisy_texts = set(crosloe_noisy['text'].tolist())
    sloberta_noisy_texts = set(sloberta_noisy['text'].tolist())
    noisy_intersection = crosloe_noisy_texts & sloberta_noisy_texts
    
    return {
        'crosloe_noisy_count': len(crosloe_noisy),
        'sloberta_noisy_count': len(sloberta_noisy),
        'noisy_intersection_count': len(noisy_intersection),
        'crosloe_error_count': len(crosloe_errors),
        'sloberta_error_count': len(sloberta_errors),
        'error_intersection_count': len(intersection),
        'only_crosloe_count': len(only_crosloe),
        'only_sloberta_count': len(only_sloberta),
        'crosloe_is_superset_of_sloberta': sloberta_error_texts.issubset(crosloe_error_texts),
        'sloberta_is_superset_of_crosloe': crosloe_error_texts.issubset(sloberta_error_texts),
        'intersection_texts': intersection,
        'only_crosloe_texts': only_crosloe,
        'only_sloberta_texts': only_sloberta,
    }


def create_intersection_dataframe(df_crosloe, df_sloberta, intersection_texts):
    """Create a DataFrame comparing both models for intersection texts."""
    crosloe_intersection = df_crosloe[df_crosloe['text'].isin(intersection_texts)].copy()
    sloberta_intersection = df_sloberta[df_sloberta['text'].isin(intersection_texts)].copy()
    
    crosloe_intersection = crosloe_intersection.rename(columns={
        'predicted_label': 'crosloe_predicted',
        'human_corrected_label': 'crosloe_corrected'
    })
    
    sloberta_intersection = sloberta_intersection.rename(columns={
        'predicted_label': 'sloberta_predicted',
        'human_corrected_label': 'sloberta_corrected'
    })
    
    merged = crosloe_intersection[['text', 'original_label', 'crosloe_predicted', 'crosloe_corrected']].merge(
        sloberta_intersection[['text', 'sloberta_predicted', 'sloberta_corrected']],
        on='text',
        how='inner'
    )
    
    merged['corrections_agree'] = merged['crosloe_corrected'] == merged['sloberta_corrected']
    return merged


def print_intersection_results(results):
    """Print the intersection analysis results."""
    print("=" * 70)
    print("REAL ERRORS INTERSECTION ANALYSIS")
    print("CroSloEngual BERT vs SloBERTa")
    print("=" * 70)
    print()
    
    print("-" * 70)
    print("DETECTED NOISY INSTANCES (is_issue=True)")
    print("-" * 70)
    print(f"CroSloEngual BERT: {results['crosloe_noisy_count']}")
    print(f"SloBERTa:          {results['sloberta_noisy_count']}")
    print(f"Intersection:      {results['noisy_intersection_count']}")
    print()
    
    print("-" * 70)
    print("REAL ERRORS (human_corrected_label != original_label)")
    print("-" * 70)
    print(f"CroSloEngual BERT real errors: {results['crosloe_error_count']}")
    print(f"SloBERTa real errors:          {results['sloberta_error_count']}")
    print()
    
    print("-" * 70)
    print("INTERSECTION OF REAL ERRORS")
    print("-" * 70)
    print(f"Both models found as error:    {results['error_intersection_count']}")
    print(f"Only CroSloEngual BERT:        {results['only_crosloe_count']}")
    print(f"Only SloBERTa:                 {results['only_sloberta_count']}")
    print()
    
    print("-" * 70)
    print("SUPERSET CHECK (for real errors)")
    print("-" * 70)
    print(f"Is CroSloEngual BERT a superset of SloBERTa? {results['crosloe_is_superset_of_sloberta']}")
    print(f"Is SloBERTa a superset of CroSloEngual BERT? {results['sloberta_is_superset_of_crosloe']}")
    print()


def run_intersection_analysis(crosloe_path, sloberta_path):
    """Run the full intersection analysis and save results."""
    print("\nLoading data for intersection analysis...")
    df_crosloe = load_and_normalize(crosloe_path)
    df_sloberta = load_and_normalize(sloberta_path)
    
    print("Analyzing intersection of real errors...")
    results = analyze_intersection(df_crosloe, df_sloberta)
    
    print_intersection_results(results)
    
    if results['error_intersection_count'] > 0:
        intersection_df = create_intersection_dataframe(
            df_crosloe, df_sloberta, results['intersection_texts']
        )
        
        agreement_count = intersection_df['corrections_agree'].sum()
        print("-" * 70)
        print("CORRECTION AGREEMENT (for intersection)")
        print("-" * 70)
        print(f"Corrections agree:    {agreement_count}")
        print(f"Corrections disagree: {len(intersection_df) - agreement_count}")
        print()
        
        # Save CSV files
        intersection_df.to_csv('real_errors_intersection.csv', index=False, encoding='utf-8')
        print("Intersection data saved to: real_errors_intersection.csv")
        
        only_crosloe_df = df_crosloe[df_crosloe['text'].isin(results['only_crosloe_texts'])]
        only_crosloe_df.to_csv('real_errors_only_crosloengual.csv', index=False, encoding='utf-8')
        print("Only-CroSloEngual errors saved to: real_errors_only_crosloengual.csv")
        
        only_sloberta_df = df_sloberta[df_sloberta['text'].isin(results['only_sloberta_texts'])]
        only_sloberta_df.to_csv('real_errors_only_sloberta.csv', index=False, encoding='utf-8')
        print("Only-SloBERTa errors saved to: real_errors_only_sloberta.csv")
        
        # Show disagreements
        disagreements = intersection_df[~intersection_df['corrections_agree']]
        if len(disagreements) > 0:
            print()
            print("-" * 70)
            print(f"DISAGREEMENTS ({len(disagreements)} instances with different corrections)")
            print("-" * 70)
            for idx, row in disagreements.iterrows():
                text_preview = row['text'][:70] if len(row['text']) > 70 else row['text']
                print(f"\nText: {text_preview}...")
                print(f"  Original label:      {row['original_label']}")
                print(f"  CroSloEngual BERT:   {row['crosloe_corrected']}")
                print(f"  SloBERTa:            {row['sloberta_corrected']}")


if __name__ == "__main__":
    import sys
    import io
    # Fix Windows console encoding for Slovenian characters
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    files = {
        "SloBERTa": "sloberta_results_corrected_labels.ods",
        "CroSloEngual-BERT": "crosloengual-bert_results_corrected_labels.ods"
    }
    
    # Individual model analysis
    for model, path in files.items():
        analyze_results(path, model)
    
    # Intersection analysis between models
    run_intersection_analysis(
        files["CroSloEngual-BERT"],
        files["SloBERTa"]
    )
