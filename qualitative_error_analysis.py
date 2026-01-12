"""
Qualitative Error Analysis for Sentiment Classification

This script analyzes misclassified instances to identify patterns
and potential error categories (irony, sarcasm, ambiguity, etc.)
"""

import sys
import io
import pandas as pd
import re
from collections import Counter

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load_misclassified(file_path):
    """Load misclassified instances from CSV."""
    return pd.read_csv(file_path, encoding='utf-8')


# ============================================================================
# ERROR CATEGORY DETECTION FUNCTIONS
# ============================================================================

def detect_irony_sarcasm(text):
    """Detect potential irony/sarcasm markers."""
    indicators = []
    text_lower = text.lower()
    
    # Emoticons and emoji-like patterns that may indicate sarcasm
    if re.search(r'[:;]-?[\)\(DPp]', text):
        indicators.append('emoticon')
    if ':D' in text or ':)' in text or ';)' in text:
        indicators.append('smiley')
    
    # Exclamation patterns (excessive or ironic)
    if '!!!' in text or '!!' in text:
        indicators.append('excessive_exclamation')
    
    # Ellipsis (often used for sarcasm/irony)
    if '...' in text:
        indicators.append('ellipsis')
    
    # Question marks with positive/negative words
    if '?' in text and any(w in text_lower for w in ['res', 'ja', 'seveda', 'itak']):
        indicators.append('rhetorical_question')
    
    # "Ha ha" or laughter
    if re.search(r'ha\s*ha|haha|hehe|lol|rofl', text_lower):
        indicators.append('laughter')
    
    # Quotation marks around words (often sarcastic)
    if re.search(r'["\"].*?["\"]', text):
        indicators.append('quoted_words')
    
    return indicators


def detect_mixed_sentiment(text):
    """Detect texts with mixed positive and negative signals."""
    text_lower = text.lower()
    
    positive_words = ['dober', 'odličn', 'super', 'bravo', 'hvala', 'lepo', 'srečn', 
                      'veseli', 'pohval', 'uspeh', 'zmaga', 'najboljš']
    negative_words = ['slab', 'žal', 'škoda', 'problem', 'slabo', 'napaka', 'kriza',
                      'propad', 'izgub', 'groz', 'nevarno', 'bedno']
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > 0 and neg_count > 0:
        return True
    return False


def detect_complex_syntax(text):
    """Detect complex sentence structures that may confuse models."""
    indicators = []
    
    # Very long text
    if len(text) > 500:
        indicators.append('very_long')
    
    # Multiple sentences
    sentence_count = len(re.split(r'[.!?]+', text))
    if sentence_count > 5:
        indicators.append('many_sentences')
    
    # Negation patterns
    if re.search(r'\bne\b|\bni\b|\bniso\b|\nnima\b|\bnobeden\b', text.lower()):
        indicators.append('negation')
    
    # Conditional statements
    if re.search(r'\bče\b|\bčeprav\b|\bampak\b|\btoda\b|\bsicer\b', text.lower()):
        indicators.append('conditional')
    
    return indicators


def detect_domain_specific(text):
    """Detect domain-specific content that may require context."""
    text_lower = text.lower()
    domains = []
    
    # Sports
    if any(w in text_lower for w in ['tekma', 'zmaga', 'gol', 'trener', 'reprezentanc', 
                                      'smučar', 'košark', 'nogomet', 'olimpij']):
        domains.append('sports')
    
    # Politics
    if any(w in text_lower for w in ['vlada', 'minister', 'politk', 'stranka', 
                                      'volitve', 'predsednik', 'levic', 'desnic']):
        domains.append('politics')
    
    # Economy/Finance
    if any(w in text_lower for w in ['cena', 'evro', 'banka', 'delnic', 'kredit',
                                      'bdp', 'inflacij', 'davek', 'proračun']):
        domains.append('economy')
    
    # Migration/Refugees
    if any(w in text_lower for w in ['begunec', 'migrant', 'meja', 'azil', 'tujec']):
        domains.append('migration')
    
    return domains


def detect_colloquial_language(text):
    """Detect informal/colloquial language."""
    text_lower = text.lower()
    indicators = []
    
    # Slang/informal expressions
    informal_words = ['fak', 'fej', 'kul', 'ful', 'kr', 'pač', 'itak', 'glih', 
                      'tko', 'bom', 'morš', 'nism', 'nebom', 'ajde', 'stari']
    if any(w in text_lower for w in informal_words):
        indicators.append('slang')
    
    # Social media style (@mentions)
    if '@' in text:
        indicators.append('social_media')
    
    # Dialectal forms
    if any(w in text_lower for w in ['kej', 'tam', 'tuki', 'zdej', 'zdle']):
        indicators.append('dialect')
    
    return indicators


def detect_short_ambiguous(text):
    """Detect very short texts that lack context."""
    word_count = len(text.split())
    if word_count <= 5:
        return True
    return False


def categorize_error(row):
    """Categorize a misclassification into potential error types."""
    text = row['text']
    true_label = row['true_label']
    pred_label = row['predicted_label']
    
    categories = []
    
    # Check for irony/sarcasm
    irony_markers = detect_irony_sarcasm(text)
    if irony_markers:
        categories.append(('irony_sarcasm', irony_markers))
    
    # Check for mixed sentiment
    if detect_mixed_sentiment(text):
        categories.append(('mixed_sentiment', []))
    
    # Check for complex syntax
    syntax_markers = detect_complex_syntax(text)
    if syntax_markers:
        categories.append(('complex_syntax', syntax_markers))
    
    # Check for domain-specific content
    domains = detect_domain_specific(text)
    if domains:
        categories.append(('domain_specific', domains))
    
    # Check for colloquial language
    colloquial = detect_colloquial_language(text)
    if colloquial:
        categories.append(('colloquial_language', colloquial))
    
    # Check for short/ambiguous texts
    if detect_short_ambiguous(text):
        categories.append(('short_ambiguous', []))
    
    # Specific confusion patterns
    confusion = f"{true_label}->{pred_label}"
    
    return {
        'categories': categories,
        'confusion_type': confusion,
        'text_length': len(text),
        'word_count': len(text.split())
    }


def analyze_misclassifications(df, version_name):
    """Perform comprehensive analysis of misclassifications."""
    print("=" * 80)
    print(f"QUALITATIVE ERROR ANALYSIS - {version_name}")
    print("=" * 80)
    print()
    
    # Basic stats
    print(f"Total misclassified instances: {len(df)}")
    print()
    
    # Confusion matrix breakdown
    print("-" * 80)
    print("CONFUSION PATTERNS")
    print("-" * 80)
    confusion_counts = df.groupby(['true_label', 'predicted_label']).size()
    print(confusion_counts.to_string())
    print()
    
    # Apply categorization to all instances
    results = df.apply(categorize_error, axis=1)
    df['analysis'] = results
    
    # Extract categories
    all_categories = []
    category_examples = {}
    
    for idx, row in df.iterrows():
        analysis = row['analysis']
        for cat, markers in analysis['categories']:
            all_categories.append(cat)
            if cat not in category_examples:
                category_examples[cat] = []
            if len(category_examples[cat]) < 3:  # Keep up to 3 examples per category
                category_examples[cat].append({
                    'text': row['text'][:100] + '...' if len(row['text']) > 100 else row['text'],
                    'true': row['true_label'],
                    'pred': row['predicted_label'],
                    'markers': markers
                })
    
    # Category distribution
    print("-" * 80)
    print("ERROR CATEGORY DISTRIBUTION")
    print("-" * 80)
    cat_counts = Counter(all_categories)
    for cat, count in cat_counts.most_common():
        percentage = (count / len(df)) * 100
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    print()
    
    # Example instances for each category
    print("-" * 80)
    print("EXAMPLE INSTANCES BY CATEGORY")
    print("-" * 80)
    
    for cat in cat_counts.keys():
        print(f"\n### {cat.upper().replace('_', ' ')} ###")
        for i, ex in enumerate(category_examples.get(cat, [])[:2], 1):
            print(f"  Example {i}:")
            print(f"    Text: {ex['text']}")
            print(f"    True: {ex['true']} | Predicted: {ex['pred']}")
            if ex['markers']:
                print(f"    Markers: {', '.join(ex['markers'])}")
    
    print()
    
    # Text length analysis
    print("-" * 80)
    print("TEXT LENGTH ANALYSIS")
    print("-" * 80)
    
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    # Group by confusion type and analyze length
    print("\nAverage word count by confusion type:")
    for (true_l, pred_l), group in df.groupby(['true_label', 'predicted_label']):
        avg_words = group['word_count'].mean()
        print(f"  {true_l} -> {pred_l}: {avg_words:.1f} words (n={len(group)})")
    
    print()
    
    return df, cat_counts, category_examples


def generate_error_categories_summary():
    """Print a summary of proposed error categories."""
    print("=" * 80)
    print("PROPOSED ERROR CATEGORIES FOR SENTIMENT MISCLASSIFICATION")
    print("=" * 80)
    print()
    
    categories = {
        "1. IRONY/SARCASM": {
            "description": "Text contains ironic or sarcastic expressions that flip the literal meaning",
            "indicators": ["Emoticons with contrasting text", "Excessive punctuation", 
                          "Laughter markers (haha)", "Rhetorical questions"],
            "example": "'Wauu, za 50 litrov bom pridobil celih 30 centov. Hvala! :)' - labeled negative, predicted positive"
        },
        "2. MIXED SENTIMENT": {
            "description": "Text contains both positive and negative elements, making overall sentiment ambiguous",
            "indicators": ["Positive and negative words in same text", "Praise followed by criticism",
                          "Comparisons with both good and bad aspects"],
            "example": "'Borili so se. ... Jaz sem z veseljem gledal... Drugi polčas malo slabša igra.'"
        },
        "3. COMPLEX SYNTAX/NEGATION": {
            "description": "Complex sentence structures, negations, or conditionals that change meaning",
            "indicators": ["Double negatives", "Conditional statements (če, ampak, toda)",
                          "Long multi-sentence texts", "Embedded clauses"],
            "example": "'Nepremičninski pok se nikoli ne bo zgodil' - negation changes meaning"
        },
        "4. DOMAIN-SPECIFIC KNOWLEDGE": {
            "description": "Requires domain knowledge (sports, politics, economy) to interpret correctly",
            "indicators": ["Sports commentary", "Political references", "Economic jargon",
                          "Cultural references specific to Slovenia"],
            "example": "Sports comments about team performance requiring context"
        },
        "5. COLLOQUIAL/INFORMAL LANGUAGE": {
            "description": "Slang, dialectal forms, or social media style that may not be in training data",
            "indicators": ["Slovenian slang (fak, ful, itak)", "@mentions",
                          "Dialectal spelling", "Abbreviations"],
            "example": "'fak stari.... to pa je pravljica.....' - slang expression"
        },
        "6. SHORT/AMBIGUOUS": {
            "description": "Very short texts lacking sufficient context for reliable classification",
            "indicators": ["Less than 5 words", "Single word/phrase",
                          "Missing context from conversation thread"],
            "example": "'A je to.' - too short to determine sentiment"
        },
        "7. RHETORICAL DEVICES": {
            "description": "Use of rhetorical questions, understatement, or other literary devices",
            "indicators": ["Questions used as statements", "Understatement for emphasis",
                          "Hyperbole"],
            "example": "'Zdaj smo vsi bogati?' - rhetorical question expressing skepticism"
        },
        "8. CONTEXT-DEPENDENT": {
            "description": "Meaning depends on broader context, conversation, or current events",
            "indicators": ["References to other comments (@user)", "Follow-up messages",
                          "References to news events"],
            "example": "Replies referencing previous comments in a thread"
        }
    }
    
    for cat_name, details in categories.items():
        print(f"\n{cat_name}")
        print(f"  Description: {details['description']}")
        print(f"  Indicators: {', '.join(details['indicators'][:3])}")
        print(f"  Example: {details['example'][:80]}...")
    
    return categories


def main():
    print("\n" + "=" * 80)
    print("MISCLASSIFIED INSTANCES QUALITATIVE ANALYSIS")
    print("=" * 80 + "\n")
    
    # Load both versions
    try:
        df_v0 = load_misclassified('missclassified_instances_V0.csv')
        print(f"Loaded V0: {len(df_v0)} misclassified instances")
    except FileNotFoundError:
        print("V0 file not found")
        df_v0 = None
    
    try:
        df_v1 = load_misclassified('misclassified_instances_V1.csv')
        print(f"Loaded V1: {len(df_v1)} misclassified instances")
    except FileNotFoundError:
        print("V1 file not found")
        df_v1 = None
    
    print()
    
    # Generate error categories summary first
    categories = generate_error_categories_summary()
    
    # Analyze V0
    if df_v0 is not None:
        print("\n")
        df_v0_analyzed, cat_v0, examples_v0 = analyze_misclassifications(df_v0, "V0 (Original Dataset)")
    
    # Analyze V1
    if df_v1 is not None:
        print("\n")
        df_v1_analyzed, cat_v1, examples_v1 = analyze_misclassifications(df_v1, "V1 (Corrected Dataset)")
    
    # Comparison if both available
    if df_v0 is not None and df_v1 is not None:
        print("\n" + "=" * 80)
        print("COMPARISON V0 vs V1")
        print("=" * 80)
        print(f"\nV0 misclassifications: {len(df_v0)}")
        print(f"V1 misclassifications: {len(df_v1)}")
        print(f"Difference: {len(df_v1) - len(df_v0)} ({(len(df_v1) - len(df_v0))/len(df_v0)*100:+.1f}%)")
        
        # Compare category distributions
        print("\nCategory comparison (V0 -> V1):")
        all_cats = set(cat_v0.keys()) | set(cat_v1.keys())
        for cat in sorted(all_cats):
            v0_count = cat_v0.get(cat, 0)
            v1_count = cat_v1.get(cat, 0)
            v0_pct = (v0_count / len(df_v0)) * 100 if len(df_v0) > 0 else 0
            v1_pct = (v1_count / len(df_v1)) * 100 if len(df_v1) > 0 else 0
            print(f"  {cat}: {v0_count} ({v0_pct:.1f}%) -> {v1_count} ({v1_pct:.1f}%)")
    
    # Save detailed analysis to CSV
    if df_v1 is not None:
        # Extract categories as string for saving
        df_v1['error_categories'] = df_v1['analysis'].apply(
            lambda x: '; '.join([c[0] for c in x['categories']]) if x else ''
        )
        df_v1_export = df_v1[['text', 'true_label', 'predicted_label', 'word_count', 'error_categories']]
        df_v1_export.to_csv('qualitative_error_analysis_V1.csv', index=False, encoding='utf-8')
        print("\n\nDetailed V1 analysis saved to: qualitative_error_analysis_V1.csv")
    
    if df_v0 is not None:
        df_v0['error_categories'] = df_v0['analysis'].apply(
            lambda x: '; '.join([c[0] for c in x['categories']]) if x else ''
        )
        df_v0_export = df_v0[['text', 'true_label', 'predicted_label', 'word_count', 'error_categories']]
        df_v0_export.to_csv('qualitative_error_analysis_V0.csv', index=False, encoding='utf-8')
        print("Detailed V0 analysis saved to: qualitative_error_analysis_V0.csv")


if __name__ == "__main__":
    main()
