# Slovenian Sentiment Analysis: A Multi-Model Comparative Study

<div align="center">

**A comprehensive research project comparing BERT-based models and Large Language Models for Slovenian sentiment analysis with a focus on label noise detection and correction.**

[Features](#-features) • [Models](#-models) • [Dataset](#-dataset) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Citation](#-citation)

</div>

---

## 📋 Overview

This project presents a systematic comparison of **transformer-based models** for sentiment analysis on **Slovenian text**, addressing the unique challenges of a low-resource Slavic language. The research includes:

- **Comparative evaluation** of SloBERTa, CroSloEngual-BERT, and GaMS-2B-Instruct
- **Label noise detection** using cleanlab and confidence-based methods
- **Dataset correction** with human-verified annotations
- **Qualitative error analysis** identifying patterns in misclassifications
- **Fine-tuning experiments** with LoRA adapters

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Multi-Model Comparison** | Side-by-side evaluation of BERT variants and LLMs |
| 🧹 **Noisy Label Detection** | Automated identification of annotation errors |
| 📊 **Cross-Validation** | Robust 5-fold stratified evaluation |
| 🔬 **Error Analysis** | Qualitative categorization of misclassifications |
| 🚀 **LLM Fine-Tuning** | LoRA-based adaptation of GaMS-2B |
| 📈 **Zero-Shot Evaluation** | Testing generalization without task-specific training |

## 🤖 Models

### BERT-Based Models

| Model | Description | Source |
|-------|-------------|--------|
| **SloBERTa** | Monolingual Slovenian RoBERTa | [EMBEDDIA/sloberta-base](https://huggingface.co/EMBEDDIA/sloberta-base) |
| **CroSloEngual-BERT** | Trilingual (Croatian, Slovenian, English) BERT | [EMBEDDIA/crosloengual-bert](https://huggingface.co/EMBEDDIA/crosloengual-bert) |

### Large Language Models

| Model | Description | Source |
|-------|-------------|--------|
| **GaMS-2B-Instruct** | Slovenian 2B parameter instruction-tuned LLM | [cjvt/GaMS-2B-Instruct](https://huggingface.co/cjvt/GaMS-2B-Instruct) |

## 📁 Dataset

### KKS Opinion Corpus (v1.001)

The primary dataset is the **KKS Sentiment Annotated Corpus**, a Slovenian opinion corpus containing:

- **~3,500 annotated instances**
- **3 sentiment classes:** Positive, Negative, Neutral
- **Domain:** User-generated content (reviews, comments)
- **Format:** XML with sentence-level annotations

### Dataset Versions

| Version | Description | File |
|---------|-------------|------|
| V0 (Original) | Raw corpus labels | `kks_v0_original.csv` |
| V1 (Corrected) | Human-corrected labels after noise detection | `kks_v1_corrected.csv` |


## 📊 Results

### Model Comparison

| Model | Accuracy | Macro-F1 | Dataset |
|-------|----------|----------|---------|
| SloBERTa | 0.78 | 0.76 | V0 |
| CroSloEngual-BERT | 0.77 | 0.75 | V0 |
| GaMS (Zero-Shot) | 0.65 | 0.62 | V0 |
| GaMS (Fine-Tuned V0) | 0.79 | 0.77 | V0 |
| GaMS (Fine-Tuned V1) | 0.82 | 0.80 | V1 |

### Label Noise Analysis

- **Noisy labels detected:** ~150 instances (4.3% of corpus)
- **Correction method:** Human review of high-confidence model disagreements
- **Impact:** +3-5% improvement in model performance on corrected labels

### Error Categories

The qualitative analysis identified these patterns in misclassifications:

| Category | Description | Frequency |
|----------|-------------|-----------|
| Irony/Sarcasm | Surface sentiment contradicts intent | 28% |
| Mixed Sentiment | Multiple conflicting signals | 22% |
| Domain-Specific | Requires contextual knowledge | 18% |
| Colloquial Language | Informal expressions | 15% |
| Short/Ambiguous | Insufficient context | 12% |
| Complex Syntax | Negation, conditionals | 5% |

## Fine-tuned GaMS-2B models

Available on HuggingFace:
- GaMS-2B model finetuned on original KKS dataset: https://huggingface.co/lea-vodopivec7/gams-2b-finetuned-kks-V0
- GaMS-2B model finetuned on corrected KKS dataset: https://huggingface.co/lea-vodopivec7/gams-2b-finetuned-kks-V1


## 📂 Project Structure

```
Modelling_the_Slovenian_sentiment_analysis/
├── 📄 classifier_sloberta.py          # SloBERTa training & noise detection
├── 📄 classifier_crosloengual-bert.py # CroSloEngual-BERT training
├── 📄 compare_bert_results.py          # Model comparison analysis
├── 📄 zero_shot_performance_gams.py    # GaMS zero-shot evaluation
├── 📄 gams_finetune_original_KKS_dataset.py   # GaMS fine-tuning (V0)
├── 📄 gams_finetune_corrected_KKS_dataset.py  # GaMS fine-tuning (V1)
├── 📄 qualitative_error_analysis.py    # Error pattern analysis
├── 📄 noisy_labels_kks.py              # TF-IDF based noise detection
├── 📄 prepare_datasets.py              # Dataset preprocessing utilities
│
├── 📁 klxSAcorpus_20160224_1001/       # Original KKS corpus
├── 📁 gams-2b-finetuned-kks-V0/        # Fine-tuned model (original)
├── 📁 gams-2b-finetuned-kks-V1/        # Fine-tuned model (corrected)
│
├── 📊 kks_opinion_corpus.csv           # Base dataset
├── 📊 kks_v0_original.csv              # Original labels
├── 📊 kks_v1_corrected.csv             # Corrected labels
├── 📊 misclassified_instances_*.csv    # Error analysis data
├── 📊 qualitative_error_analysis_*.csv # Categorized errors
│
├── 📋 Report.pdf                        # Research report
└── 📖 README.md                         # This file

```

