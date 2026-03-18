#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compare classification errors between Raw Text and Masked Text.
It identifies specific examples where:
1. Masking helped (Raw failed, Masked predicted correctly).
2. Masking hurt (Raw predicted correctly, Masked failed).
"""

import argparse
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
#  Text Cleaning
# ==============================
URL_RE     = re.compile(r"(https?://\S+|www\.\S+|\{URL\})", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = URL_RE.sub(" [URL] ", s)
    s = MENTION_RE.sub(" [MENTION] ", s)
    s = HASHTAG_RE.sub(r" [HASHTAG:\1] ", s)
    s = re.sub(r"\s+", " ", s)
    return s

# ==============================
#  Data Loading
# ==============================
def load_data(path: str):
    print(f"Loading {path}...")
    df = pd.read_json(path, lines=True)
    
    # Ensure both columns exist
    if "text" not in df.columns or "text_masked" not in df.columns:
        raise ValueError(f"Dataset must contain both 'text' and 'text_masked' columns. Found: {df.columns}")

    # Drop missing targets
    df = df.dropna(subset=["predicted_hate_category"])
    df = df[df["predicted_hate_category"].astype(str).str.lower() != "nan"]
    
    # Clean both versions
    df["clean_raw"] = df["text"].apply(clean_text)
    df["clean_mask"] = df["text_masked"].apply(clean_text)
    
    # Normalize Label
    df["label_str"] = df["predicted_hate_category"].astype(str).str.strip().str.lower()
    
    return df

# ==============================
#  Training Helper
# ==============================
def train_and_predict(df_train, df_test, col_name, label2id):
    print(f"\n--- Training on column: '{col_name}' ---")
    
    y_train = df_train["label_str"].map(label2id).values
    
    # Vectorization
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.90, sublinear_tf=True, stop_words="english")
    X_train = vec.fit_transform(df_train[col_name])
    X_test  = vec.transform(df_test[col_name])
    
    # Model (Logistic Regression - Deterministic and fast)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42, solver='liblinear')
    clf.fit(X_train, y_train)
    
    preds_idx = clf.predict(X_test)
    
    # Map back to string labels for easier reading
    id2label = {v: k for k, v in label2id.items()}
    preds_str = [id2label[i] for i in preds_idx]
    
    acc = accuracy_score(df_test["label_str"].map(label2id), preds_idx)
    print(f"Accuracy on '{col_name}': {acc:.4f}")
    
    return preds_str

# ==============================
#  Main Analysis
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="error_analysis_examples.csv")
    args = parser.parse_args()

    # 1. Load Data
    df_train = load_data(args.input_json)
    df_test = load_data(args.test_json)
    
    # Label Mapping
    all_labels = sorted(set(df_train["label_str"]) | set(df_test["label_str"]))
    label2id = {l: i for i, l in enumerate(all_labels)}
    print(f"Labels: {label2id}")

    # 2. Train & Predict RAW
    preds_raw = train_and_predict(df_train, df_test, "clean_raw", label2id)
    
    # 3. Train & Predict MASKED
    preds_mask = train_and_predict(df_train, df_test, "clean_mask", label2id)
    
    # 4. Compare results
    df_test["Pred_Raw"] = preds_raw
    df_test["Pred_Mask"] = preds_mask
    
    # Logic:
    # GT = Ground Truth
    
    # Case A: MASKING WINS (Raw wrong, Mask right) -> "How masking contributes"
    mask_wins = df_test[
        (df_test["label_str"] != df_test["Pred_Raw"]) & 
        (df_test["label_str"] == df_test["Pred_Mask"])
    ].copy()
    
    # Case B: MASKING FAILS (Raw right, Mask wrong) -> "Where approach fails"
    mask_fails = df_test[
        (df_test["label_str"] == df_test["Pred_Raw"]) & 
        (df_test["label_str"] != df_test["Pred_Mask"])
    ].copy()

    print("\n" + "="*40)
    print("ANALYSIS SUMMARY")
    print("="*40)
    print(f"Total Test Samples: {len(df_test)}")
    print(f"Masking WINS (Helped): {len(mask_wins)} examples")
    print(f"Masking FAILS (Hurt):  {len(mask_fails)} examples")
    
    # Select columns for export
    cols_to_save = ["text", "text_masked", "label_str", "Pred_Raw", "Pred_Mask"]
    
    # Save combined file
    mask_wins["Type"] = "Masking_HELPED"
    mask_fails["Type"] = "Masking_HURT"
    
    df_final = pd.concat([mask_wins, mask_fails])
    df_final = df_final[["Type"] + cols_to_save]
    
    df_final.to_csv(args.output_csv, index=False)
    print(f"\nDetailed examples saved to: {args.output_csv}")
    
    # --- SHOW EXAMPLES IN CONSOLE ---
    print("\n--- EXAMPLES: Masking HELPED (Raw confused, Mask clear) ---")
    for _, row in mask_wins.head(3).iterrows():
        print(f"GT: {row['label_str']} | RawPred: {row['Pred_Raw']} -> MaskPred: {row['Pred_Mask']}")
        print(f"Raw:  {row['text']}")
        print(f"Mask: {row['text_masked']}")
        print("-" * 20)

    print("\n--- EXAMPLES: Masking FAILED (Context lost) ---")
    for _, row in mask_fails.head(3).iterrows():
        print(f"GT: {row['label_str']} | RawPred: {row['Pred_Raw']} -> MaskPred: {row['Pred_Mask']}")
        print(f"Raw:  {row['text']}")
        print(f"Mask: {row['text_masked']}")
        print("-" * 20)

if __name__ == "__main__":
    main()