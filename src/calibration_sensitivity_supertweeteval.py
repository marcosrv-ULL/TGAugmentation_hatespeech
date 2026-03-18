#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

# ==============================
#  Limpieza de Texto
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
#  Carga de Datos
# ==============================
def load_and_clean(path: str, text_field: str) -> pd.DataFrame:
    print(f"Cargando: {path} ...")
    df = pd.read_json(path, lines=True)
    df = df[df[text_field].apply(lambda x: isinstance(x, str) and x.strip() != "")].copy()
    df = df.dropna(subset=["predicted_hate_category"]).copy()
    
    # Limpieza
    df["clean"] = df[text_field].apply(clean_text)
    df["phc"] = df["predicted_hate_category"].astype(str).str.strip().str.lower()
    df = df[df["phc"] != "nan"]
    return df

# ==============================
#  Análisis de Sensibilidad
# ==============================
def analyze_threshold_sensitivity(model, X, y, label2id, output_prefix, model_name="Model"):
    # 1. Definir índice para "Rechazado/None"
    none_labels = ["none", "not_hate", "neutral", "nan"]
    none_idx = -1
    for lbl in none_labels:
        if lbl in label2id:
            none_idx = label2id[lbl]
            break
            
    if none_idx == -1:
        none_idx = max(label2id.values()) + 1
    
    print(f"\nCalculando probabilidades para {model_name}...")
    probs = model.predict_proba(X) 
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    f1_scores = []
    accuracies = []
    
    for t in thresholds:
        raw_preds = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        
        # Filtro: Si confianza < t -> none_idx
        final_preds = np.where(max_probs < t, none_idx, raw_preds)
        
        f1 = f1_score(y, final_preds, average="macro")
        acc = accuracy_score(y, final_preds)
        f1_scores.append(f1)
        accuracies.append(acc)

    # Crear DataFrame
    df_res = pd.DataFrame({
        "Threshold": thresholds,
        "Macro-F1": f1_scores,
        "Accuracy": accuracies
    })
    
    # --- MOSTRAR TABLA EN CONSOLA ---
    print("\n" + "="*40)
    print(f"RESULTADOS: {model_name}")
    print("="*40)
    print(df_res.round(4).to_string(index=False))
    print("="*40 + "\n")

    # Guardar CSV
    csv_path = f"{output_prefix}.csv"
    df_res.to_csv(csv_path, index=False)
    
    # Graficar
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, marker='o', linewidth=2, label='Macro-F1')
    plt.plot(thresholds, accuracies, marker='s', linestyle='--', color='grey', label='Accuracy')
    
    plt.title(f"Threshold Sensitivity ({model_name})")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Guardar en PNG (para vista rápida)
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    
    # Guardar en PDF (CALIDAD PAPER - VECTORIAL)
    plt.savefig(f"{output_prefix}.pdf", format='pdf', bbox_inches='tight')
    
    plt.close() 
    print(f"Gráficas guardadas en: {output_prefix}.png y .pdf")

# ==============================
#  Main
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="merged_masked_unionv2_new/merged_masked_unionv2_new_train.jsonl")
    parser.add_argument("--test_json", type=str, default="merged_masked_unionv2_new/merged_masked_unionv2_new_test.jsonl")
    parser.add_argument("--text_field", type=str, default="text_masked")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Cargar
    df_train = load_and_clean(args.input_json, args.text_field)
    df_test = load_and_clean(args.test_json, args.text_field)
    
    # Etiquetas
    all_labels = sorted(set(df_train["phc"]) | set(df_test["phc"]))
    label2id = {l: i for i, l in enumerate(all_labels)}
    
    y_train = df_train["phc"].map(label2id).values
    y_test  = df_test["phc"].map(label2id).values
    
    # Vectorizar
    print("Vectorizando...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.90, sublinear_tf=True, stop_words="english")
    X_train = vectorizer.fit_transform(df_train["clean"])
    X_test  = vectorizer.transform(df_test["clean"])
    
    # ---------------------------------------------------------
    # 1. LOGISTIC REGRESSION (Tu configuración óptima)
    # ---------------------------------------------------------
    print("\n>>> Entrenando Logistic Regression...")
    clf_lr = LogisticRegression(
        max_iter=3000, 
        random_state=args.seed, 
        n_jobs=-1,
        solver='liblinear',
        multi_class='ovr',
        C=10.0  # Confianza alta
    )
    clf_lr.fit(X_train, y_train)
    analyze_threshold_sensitivity(clf_lr, X_test, y_test, label2id, "threshold_sensitivity_logreg", "LogisticRegression")

    # ---------------------------------------------------------
    # 2. SGD CLASSIFIER (Tuneado para imitar la confianza alta)
    # ---------------------------------------------------------
    print("\n>>> Entrenando SGD Classifier...")
    # NOTA: alpha=1e-5 es una regularización baja (equivale a C alto), 
    # lo que permite al modelo tener probabilidades más extremas (confiadas).
    clf_sgd = SGDClassifier(
        loss='log_loss',       # Necesario para probabilidades
        alpha=1e-5,            # Regularización baja (Robustez)
        max_iter=3000,
        tol=1e-3,
        random_state=args.seed,
        n_jobs=-1
    )
    clf_sgd.fit(X_train, y_train)
    analyze_threshold_sensitivity(clf_sgd, X_test, y_test, label2id, "threshold_sensitivity_sgd", "SGDClassifier")

if __name__ == "__main__":
    main()