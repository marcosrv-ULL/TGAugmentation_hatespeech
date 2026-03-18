#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment script for multiclass hate speech detection with:
- Linear TF-IDF models (LogReg, Linear SVM, SGD, PA, ComplementNB)
- DistilBERT
- XLM-R (base; used as "small" multilingual transformer)

It expects two JSONL files in SuperTweetEval-like format:
    - train JSONL (with augmented or original data)
    - test JSONL (only originals)

It will:
- load and clean the data,
- optionally drop "hate_age" and collapse "hate_race" into "hate_origin",
- run multiple seeds,
- train/evaluate linear and transformer models,
- aggregate metrics (macro-F1 + accuracy) with 95% CIs,
- save per-run and summary CSVs under final_out/.
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

# Transformers imports (for DistilBERT / XLM-R)
try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        set_seed as hf_set_seed,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ==============================
#  Text cleaning
# ==============================

URL_RE     = re.compile(r"(https?://\S+|www\.\S+|\{URL\})", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = URL_RE.sub(" [URL] ", s)
    s = MENTION_RE.sub(" [MENTION] ", s)
    s = HASHTAG_RE.sub(r" [HASHTAG:\1] ", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ==============================
#  Helpers
# ==============================

def eval_split(model, X, y, split_name: str = "VAL") -> Dict[str, float]:
    """
    Evalúa un modelo sklearn y devuelve métricas clave.
    También imprime informe detallado (como en tu notebook actual).
    """
    preds = model.predict(X)

    f1m = f1_score(y, preds, average="macro")
    acc = accuracy_score(y, preds)

    print(f"\n=== {split_name} ===")
    print(classification_report(y, preds, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y, preds))
    print("F1-macro:", f"{f1m:.4f}")
    print("Accuracy:", f"{acc:.4f}")

    return {
        "f1_macro": f1m,
        "accuracy": acc,
    }


def load_any(path: str) -> pd.DataFrame:
    """
    Carga un .csv o un .jsonl y devuelve un DataFrame.
    """
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Formato no soportado: {path}. Usa .csv o .jsonl")



def load_and_clean_tweethate(path: str, text_field: str = "text") -> pd.DataFrame:
    """
    Carga un JSONL tipo SuperTweetEval, filtra texto vacío,
    elimina NaN reales y 'nan' como string en predicted_hate_category,
    genera columna 'clean' con texto limpio.

    Devuelve:
        df : DataFrame con columnas:
             - text_field
             - predicted_hate_category
             - clean (texto limpio)
    """
    df = load_any(path)

    # 1) Filtrar texto no vacío
    df = df[df[text_field].apply(lambda x: isinstance(x, str) and x.strip() != "")].copy()

    # 2) Eliminar NaN reales en predicted_hate_category
    df = df.dropna(subset=["predicted_hate_category"]).copy()

    # 3) Eliminar "nan" como texto
    df = df[df["predicted_hate_category"].astype(str).str.lower() != "nan"].copy()

    # 4) Texto limpio
    df["clean"] = df[text_field].apply(clean_text)

    # 5) Normalizar columna de clase
    df["phc"] = df["predicted_hate_category"].astype(str).str.strip().str.lower()

    return df


def remove_values_from_column(df: pd.DataFrame, column: str, values_to_remove: List[str]) -> pd.DataFrame:
    remove_set = set(v.lower() for v in values_to_remove)
    df = df.copy()
    df[column] = df[column].astype(str)
    mask = ~df[column].str.lower().isin(remove_set)
    return df[mask].copy()


def collapse_race_into_origin(df: pd.DataFrame, col: str = "predicted_hate_category") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].replace({"hate_race": "hate_origin"})
    df["phc"] = df[col].astype(str).str.strip().str.lower()
    return df


def ci95(series: pd.Series) -> Tuple[float, float, float, float]:
    arr = np.array(series, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    if len(arr) > 1:
        ci_low  = mean - 1.96 * std / np.sqrt(len(arr))
        ci_high = mean + 1.96 * std / np.sqrt(len(arr))
    else:
        ci_low = ci_high = mean
    return mean, std, ci_low, ci_high


# ==============================
#  Splits compartidos por seed
# ==============================

def prepare_splits(
    df_trainval: pd.DataFrame,
    df_test: pd.DataFrame,
    text_field: str,
    seed: int,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Aplica filtros, crea mapping de etiquetas y devuelve splits consistentes
    para un seed dado. Se usa tanto para modelos lineales como transformers.
    """
    random.seed(seed)
    np.random.seed(seed)

    df_trainval = df_trainval.copy()
    df_test = df_test.copy()

    # Filtrar textos vacíos por si acaso
    df_trainval = df_trainval[df_trainval[text_field].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    df_test = df_test[df_test[text_field].apply(lambda x: isinstance(x, str) and x.strip() != "")]

    # Normalizar clase y texto limpio
    for df in (df_trainval, df_test):
        df["phc"] = df["predicted_hate_category"].astype(str).str.strip().str.lower()
        df["clean"] = df[text_field].apply(clean_text)

    if len(df_trainval) == 0 or len(df_test) == 0:
        raise RuntimeError("Train/Val o Test vacío tras filtrar 'nan' y texto.")

    # Etiquetado consistente entre trainval y test
    all_labels = sorted(set(df_trainval["phc"]) | set(df_test["phc"]))
    label2id = {c: i for i, c in enumerate(all_labels)}
    id2label = {i: c for c, i in label2id.items()}

    for df in (df_trainval, df_test):
        df["label"] = df["phc"].map(label2id).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        df_trainval["clean"].values,
        df_trainval["label"].values,
        test_size=val_size,
        random_state=seed,
        stratify=df_trainval["label"].values,
    )
    X_test = df_test["clean"].values
    y_test = df_test["label"].values

    return X_train, X_val, X_test, y_train, y_val, y_test, label2id, id2label


# ==============================
#  Modelos lineales
# ==============================

def single_seed_linear_experiment(
    df_trainval: pd.DataFrame,
    df_test: pd.DataFrame,
    text_field: str,
    seed: int,
    val_size: float,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_df: float,
    max_features: int = None,
) -> pd.DataFrame:

    X_train, X_val, X_test, y_train, y_val, y_test, label2id, id2label = prepare_splits(
        df_trainval,
        df_test,
        text_field=text_field,
        seed=seed,
        val_size=val_size,
    )

    # Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        sublinear_tf=True,
        stop_words="english",
        max_features=max_features,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)

    # Pesos de clase
    uniq = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=uniq, y=y_train)
    cw_map = {int(c): float(w) for c, w in zip(uniq, class_weights)}
    sw_train = np.vectorize(cw_map.get)(y_train)

    models = {
        "logreg_ovr": LogisticRegression(
            max_iter=8000,
            n_jobs=-1,
            class_weight=cw_map,
            C=2.0,
            solver="liblinear",
            multi_class="ovr",
            random_state=seed,
        ),
        "linear_svm": LinearSVC(class_weight=cw_map, random_state=seed),
        "sgd_log": SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            random_state=seed,
        ),
        "passive_aggressive": PassiveAggressiveClassifier(
            C=0.5,
            max_iter=2000,
            random_state=seed,
        ),
        "complement_nb": ComplementNB(alpha=0.5),
    }

    seed_results = []

    for name, clf in models.items():
        print(f"\n=== [LINEAR] Seed {seed} – Modelo: {name} ===")

        # Entrenamiento
        try:
            clf.fit(Xtr, y_train, sample_weight=sw_train)
        except TypeError:
            clf.fit(Xtr, y_train)

        # Evaluación
        val_metrics = eval_split(clf, Xva, y_val, split_name="VAL")
        test_metrics = eval_split(clf, Xte, y_test, split_name="TEST (fijo)")

        seed_results.append(
            {
                "model": name,
                "seed": seed,
                "val_f1_macro": val_metrics["f1_macro"],
                "val_accuracy": val_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

    return pd.DataFrame(seed_results)


# ==============================
#  Transformers (DistilBERT / XLM-R)
# ==============================

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if TRANSFORMERS_AVAILABLE:
        hf_set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def single_seed_transformer_experiment(
    df_trainval: pd.DataFrame,
    df_test: pd.DataFrame,
    text_field: str,
    seed: int,
    val_size: float,
    model_name: str,
    hf_model_id: str,
    max_length: int = 128,
    num_train_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> pd.DataFrame:
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers/torch no están disponibles. Omite modelos:", model_name)
        return pd.DataFrame([])

    set_global_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test, label2id, id2label = prepare_splits(
        df_trainval,
        df_test,
        text_field=text_field,
        seed=seed,
        val_size=val_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    def tokenize_texts(texts):
        return tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    train_enc = tokenize_texts(X_train)
    val_enc = tokenize_texts(X_val)
    test_enc = tokenize_texts(X_test)

    train_dataset = TextDataset(train_enc, y_train)
    val_dataset = TextDataset(val_enc, y_val)
    test_dataset = TextDataset(test_enc, y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1m = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        return {"f1_macro": f1m, "accuracy": acc}

    training_args = TrainingArguments(
        output_dir=f"./tmp_{model_name}_seed{seed}",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.0,
        logging_steps=50,
        save_strategy="no",          # válido en todas las versiones recientes
        seed=seed,
        report_to=[],                # evita que intente usar wandb/tensorboard
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n=== [TRANSFORMER] Seed {seed} – Modelo: {model_name} ({hf_model_id}) ===")
    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    # val_metrics keys: 'eval_loss', 'eval_f1_macro', 'eval_accuracy', ...
    # test_metrics keys: 'test_loss', 'test_f1_macro', 'test_accuracy', ...
    results = pd.DataFrame(
        [
            {
                "model": model_name,
                "seed": seed,
                "val_f1_macro": float(val_metrics.get("eval_f1_macro", np.nan)),
                "val_accuracy": float(val_metrics.get("eval_accuracy", np.nan)),
                "test_f1_macro": float(test_metrics.get("test_f1_macro", np.nan)),
                "test_accuracy": float(test_metrics.get("test_accuracy", np.nan)),
            }
        ]
    )

    return results


# ==============================
#  Agregado multi-seed
# ==============================

def aggregate_runs(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df_all.groupby("model"):
        mean_val_f1, std_val_f1, lo_val_f1, hi_val_f1 = ci95(sub["val_f1_macro"])
        mean_te_f1, std_te_f1, lo_te_f1, hi_te_f1 = ci95(sub["test_f1_macro"])
        mean_val_acc, std_val_acc, lo_val_acc, hi_val_acc = ci95(sub["val_accuracy"])
        mean_te_acc, std_te_acc, lo_te_acc, hi_te_acc = ci95(sub["test_accuracy"])

        rows.append(
            {
                "model": model,
                "val_f1_macro_mean": mean_val_f1,
                "val_f1_macro_std": std_val_f1,
                "val_f1_macro_CI_low": lo_val_f1,
                "val_f1_macro_CI_high": hi_val_f1,
                "test_f1_macro_mean": mean_te_f1,
                "test_f1_macro_std": std_te_f1,
                "test_f1_macro_CI_low": lo_te_f1,
                "test_f1_macro_CI_high": hi_te_f1,
                "val_accuracy_mean": mean_val_acc,
                "val_accuracy_std": std_val_acc,
                "val_accuracy_CI_low": lo_val_acc,
                "val_accuracy_CI_high": hi_val_acc,
                "test_accuracy_mean": mean_te_acc,
                "test_accuracy_std": std_te_acc,
                "test_accuracy_CI_low": lo_te_acc,
                "test_accuracy_CI_high": hi_te_acc,
            }
        )

    df_summary = pd.DataFrame(rows).sort_values(by="val_f1_macro_mean", ascending=False)
    return df_summary


# ==============================
#  MAIN
# ==============================

def main():
    parser = argparse.ArgumentParser(description="Run linear + transformer experiments on SuperTweetEval-style JSONL.")
    parser.add_argument("--input_json", type=str, required=True, help="Ruta al JSONL de train/val (original o augmentado).")
    parser.add_argument("--test_json", type=str, required=True, help="Ruta al JSONL de test (solo originales).")
    parser.add_argument("--text_field", type=str, default="text", help="Nombre de la columna de texto (por defecto 'text').")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13, 17, 19], help="Seeds a usar.")
    parser.add_argument("--val_size", type=float, default=0.10, help="Proporción para validación.")
    parser.add_argument("--output_dir", type=str, default="final_out", help="Directorio donde guardar los CSV.")
    parser.add_argument("--no_drop_hate_age", action="store_true", help="No eliminar la clase 'hate_age'.")
    parser.add_argument("--no_collapse_race", action="store_true", help="No colapsar 'hate_race' en 'hate_origin'.")
    parser.add_argument("--no_transformers", action="store_true", help="No ejecutar DistilBERT/XLM-R, solo modelos lineales.")
    parser.add_argument("--distilbert_model", type=str, default="distilbert-base-uncased", help="ID de modelo HF para DistilBERT.")
    parser.add_argument("--xlmr_model", type=str, default="xlm-roberta-base", help="ID de modelo HF para XLM-R.")
    parser.add_argument("--transformer_max_length", type=int, default=128, help="Longitud máxima de secuencia para transformers.")
    parser.add_argument("--transformer_epochs", type=int, default=3, help="Número de épocas para transformers.")
    parser.add_argument("--transformer_batch_size", type=int, default=16, help="Batch size para transformers.")
    parser.add_argument("--transformer_lr", type=float, default=2e-5, help="Learning rate para transformers.")

    args = parser.parse_args()

    input_json = args.input_json
    test_json = args.test_json

    os.makedirs(args.output_dir, exist_ok=True)

    print("Cargando train/val desde:", input_json)
    print("Cargando test desde:", test_json)

    df_trainval = load_and_clean_tweethate(input_json, text_field=args.text_field)
    df_test = load_and_clean_tweethate(test_json, text_field=args.text_field)

    if not args.no_drop_hate_age:
        print("Eliminando clase 'hate_age' de train y test...")
        df_trainval = remove_values_from_column(df_trainval, "predicted_hate_category", ["hate_age"])
        df_test = remove_values_from_column(df_test, "predicted_hate_category", ["hate_age"])

    if not args.no_collapse_race:
        print("Colapsando 'hate_race' -> 'hate_origin'...")
        df_trainval = collapse_race_into_origin(df_trainval, col="predicted_hate_category")
        df_test = collapse_race_into_origin(df_test, col="predicted_hate_category")

    print("Distribución de clases (train):")
    print(df_trainval["phc"].value_counts())
    print("\nDistribución de clases (test):")
    print(df_test["phc"].value_counts())

    all_runs = []

    print("=====================================")
    print(" Ejecutando modelos lineales")
    print("=====================================")

    for sd in args.seeds:
        print(f"\n##### SEED = {sd} #####\n")
        df_seed = single_seed_linear_experiment(
            df_trainval=df_trainval.copy(),
            df_test=df_test.copy(),
            text_field=args.text_field,
            seed=sd,
            val_size=args.val_size,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            max_features=None,
        )
        all_runs.append(df_seed)

    if not args.no_transformers:
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers/torch NO están instalados; se omiten los modelos grandes.")
        else:
            print("=====================================")
            print(" Ejecutando modelos Transformers")
            print("=====================================")

            for sd in args.seeds:
                # DistilBERT
                df_distil = single_seed_transformer_experiment(
                    df_trainval=df_trainval.copy(),
                    df_test=df_test.copy(),
                    text_field=args.text_field,
                    seed=sd,
                    val_size=args.val_size,
                    model_name="distilbert",
                    hf_model_id=args.distilbert_model,
                    max_length=args.transformer_max_length,
                    num_train_epochs=args.transformer_epochs,
                    batch_size=args.transformer_batch_size,
                    learning_rate=args.transformer_lr,
                )
                if not df_distil.empty:
                    all_runs.append(df_distil)

                # XLM-R
                df_xlmr = single_seed_transformer_experiment(
                    df_trainval=df_trainval.copy(),
                    df_test=df_test.copy(),
                    text_field=args.text_field,
                    seed=sd,
                    val_size=args.val_size,
                    model_name="xlm_roberta",
                    hf_model_id=args.xlmr_model,
                    max_length=args.transformer_max_length,
                    num_train_epochs=args.transformer_epochs,
                    batch_size=args.transformer_batch_size,
                    learning_rate=args.transformer_lr,
                )
                if not df_xlmr.empty:
                    all_runs.append(df_xlmr)

    if not all_runs:
        print("No se han ejecutado modelos. ¿Quizá desactivaste todo por flags?")
        return

    df_all = pd.concat(all_runs, ignore_index=True)
    df_summary = aggregate_runs(df_all)

    train_name = Path(input_json).stem
    test_name = Path(test_json).stem

    runs_path = Path(args.output_dir) / f"runs_{train_name}__{test_name}.csv"
    summary_path = Path(args.output_dir) / f"summary_{train_name}__{test_name}.csv"

    df_all.to_csv(runs_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print("\n=====================================")
    print("Resumen Multi-Seed (95% CI)")
    print("=====================================")
    print(df_summary.to_string(index=False))

    print(f"\nGuardado detalle de runs en:   {runs_path}")
    print(f"Guardado resumen de modelos en: {summary_path}")


if __name__ == "__main__":
    main()
