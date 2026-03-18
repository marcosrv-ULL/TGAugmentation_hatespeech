#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Back-translation augmentation script for hate-speech dataset.

- Reads a JSONL or CSV file with a text column (e.g. "text").
- Optionally filters by language (e.g. lang == "en").
- Uses Helsinki-NLP MT models to do chained back-translation:
    src_lang -> pivot_lang -> src_lang, repeated N cycles.
- Appends back-translated examples to the original dataset.
- Writes an augmented JSONL or CSV.

Requirements:
    pip install transformers torch pandas

Example:
    python backtranslate_dataset.py \
        --input_path merged_masked_unionv2_new/merged_masked_unionv2_new_train.jsonl \
        --output_path merged_masked_unionv2_new/merged_masked_unionv2_new_train_bt.jsonl \
        --text_field text \
        --lang_field lang \
        --lang_filter en \
        --batch_size 16 \
        --bt_cycles 3
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # <-- keep this very early

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==============================
#  IO helpers
# ==============================

def load_any(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported extension for {path}. Use .jsonl or .csv")


def save_any(df: pd.DataFrame, path: str):
    path = str(path)
    if path.endswith(".jsonl"):
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported extension for {path}. Use .jsonl or .csv")


# ==============================
#  Back-translation class
# ==============================

class BackTranslator:
    """
    Simple back-translation:
        src_lang -> pivot_lang -> src_lang
    using Helsinki-NLP models.
    """

    def __init__(
        self,
        src_lang: str = "en",
        pivot_lang: str = "es",
        device: str = None,
        src_pivot_model_name: str = None,
        pivot_src_model_name: str = None,
        max_length: int = 128,
    ):
        self.src_lang = src_lang
        self.pivot_lang = pivot_lang
        self.max_length = max_length

        # Model IDs (can be overridden by args if you want different pivots)
        if src_pivot_model_name is None:
            src_pivot_model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}"
        if pivot_src_model_name is None:
            pivot_src_model_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading MT model: {src_pivot_model_name}")
        self.tok_src_pivot = AutoTokenizer.from_pretrained(src_pivot_model_name)
        self.mod_src_pivot = AutoModelForSeq2SeqLM.from_pretrained(src_pivot_model_name).to(self.device)

        print(f"Loading MT model: {pivot_src_model_name}")
        self.tok_pivot_src = AutoTokenizer.from_pretrained(pivot_src_model_name)
        self.mod_pivot_src = AutoModelForSeq2SeqLM.from_pretrained(pivot_src_model_name).to(self.device)

    @torch.no_grad()
    def _translate_batch(self, texts, tokenizer, model):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        outputs = model.generate(
            **inputs,
            max_length=self.max_length,
            num_beams=4,
        )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def backtranslate_batch(self, texts):
        """
        Single back-translation hop:
            src_lang -> pivot_lang -> src_lang
        """
        # src -> pivot
        pivot_texts = self._translate_batch(texts, self.tok_src_pivot, self.mod_src_pivot)
        # pivot -> src
        bt_texts = self._translate_batch(pivot_texts, self.tok_pivot_src, self.mod_pivot_src)
        return bt_texts


# ==============================
#  Main augmentation logic
# ==============================

def augment_with_backtranslation(
    df: pd.DataFrame,
    text_field: str = "text",
    lang_field: str = None,
    lang_filter: str = None,
    batch_size: int = 16,
    max_examples: int = None,
    src_lang: str = "en",
    pivot_lang: str = "es",
    max_length: int = 128,
    bt_cycles: int = 1,
) -> pd.DataFrame:
    """
    Returns a new DataFrame containing:
        - all original rows
        - + N backtranslated rows per selected original row, where N = bt_cycles

    For each original row, we do:
        cycle 1: original -> BT_1
        cycle 2: BT_1     -> BT_2
        ...
        cycle N: BT_{N-1} -> BT_N
    """

    df = df.copy()

    if bt_cycles < 1:
        print(f"[WARNING] bt_cycles={bt_cycles} is < 1, forcing it to 1.")
        bt_cycles = 1

    # Optional language filter
    if lang_field is not None and lang_filter is not None and lang_field in df.columns:
        before = len(df)
        df = df[df[lang_field].astype(str).str.lower() == lang_filter.lower()].copy()
        print(f"Filtered by {lang_field} == '{lang_filter}': {before} -> {len(df)} rows")

    # Drop rows with empty or non-string text
    df = df[df[text_field].apply(lambda x: isinstance(x, str) and x.strip() != "")].copy()

    # Subsample if requested (e.g. for quick tests)
    if max_examples is not None and max_examples > 0:
        df = df.iloc[:max_examples].copy()
        print(f"Using only first {max_examples} examples for BT.")

    # Prepare back-translator
    bt = BackTranslator(
        src_lang=src_lang,
        pivot_lang=pivot_lang,
        max_length=max_length,
    )

    bt_rows = []

    # We will iterate row indices in batches
    indices = df.index.to_list()
    for start in tqdm(range(0, len(indices), batch_size), desc="Back-translating"):
        end = start + batch_size
        batch_idx = indices[start:end]
        batch_texts = df.loc[batch_idx, text_field].tolist()

        # current_texts holds the texts for this cycle
        current_texts = batch_texts

        for cycle in range(1, bt_cycles + 1):
            try:
                current_texts = bt.backtranslate_batch(current_texts)
            except Exception as e:
                print(f"\n[WARNING] Error during BT (cycle {cycle}) for batch {start}:{end} -> {e}")
                break

            # Create one augmented row per item for this cycle
            for idx, bt_text in zip(batch_idx, current_texts):
                orig_row = df.loc[idx]
                new_row = orig_row.copy()

                # new text
                new_row[text_field] = bt_text

                # track provenance
                if "id" in orig_row:
                    new_row["bt_from_id"] = orig_row["id"]
                    new_row["id"] = f"{orig_row['id']}_bt{cycle}"
                else:
                    new_row["bt_from_id"] = idx
                    new_row["id"] = f"{idx}_bt{cycle}"

                new_row["aug_source"] = f"backtranslation_c{cycle}"
                new_row["bt_cycle"] = cycle
                bt_rows.append(new_row)

    df_bt = pd.DataFrame(bt_rows)

    # Mark originals too
    df_orig = df.copy()
    if "aug_source" not in df_orig.columns:
        df_orig["aug_source"] = "original"
    else:
        df_orig["aug_source"] = df_orig["aug_source"].fillna("original")

    print(f"Original rows: {len(df_orig)} | BT rows: {len(df_bt)}")

    # Concatenate originals + BT (if any BT rows were created)
    if len(df_bt) > 0:
        df_aug = pd.concat([df_orig, df_bt], ignore_index=True)
    else:
        df_aug = df_orig

    print(f"Total rows in augmented dataset: {len(df_aug)}")
    return df_aug


# ==============================
#  CLI
# ==============================

def parse_args():
    ap = argparse.ArgumentParser(description="Back-translation augmentation for hate-speech dataset.")
    ap.add_argument("--input_path", type=str, required=True, help="Input JSONL/CSV with a 'text' column.")
    ap.add_argument("--output_path", type=str, required=True, help="Output JSONL/CSV with originals + BT.")
    ap.add_argument("--text_field", type=str, default="text", help="Name of the text column to back-translate.")
    ap.add_argument("--lang_field", type=str, default=None, help="Optional language column (e.g. 'lang').")
    ap.add_argument("--lang_filter", type=str, default=None, help="Keep only rows where lang_field == this value (e.g. 'en').")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for MT models.")
    ap.add_argument("--max_examples", type=int, default=None, help="Optional limit of examples to BT (for debugging).")
    ap.add_argument("--src_lang", type=str, default="en", help="Source language code (Helsinki-NLP style).")
    ap.add_argument("--pivot_lang", type=str, default="es", help="Pivot language code (Helsinki-NLP style).")
    ap.add_argument("--max_length", type=int, default=500, help="Max sequence length for MT models.")
    ap.add_argument("--bt_cycles", type=int, default=1, help="Number of back-translation cycles per example.")
    return ap.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    print(f"Loading data from: {input_path}")
    df = load_any(input_path)

    df_aug = augment_with_backtranslation(
        df=df,
        text_field=args.text_field,
        lang_field=args.lang_field,
        lang_filter=args.lang_filter,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        src_lang=args.src_lang,
        pivot_lang=args.pivot_lang,
        max_length=args.max_length,
        bt_cycles=args.bt_cycles,
    )

    print(f"Saving augmented data to: {output_path}")
    save_any(df_aug, output_path)


if __name__ == "__main__":
    main()
