#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Masking script for hate-speech style datasets.

- Reads a JSONL or CSV file with at least a text field (default: "text")
  and an optional predicted hate category field (default: "predicted_hate_category").
- Uses an OpenAI chat model to produce a masked version of the text
  where protected-group mentions and slurs are replaced by placeholders.

Placeholders:
  [TARGET:<CODE>] for group mentions
  [SLUR:<CODE>]   for offensive epithets
  [MENTION]       for @handles
  [URL]           for URLs
  [HASHTAG:tag]   for hashtags

It adds:
  - "text_masked"
  - "groups_from_mask"
  - "taxonomy_version"
  - "masking_model"

Usage example:

    python mask_hatespeech_dataset.py \
        --input_path MultilingualTweetEval2024/test1.jsonl \
        --output_path MultilingualTweetEval2024/test1_masked.jsonl \
        --text_field text \
        --cat_field predicted_hate_category \
        --model gpt-4o-mini

    python mask_hatespeech_dataset.py \
        --input_path MultilingualTweetEval2024/test1.csv \
        --output_path MultilingualTweetEval2024/test1_masked.jsonl \
        --text_field text \
        --cat_field predicted_hate_category
"""

import os
import re
import csv
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm
from openai import OpenAI

# ===================== Default config =====================
SEED = 42
random.seed(SEED)

# Retries in case of rate limits / transient errors
RETRIES = 2
RETRY_SLEEP = 1.0

# =================== Taxonomy ====================
TAXONOMY = [
    # Gender Identity
    "GENDER_WOMEN", "GENDER_MEN", "GENDER_NONBINARY",
    "GENDER_IDENTITY_TRANS", "GENDER_IDENTITY_OTHER",
    # Ethnicity / Race
    "RACE_BLACK", "RACE_WHITE", "RACE_OTHER",
    "ETH_ASIAN", "ETH_LATINO", "ETH_ARAB", "ETH_JEWISH",
    "ETH_ROMA", "ETH_OTHER",
    # Sexual Orientation
    "SEXUAL_ORIENTATION_GAY", "SEXUAL_ORIENTATION_LESBIAN",
    "SEXUAL_ORIENTATION_BI", "SEXUAL_ORIENTATION_OTHER",
    # Religion
    "RELIGION_ISLAM", "RELIGION_CHRIST", "RELIGION_JUDAISM",
    "RELIGION_HINDU", "RELIGION_SIKH", "RELIGION_BUDDHIST",
    "RELIGION_ATHEIST", "RELIGION_OTHER",
    # National Origin
    "NATIONAL_ORIGIN_IMMIGRANTS", "NATIONAL_ORIGIN_REFUGEES",
    "NATIONAL_ORIGIN_OTHER",
    # Disability Status
    "DISABILITY_PHYSICAL", "DISABILITY_MENTAL",
    "DISABILITY_DEVELOPMENTAL", "DISABILITY_OTHER",
    # Age
    "AGE_YOUTH", "AGE_MIDDLE", "AGE_ELDERLY", "AGE_OTHER",
]
TAXO_STR = ", ".join(TAXONOMY)

# =================== Regex helpers ====================
MENTION_RE = re.compile(r"@\w+")
URL_RE     = re.compile(r"https?://\S+|\{URL\}")
HASHTAG_RE = re.compile(r"#(\w+)")
PLACEHOLDER_RE = re.compile(
    r"\[(?:TARGET|SLUR):[A-Z0-9_]+\]|\[MENTION\]|\[URL\]|\[HASHTAG:[A-Za-z0-9_]+\]"
)
CODE_RE = re.compile(r"\[(?:TARGET|SLUR):([A-Z0-9_]+)\]")

# =================== Prompts ====================

MASK_PROMPT = """You are a safety redactor. Given the TEXT, pick appropriate GROUP CODES ONLY from this closed taxonomy (do not invent labels):
[{taxo}]
{hint_block}
Then produce a single masked string where:
- Replace any explicit mentions or epithets against those groups with placeholders:
  [TARGET:<CODE>] for group mentions, [SLUR:<CODE>] for offensive epithets.
- Replace @handles → [MENTION], URLs → [URL], and hashtags #tag → [HASHTAG:tag].
- Keep everything else as-is, and NEVER output explicit slurs.
Return ONLY the masked text string. No JSON, no explanations.

TEXT:
\"\"\"{text}\"\"\"
"""

def build_mask_messages(text: str, pred_label: str | None = None) -> list:
    system_msg = (
        "You are a careful safety redactor. "
        "You must output only the masked text, no explanations."
    )

    hint_block = ""
    if pred_label:
        hint_block = (
            "You are also given a weak classifier hint: "
            f'the predicted hate category for this text is "{pred_label}". '
            "If this clearly corresponds to one or more TAXONOMY CODES, "
            "prefer those codes when masking; otherwise ignore this hint."
        )

    user_msg = MASK_PROMPT.format(
        taxo=TAXO_STR,
        text=text,
        hint_block=hint_block,
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

# =================== I/O utilities ====================

def read_api_key_from_file(path: str = "API_KEY") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def ensure_api_key(api_key_file: str = "API_KEY"):
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = read_api_key_from_file(api_key_file)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def read_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows

def read_rows_auto(path: str) -> List[Dict[str, Any]]:
    ext = Path(path).suffix.lower()
    if ext in {".jsonl", ".json"}:
        print(f"Detected JSONL/JSON input: {path}")
        return read_jsonl(path)
    elif ext == ".csv":
        print(f"Detected CSV input: {path}")
        return read_csv(path)
    else:
        raise ValueError(f"Unsupported input extension for {path}. Use .jsonl/.json or .csv")

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    out_dir = os.path.dirname(path)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def is_valid_row(row: Dict[str, Any], text_field: str, cat_field: str) -> bool:
    text = str(row.get(text_field, "")).strip()
    if not text:
        return False
    cat = row.get(cat_field, None)
    # Allow neutral / other labels; only filter out pure NaNs/empties
    if cat is None:
        return False
    cat_str = str(cat).strip().lower()
    return cat_str != "nan"

def normalize_label(label: str) -> str:
    x = re.sub(r"[^A-Za-z0-9/_-]+", "_", str(label)).upper()
    x = x.replace("/", "_")
    return x or "UNKNOWN"

# =================== OpenAI client ====================

def get_client(api_key_file: str = "API_KEY") -> OpenAI:
    ensure_api_key(api_key_file)
    return OpenAI()

def chat_text(
    client: OpenAI,
    messages: list,
    model: str,
    max_tokens: int = 700,
    temperature: float = 0.1,
    seed: int = SEED,
) -> str:
    """Chat Completions with textual output only."""
    for attempt in range(RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt >= RETRIES:
                raise
            time.sleep(RETRY_SLEEP * (attempt + 1))
    return ""

# =================== Main pipeline ====================

def mask_dataset(
    input_path: str,
    output_path: str,
    text_field: str = "text",
    cat_field: str = "predicted_hate_category",
    model: str = None,
    api_key_file: str = "API_KEY",
    max_tokens: int = 900,
    temperature: float = 0.1,
):
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    client = get_client(api_key_file)
    rows = read_rows_auto(input_path)

    # Filter invalid rows
    valid_rows = [r for r in rows if is_valid_row(r, text_field, cat_field)]
    print(f"Valid rows: {len(valid_rows)} (from {len(rows)} total)")

    out_rows: List[Dict[str, Any]] = []

    for r in tqdm(valid_rows, desc="Masking with OpenAI"):
        original_text = str(r.get(text_field, ""))
        pred_cat_raw = r.get(cat_field, "")
        pred_cat = str(pred_cat_raw).strip()

        # 1) Mask with the LLM (output must be plain text), passing category hint
        pred_cat_hint = pred_cat if pred_cat and pred_cat.lower() not in ("nan", "") else None

        masked = chat_text(
            client=client,
            messages=build_mask_messages(original_text, pred_label=pred_cat_hint),
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=SEED,
        )

        # 2) Defensive replacements in case the LLM missed them
        masked = MENTION_RE.sub("[MENTION]", masked)
        masked = URL_RE.sub("[URL]", masked)
        masked = HASHTAG_RE.sub(lambda m: f"[HASHTAG:{m.group(1)}]", masked)

        # 3) If no placeholders but row is not neutral, inject [TARGET:<LABEL>]
        if not PLACEHOLDER_RE.search(masked):
            if pred_cat and pred_cat.lower() not in ("neutral", "nan", ""):
                masked = f"[TARGET:{normalize_label(pred_cat)}] {masked}"

        # 4) Extract groups from placeholders
        groups = sorted(set(CODE_RE.findall(masked)))

        # 5) Output: keep everything original + added fields
        out_row = dict(r)
        out_row.update(
            {
                "text_masked": masked,
                "groups_from_mask": groups,
                "taxonomy_version": "v1.1-closed",
                "masking_model": model,
            }
        )
        out_rows.append(out_row)

    write_jsonl(output_path, out_rows)
    print(f"Masked dataset saved to: {output_path}")
    print(f"Masked rows: {len(out_rows)}")

# =================== CLI ====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask protected-group mentions and slurs in a hate-speech dataset."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input JSONL/JSON or CSV file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSONL file (masked).",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Name of the text field (default: text).",
    )
    parser.add_argument(
        "--cat_field",
        type=str,
        default="predicted_hate_category",
        help="Name of the hate-category field (default: predicted_hate_category).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI chat model name (default: env OPENAI_MODEL or gpt-4o-mini).",
    )
    parser.add_argument(
        "--api_key_file",
        type=str,
        default="API_KEY",
        help="File containing your OpenAI API key (default: API_KEY).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=900,
        help="Maximum tokens for the LLM response (default: 900).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for the LLM (default: 0.1).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    mask_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        text_field=args.text_field,
        cat_field=args.cat_field,
        model=args.model,
        api_key_file=args.api_key_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

if __name__ == "__main__":
    main()
