#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apply a hate-speech lexicon to a dataset and create a `text_masked` column.

- Input:  .csv or .jsonl file with a text column (default: "text")
- Lexicon: JSONL file, same structure as in the notebook:
    {
        "TARGET": {
            "GENDER_WOMEN": ["women", "woman", ...],
            "GENDER_MEN":   ["men", "man", ...],
            ...
        },
        "SLUR": {
            "RACE_BLACK": ["..."],
            ...
        }
    }

- Output: same format as input, with an extra column `text_masked`.
  By default written to: <stem>_masked.<ext>
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd


# ==============================
#  Basic JSONL loader
# ==============================

def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ==============================
#  Lexicon helpers
# ==============================

def load_lexicon_from_jsonl(path: str = "lexicon/lexicon.jsonl") -> Dict[str, Any]:
    """
    Read a JSONL lexicon file and return a Python dict.

    Expected formats (same as in the notebook):
    - A column named 'lexicon' with a dict in the first row.
    - A single column whose first row is a dict.
    - Fallback: treat the first row as a dict via row.to_dict().
    """
    df_lex = load_jsonl(path)
    if len(df_lex) == 0:
        raise RuntimeError(f"No rows found in {path}")

    row0 = df_lex.iloc[0]

    # Case 1: explicit 'lexicon' column
    if "lexicon" in df_lex.columns and isinstance(row0["lexicon"], dict):
        return row0["lexicon"]

    # Case 2: single column whose value is the dict
    if len(df_lex.columns) == 1:
        col = df_lex.columns[0]
        if isinstance(row0[col], dict):
            return row0[col]

    # Fallback: treat full row as dict
    maybe_dict = row0.to_dict()
    if isinstance(maybe_dict, dict):
        return maybe_dict

    raise RuntimeError("Unrecognized lexicon.jsonl format.")


def build_lexicon_entries(lexicon, placeholder_keys=("TARGET", "SLUR")):
    """
    Flatten the lexicon into a list of entries:

        {
            "phrase": original phrase (str),
            "pattern": compiled regex,
            "placeholder": "[TARGET:GENDER_WOMEN]",
            "group": "GENDER_WOMEN",
            "kind": "TARGET" or "SLUR",
        }

    - Ignores very short phrases (len < 2) to avoid noise.
    - For “normal word” phrases (letters/digits/spaces/'/-), we add word
      boundaries so we match full words, not substrings.
    """
    entries = []

    for kind in placeholder_keys:  # "TARGET", "SLUR"
        if kind not in lexicon:
            continue

        by_axis = lexicon[kind]   # e.g. gender / ethnicity / ...
        if not isinstance(by_axis, dict):
            continue

        for axis, by_group in by_axis.items():
            if not isinstance(by_group, dict):
                continue

            for group_code, phrase_dict in by_group.items():
                if not isinstance(phrase_dict, dict):
                    continue

                for phrase in phrase_dict.keys():
                    if not phrase:
                        continue

                    phrase_stripped = str(phrase).strip()
                    if len(phrase_stripped) < 2:
                        # skip ultra-short stuff: "u", etc.
                        continue

                    # "Normal word" → use word boundaries
                    if re.fullmatch(r"[0-9A-Za-z' \-]+", phrase_stripped):
                        pattern = re.compile(
                            r"\b" + re.escape(phrase_stripped) + r"\b",
                            flags=re.IGNORECASE,
                        )
                    else:
                        # Emoji / punctuation / weird chars → literal match
                        pattern = re.compile(
                            re.escape(phrase_stripped),
                            flags=re.IGNORECASE,
                        )

                    entries.append({
                        "phrase": phrase_stripped,
                        "pattern": pattern,
                        "placeholder": f"[{kind}:{group_code}]",
                        "group": group_code,
                        "kind": kind,
                    })

    # Longer phrases first (e.g. "black women" before "black")
    entries.sort(key=lambda e: len(e["phrase"]), reverse=True)
    return entries


# ==============================
#  Text cleaning helpers (NEW)
# ==============================

MENTION_PATTERN = re.compile(r"@user\b", flags=re.IGNORECASE)
CONTROL_CHARS_RE = re.compile(r"[\u0000-\u001F\u007F-\u009F]")


def basic_clean_text(text: Any) -> str:
    """
    Basic cleaning for the text column:
    - Convert non-strings to string (or empty).
    - Replace '@user' (case-insensitive) with '[MENTION]'.
    - Remove control / non-printable characters.
    - Collapse multiple spaces and strip.
    """
    if not isinstance(text, str):
        if pd.isna(text):
            text = ""
        else:
            text = str(text)

    # Replace @user with [MENTION]
    text = MENTION_PATTERN.sub("[MENTION]", text)

    # Remove control chars
    text = CONTROL_CHARS_RE.sub("", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def mask_text_with_lexicon(text: str,
                           entries: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Apply lexicon entries to a single text:

    - Avoids overlapping matches using a character-level occupancy array.
    - Replaces spans with placeholders like [TARGET:...], [SLUR:...].
    - Returns (masked_text, sorted_groups).

    If nothing is masked, returns (original_text, []).
    """
    if not isinstance(text, str) or not text:
        return text, []

    n = len(text)
    occupied = [False] * n
    spans: List[Tuple[int, int, str, str]] = []  # (start, end, placeholder, group)
    groups = set()

    for e in entries:
        pattern = e["pattern"]
        placeholder = e["placeholder"]
        group = e["group"]

        for m in pattern.finditer(text):
            start, end = m.span()
            if start < 0 or end > n:
                continue

            # Skip if any character in this span is already taken
            if any(occupied[i] for i in range(start, end)):
                continue

            # Mark span as occupied
            for i in range(start, end):
                occupied[i] = True

            spans.append((start, end, placeholder, group))
            groups.add(group)

    if not spans:
        return text, []

    # Sort spans by start index
    spans.sort(key=lambda x: x[0])

    out = []
    last = 0
    for start, end, placeholder, _group in spans:
        out.append(text[last:start])
        out.append(placeholder)
        last = end
    out.append(text[last:])

    masked_text = "".join(out)
    return masked_text, sorted(groups)


# ==============================
#  Data loading & saving
# ==============================

def load_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported input format: {path}. Use .csv or .jsonl")


def save_table(df: pd.DataFrame, path: str):
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".jsonl"):
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {path}. Use .csv or .jsonl")


# ==============================
#  Main CLI
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="Apply lexicon masking to a dataset and create `text_masked`."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .csv or .jsonl",
    )
    parser.add_argument(
        "--lexicon",
        required=True,
        help="Path to lexicon JSONL (same as in the notebook, e.g. lexicon/lexicon.jsonl)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Name of the text column to mask (default: 'text')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. If not set, uses <stem>_masked.<ext>",
    )

    args = parser.parse_args()

    input_path = args.input
    lexicon_path = args.lexicon
    text_field = args.text_field

    # Decide output path
    if args.output is None:
        p = Path(input_path)
        out_path = str(p.with_name(p.stem + "_masked" + p.suffix))
    else:
        out_path = args.output

    print(f"Loading data from: {input_path}")
    df = load_table(input_path)

    if text_field not in df.columns:
        raise KeyError(f"Column '{text_field}' not found in input file.")

    # NEW: clean text column (mentions + weird chars)
    print(f"Cleaning text field '{text_field}' (mentions + control chars)...")
    df[text_field] = df[text_field].apply(basic_clean_text)

    print(f"Loading lexicon from: {lexicon_path}")
    lexicon = load_lexicon_from_jsonl(lexicon_path)
    entries = build_lexicon_entries(lexicon)

    print(f"Lexicon entries built: {len(entries)}")

    # Apply masking row-wise
    print("Applying masking...")
    masked_and_groups = df[text_field].apply(
        lambda t: mask_text_with_lexicon(t, entries)
    )

    df["text_masked"] = masked_and_groups.apply(lambda x: x[0])
    df["mask_groups"] = masked_and_groups.apply(lambda x: ",".join(x[1]) if x[1] else "")

    n_masked = (df["text_masked"] != df[text_field]).sum()
    print(f"Rows with at least one mask applied: {n_masked} / {len(df)}")

    print(f"Saving output to: {out_path}")
    save_table(df, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
