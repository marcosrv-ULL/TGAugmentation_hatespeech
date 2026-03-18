#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd


def sample_low_regime(
    input_path: Path,
    label_col: str = "predicted_hate_category",
    max_per_class: int = 20,
    random_state: int = 42,
):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = input_path.suffix.lower()

    if ext == ".jsonl":
        df = pd.read_json(input_path, lines=True)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (use .jsonl or .csv)")

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in input file.")

    # Sample up to `max_per_class` rows per class
    df_low = (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=random_state))
    )

    # Build output path: <original_name>_low_regime.<extension>
    output_path = input_path.with_name(f"{input_path.stem}_low_regime{ext}")

    if ext == ".jsonl":
        df_low.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:  # .csv
        df_low.to_csv(output_path, index=False)

    print(f"Saved low-regime dataset to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a low-regime subset from a JSONL or CSV file by sampling "
            "up to 20 examples per class from 'predicted_hate_category'."
        )
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input dataset (.jsonl or .csv)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="predicted_hate_category",
        help="Name of the label column (default: predicted_hate_category)",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=20,
        help="Maximum number of examples per class (default: 20)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()

    try:
        sample_low_regime(
            input_path=Path(args.input_path),
            label_col=args.label_col,
            max_per_class=args.max_per_class,
            random_state=args.random_state,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
