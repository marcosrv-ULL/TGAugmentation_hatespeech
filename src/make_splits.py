import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_file(path: str) -> pd.DataFrame:
    """Load CSV or JSONL into a DataFrame."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    else:
        raise ValueError("Unsupported file format. Use .csv or .jsonl")


def save_split(df: pd.DataFrame, out_dir: Path, name: str, split: str, fmt: str):
    """Save dataframe in the correct format."""
    outfile = out_dir / f"{name}_{split}.{fmt}"

    if fmt == "csv":
        df.to_csv(outfile, index=False)
    else:  # jsonl
        df.to_json(outfile, orient="records", lines=True)


def main():
    parser = argparse.ArgumentParser(description="Split a dataset into train/val/test.")
    parser.add_argument("--input", required=True, help="Path to input .csv or .jsonl")

    parser.add_argument("--train", type=float, default=0.8, help="Train percentage")
    parser.add_argument("--val",   type=float, default=0.1, help="Validation percentage")
    parser.add_argument("--test",  type=float, default=0.1, help="Test percentage")
    parser.add_argument("--seed",  type=int,   default=42,  help="Random seed")

    # === NUEVO: columna de clase (Y) y opciones de balanceo ===
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Nombre de la columna de etiqueta (Y). Si se da, se usan splits estratificados."
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="*",
        default=[],
        help="Clases de Y a excluir por completo (espacio separadas)."
    )
    parser.add_argument(
        "--balance-per-class",
        action="store_true",
        help="Si se activa, downsamplea todas las clases de Y al tamaño de la minoritaria antes de hacer los splits."
    )

    args = parser.parse_args()
    input_path = args.input

    # Load
    df = load_file(input_path)

    # Check percentages
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train + val + test ratios must sum to 1.0")

    # Base name and format
    base = Path(input_path).stem
    out_format = input_path.split(".")[-1]  # csv or jsonl
    out_dir = Path(base)
    out_dir.mkdir(exist_ok=True)

    # ============================
    # 1) Filtrado por clases de Y
    # ============================
    if args.label_col is not None:
        if args.label_col not in df.columns:
            raise ValueError(f"label-col '{args.label_col}' no existe en el DataFrame.")

        # eliminar filas con Y NaN
        df = df[df[args.label_col].notna()].copy()

        # excluir clases si se ha pedido
        if args.exclude_classes:
            excl = set(args.exclude_classes)
            before = len(df)
            df = df[~df[args.label_col].isin(excl)].copy()
            after = len(df)
            print(f"Excluidas clases {excl}: {before} → {after} filas")

        # opcional: balancear por clase (downsample)
        if args.balance_per_class:
            counts = df[args.label_col].value_counts()
            min_count = counts.min()
            print("Balanceando por clase (downsample) según label_col:")
            print(counts)
            print(f"Cada clase se recorta a n={min_count}")
            df = (
                df.groupby(args.label_col, group_keys=False)
                  .apply(lambda g: g.sample(n=min_count, random_state=args.seed))
                  .reset_index(drop=True)
            )
            print("Distribución tras balancear:")
            print(df[args.label_col].value_counts())

    # ============================
    # 2) Splits estratificados (si hay label-col)
    # ============================
    stratify_all = None
    if args.label_col is not None:
        n_classes = df[args.label_col].nunique()
        if n_classes >= 2:
            stratify_all = df[args.label_col]
        else:
            print("Solo hay una clase en label-col; no se puede estratificar.")
            stratify_all = None

    # Train + temp split
    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - args.train),
        random_state=args.seed,
        shuffle=True,
        stratify=stratify_all
    )

    # Val + Test from temp
    if args.val == 0.0:
        df_val  = df_temp.iloc[0:0].copy()   # empty dataframe
        df_test = df_temp.copy()

    elif args.test == 0.0:
        df_val  = df_temp.copy()
        df_test = df_temp.iloc[0:0].copy()   # empty dataframe

    else:
        val_ratio = args.val / (args.val + args.test)

        # estratificar también en el split val/test si hay label-col
        stratify_temp = None
        if args.label_col is not None:
            n_classes_temp = df_temp[args.label_col].nunique()
            if n_classes_temp >= 2:
                stratify_temp = df_temp[args.label_col]

        df_val, df_test = train_test_split(
            df_temp,
            test_size=(1 - val_ratio),
            random_state=args.seed,
            shuffle=True,
            stratify=stratify_temp
        )

    # Save files
    save_split(df_train, out_dir, base, "train", out_format)
    save_split(df_val,   out_dir, base, "val",   out_format)
    save_split(df_test,  out_dir, base, "test",  out_format)

    print(f"Saved splits into directory: {out_dir}")

    # Info extra de distribución si tenemos label-col
    if args.label_col is not None:
        print("\nDistribución por clase en TRAIN:")
        print(df_train[args.label_col].value_counts())
        print("\nDistribución por clase en VAL:")
        print(df_val[args.label_col].value_counts())
        print("\nDistribución por clase en TEST:")
        print(df_test[args.label_col].value_counts())


if __name__ == "__main__":
    main()
