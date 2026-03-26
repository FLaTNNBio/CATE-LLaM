import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def detect_column_role(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) <= 2:
            return "binary"
        return "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    else:
        return "categorical"


def numeric_summary(series: pd.Series, quantiles):
    s = series.dropna()

    if len(s) == 0:
        return {}

    q_values = s.quantile(quantiles).to_dict()

    stats = {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(s.median()),
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurt())
    }

    for q, val in q_values.items():
        stats[f"p{int(q*100):02d}"] = float(val)

    return stats


def categorical_summary(series: pd.Series, top_k):
    s = series.dropna()
    value_counts = s.value_counts()

    top_values = []
    total = len(s)

    for val, count in value_counts.head(top_k).items():
        top_values.append({
            "value": str(val),
            "count": int(count),
            "pct": float(count / total)
        })

    return top_values


def datetime_summary(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return {}

    return {
        "min": str(s.min()),
        "max": str(s.max())
    }


def build_dataset_card(df, file_path, quantiles, top_k):
    card = {}

    card["dataset_name"] = os.path.basename(file_path)
    card["generated_at"] = datetime.now().isoformat()
    card["n_rows"] = int(df.shape[0])
    card["n_cols"] = int(df.shape[1])

    card["file"] = {
        "path": file_path,
        "size_bytes": os.path.getsize(file_path)
    }

    card["schema"] = [
        {"name": col, "dtype": str(df[col].dtype)}
        for col in df.columns
    ]

    columns_dict = {}

    for col in df.columns:
        series = df[col]
        role = detect_column_role(series)

        col_info = {
            "dtype": str(series.dtype),
            "role": role,
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique": int(series.nunique(dropna=True))
        }

        if role == "numeric" or role == "binary":
            col_info["stats"] = numeric_summary(series, quantiles)

        elif role == "categorical":
            col_info["top_values"] = categorical_summary(series, top_k)

        elif role == "datetime":
            col_info["stats"] = datetime_summary(series)

        columns_dict[col] = col_info

    card["columns"] = columns_dict

    return card


def main():
    parser = argparse.ArgumentParser(description="Generate dataset JSON card from parquet file.")
    parser.add_argument("--input", required=True, help="Path to parquet file")
    parser.add_argument("--output", default="dataset_card.json", help="Output JSON file")
    parser.add_argument("--quantiles", nargs="+", type=float,
                        default=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                        help="Quantiles to compute")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top K categories for categorical variables")

    args = parser.parse_args()

    print("Reading parquet file...")
    df = pd.read_parquet(args.input)

    print("Building dataset card...")
    card = build_dataset_card(df, args.input, args.quantiles, args.top_k)

    print("Saving JSON...")
    with open(args.output, "w") as f:
        json.dump(card, f, indent=2)

    print(f"Dataset card saved to {args.output}")


if __name__ == "__main__":
    main()