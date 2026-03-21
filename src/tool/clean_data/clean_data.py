"""

python clean_data/clean_data.py --reference ../../data/analytic/aids/aids_rct_id.parquet --input ../results/confounding_bias/actg175_observational.parquet --output ../results/final/actg175_observational_final.parquet --output-report ../results/final/actg175_observational_final_report.json --original-treatment-col treat --observational-treatment-col treat_obs

    
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if df.empty:
        raise ValueError(f"Loaded dataframe is empty: {path}")

    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def build_final_dataset(
    reference_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    *,
    original_treatment_col: str,
    observational_treatment_col: str,
    backup_original_treatment: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the final harmonized dataset.

    Rules:
    - keep exactly the columns of the reference/original dataset
    - preserve the original column order
    - replace original_treatment_col with observational_treatment_col values
    - drop all extra technical columns
    """
    reference_columns = list(reference_df.columns)

    if original_treatment_col not in reference_columns:
        raise ValueError(
            f"Original treatment column '{original_treatment_col}' "
            f"is not present in reference dataset."
        )

    if original_treatment_col not in transformed_df.columns:
        raise ValueError(
            f"Original treatment column '{original_treatment_col}' "
            f"is not present in transformed dataset."
        )

    if observational_treatment_col not in transformed_df.columns:
        raise ValueError(
            f"Observational treatment column '{observational_treatment_col}' "
            f"is not present in transformed dataset."
        )

    missing_reference_columns_in_transformed = [
        col for col in reference_columns if col not in transformed_df.columns
    ]
    if missing_reference_columns_in_transformed:
        raise ValueError(
            "The transformed dataset is missing columns from the reference dataset: "
            f"{missing_reference_columns_in_transformed}"
        )

    final_df = transformed_df.copy()

    if backup_original_treatment:
        backup_col = f"{original_treatment_col}_original_backup"
        final_df[backup_col] = final_df[original_treatment_col]

    # Replace the original treatment with the observational one
    final_df[original_treatment_col] = final_df[observational_treatment_col]

    # Keep exactly the original schema and original column order
    final_df = final_df[reference_columns].copy()

    dropped_extra_columns = [
        col for col in transformed_df.columns if col not in reference_columns
    ]

    metadata = {
        "n_rows_reference": int(len(reference_df)),
        "n_rows_transformed": int(len(transformed_df)),
        "n_rows_final": int(len(final_df)),
        "reference_columns": reference_columns,
        "original_treatment_col": original_treatment_col,
        "observational_treatment_col": observational_treatment_col,
        "backup_original_treatment": bool(backup_original_treatment),
        "dropped_extra_columns": dropped_extra_columns,
    }

    return final_df, metadata


def build_report(metadata: dict) -> dict:
    return {
        "finalization": metadata
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a cleaned final observational dataset by restoring the original schema, "
            "replacing the original treatment column with the observational treatment, "
            "and removing all extra technical columns."
        )
    )

    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the original/reference dataset (schema source).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the transformed dataset (after selection/confounding).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the cleaned final dataset (.parquet or .csv).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to a JSON report.",
    )
    parser.add_argument(
        "--original-treatment-col",
        type=str,
        default="treat",
        help="Name of the treatment column in the original/reference schema.",
    )
    parser.add_argument(
        "--observational-treatment-col",
        type=str,
        default="treat_obs",
        help="Name of the observational treatment column in the transformed dataset.",
    )
    parser.add_argument(
        "--backup-original-treatment",
        action="store_true",
        help=(
            "Create a backup of the original treatment column before replacement. "
            "Note: the backup will not be present in the final cleaned dataset, "
            "because the final dataset keeps exactly the original schema."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_df = load_dataframe(args.reference)
    transformed_df = load_dataframe(args.input)

    final_df, metadata = build_final_dataset(
        reference_df=reference_df,
        transformed_df=transformed_df,
        original_treatment_col=args.original_treatment_col,
        observational_treatment_col=args.observational_treatment_col,
        backup_original_treatment=args.backup_original_treatment,
    )

    save_dataframe(final_df, args.output)

    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        report = build_report(metadata)
        with args.output_report.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    print("\n=== Finalization completed ===")
    print(f"Reference rows:   {len(reference_df)}")
    print(f"Transformed rows: {len(transformed_df)}")
    print(f"Final rows:       {len(final_df)}")
    print(f"Columns kept:     {len(final_df.columns)}")
    print(f"Treatment column replaced: {args.original_treatment_col} <- {args.observational_treatment_col}")


if __name__ == "__main__":
    main()