#!/usr/bin/env python3

# USAGE: use csv file of AIDS rtc (put it in the same folder as this script or provide path) and run:
# COMMAND: python ps_confounding_bias.py --input aids_csv.csv --output-csv actg175_observational.csv --output-report actg175_observational_report.json --covariates age wtkg karnof oprior preanti strat cd40 symptom --verbose
"""
Transform an RCT dataset into a pseudo-observational dataset
by generating an artificial propensity score and resampling treatment.

Use case:
- Input dataset: ACTG175-like CSV
- Outcome: label
- Original treatment: treat
- New pseudo-observational treatment: treat_obs

Author: <your_name>
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# =========================
# Configuration
# =========================

DEFAULT_COVARIATES = [
    "age",
    "wtkg",
    "karnof",
    "oprior",
    "preanti",
    "strat",
    "cd40",
    "symptom",
]

DEFAULT_OUTCOME = "label"
DEFAULT_ORIGINAL_TREATMENT = "treat"
DEFAULT_NEW_TREATMENT = "treat_obs"
DEFAULT_PS_COL = "ps_artificial"


@dataclass
class TransformConfig:
    input_path: Path
    output_csv: Path
    output_report: Path
    covariates: List[str]
    outcome_col: str
    original_treatment_col: str
    new_treatment_col: str
    ps_col: str
    clip_min: float
    clip_max: float
    intercept: float
    seed: int


# =========================
# Logging
# =========================

def setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# =========================
# Utilities
# =========================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    return df


def select_and_clean_data(
    df: pd.DataFrame,
    covariates: List[str],
    outcome_col: str,
    original_treatment_col: str,
) -> pd.DataFrame:
    required = covariates + [outcome_col, original_treatment_col]
    validate_columns(df, required)

    work_df = df[required].copy()

    # Drop rows with missing values only on needed columns
    before = len(work_df)
    work_df = work_df.dropna().reset_index(drop=True)
    after = len(work_df)

    logging.info("Rows before dropna: %d", before)
    logging.info("Rows after dropna:  %d", after)

    if after == 0:
        raise ValueError("No rows left after dropping missing values.")

    return work_df


def standardize_covariates(work_df: pd.DataFrame, covariates: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(work_df[covariates]),
        columns=covariates,
        index=work_df.index,
    )
    return X_scaled, scaler


def build_default_weights(covariates: List[str]) -> Dict[str, float]:
    """
    Reasonable default weights for a clinically plausible artificial assignment policy.
    Signs are illustrative, not claims of true clinical decision-making.
    """
    candidate_weights = {
        "age": 0.35,
        "wtkg": -0.20,
        "karnof": -0.60,
        "oprior": 0.55,
        "preanti": 0.45,
        "strat": 0.40,
        "cd40": -1.00,
        "symptom": 0.50,
    }
    return {k: candidate_weights[k] for k in covariates if k in candidate_weights}


def compute_artificial_propensity(
    X_scaled: pd.DataFrame,
    weights: Dict[str, float],
    intercept: float,
    clip_min: float,
    clip_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not weights:
        raise ValueError("Weights dictionary is empty. Cannot build propensity score.")

    linear_predictor = np.full(shape=len(X_scaled), fill_value=intercept, dtype=float)

    for col, weight in weights.items():
        if col not in X_scaled.columns:
            raise ValueError(f"Weight specified for unknown covariate: {col}")
        linear_predictor += weight * X_scaled[col].values

    ps_raw = sigmoid(linear_predictor)
    ps = np.clip(ps_raw, clip_min, clip_max)

    return ps_raw, ps


def sample_treatment(ps: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(n=1, p=ps, size=len(ps))


def evaluate_assignment_strength(X_scaled: pd.DataFrame, treatment: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_scaled, treatment)
    pred = clf.predict_proba(X_scaled)[:, 1]
    return roc_auc_score(treatment, pred)


def standardized_mean_difference(x_treated: pd.Series, x_control: pd.Series) -> float:
    mean_t = x_treated.mean()
    mean_c = x_control.mean()
    var_t = x_treated.var(ddof=1)
    var_c = x_control.var(ddof=1)
    pooled_sd = np.sqrt((var_t + var_c) / 2.0)
    if pooled_sd == 0:
        return 0.0
    return (mean_t - mean_c) / pooled_sd


def compute_balance_table(
    df: pd.DataFrame,
    covariates: List[str],
    treatment_col: str,
) -> pd.DataFrame:
    rows = []
    treated_mask = df[treatment_col] == 1
    control_mask = df[treatment_col] == 0

    for col in covariates:
        treated = df.loc[treated_mask, col]
        control = df.loc[control_mask, col]

        rows.append({
            "covariate": col,
            "treated_mean": treated.mean(),
            "control_mean": control.mean(),
            "mean_diff": treated.mean() - control.mean(),
            "smd": standardized_mean_difference(treated, control),
        })

    balance_df = pd.DataFrame(rows).sort_values(by="smd", key=np.abs, ascending=False)
    return balance_df


def build_report(
    work_df: pd.DataFrame,
    covariates: List[str],
    outcome_col: str,
    original_treatment_col: str,
    new_treatment_col: str,
    ps_col: str,
    auc: float,
    weights: Dict[str, float],
    balance_df: pd.DataFrame,
    n_original_rows: int,
) -> Dict:
    report = {
        "n_rows_final": int(len(work_df)),
        "n_rows_original": int(n_original_rows),
        "outcome_col": outcome_col,
        "original_treatment_col": original_treatment_col,
        "new_treatment_col": new_treatment_col,
        "propensity_score_col": ps_col,
        "covariates_used": covariates,
        "weights_used": weights,
        "original_treatment_rate": float(work_df[original_treatment_col].mean()),
        "new_treatment_rate": float(work_df[new_treatment_col].mean()),
        "propensity_score_summary": {
            "min": float(work_df[ps_col].min()),
            "p01": float(work_df[ps_col].quantile(0.01)),
            "p05": float(work_df[ps_col].quantile(0.05)),
            "median": float(work_df[ps_col].median()),
            "p95": float(work_df[ps_col].quantile(0.95)),
            "p99": float(work_df[ps_col].quantile(0.99)),
            "max": float(work_df[ps_col].max()),
            "mean": float(work_df[ps_col].mean()),
        },
        "assignment_auc_predicting_new_treatment_from_X": float(auc),
        "top_balance_shifts_by_abs_smd": balance_df.head(10).to_dict(orient="records"),
    }
    return report


def parse_args() -> TransformConfig:
    parser = argparse.ArgumentParser(
        description="Create a pseudo-observational version of an RCT dataset using an artificial propensity score."
    )

    parser.add_argument("--input", type=Path, required=True, help="Path to input CSV.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to output transformed CSV.")
    parser.add_argument("--output-report", type=Path, required=True, help="Path to output JSON report.")
    parser.add_argument(
        "--covariates",
        type=str,
        nargs="+",
        default=DEFAULT_COVARIATES,
        help="List of pre-treatment covariates to use in the artificial propensity score."
    )
    parser.add_argument("--outcome-col", type=str, default=DEFAULT_OUTCOME)
    parser.add_argument("--original-treatment-col", type=str, default=DEFAULT_ORIGINAL_TREATMENT)
    parser.add_argument("--new-treatment-col", type=str, default=DEFAULT_NEW_TREATMENT)
    parser.add_argument("--ps-col", type=str, default=DEFAULT_PS_COL)
    parser.add_argument("--clip-min", type=float, default=0.05)
    parser.add_argument("--clip-max", type=float, default=0.95)
    parser.add_argument("--intercept", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not (0.0 < args.clip_min < args.clip_max < 1.0):
        raise ValueError("clip_min and clip_max must satisfy 0 < clip_min < clip_max < 1.")

    return TransformConfig(
        input_path=args.input,
        output_csv=args.output_csv,
        output_report=args.output_report,
        covariates=args.covariates,
        outcome_col=args.outcome_col,
        original_treatment_col=args.original_treatment_col,
        new_treatment_col=args.new_treatment_col,
        ps_col=args.ps_col,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        intercept=args.intercept,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()

    logging.info("Loading data from %s", config.input_path)
    df = load_data(config.input_path)
    n_original_rows = len(df)

    logging.info("Selecting and cleaning data")
    work_df = select_and_clean_data(
        df=df,
        covariates=config.covariates,
        outcome_col=config.outcome_col,
        original_treatment_col=config.original_treatment_col,
    )

    logging.info("Standardizing covariates")
    X_scaled, _ = standardize_covariates(work_df, config.covariates)

    logging.info("Building artificial propensity score")
    weights = build_default_weights(config.covariates)
    ps_raw, ps = compute_artificial_propensity(
        X_scaled=X_scaled,
        weights=weights,
        intercept=config.intercept,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
    )

    work_df[f"{config.ps_col}_raw"] = ps_raw
    work_df[config.ps_col] = ps

    logging.info("Sampling new pseudo-observational treatment")
    work_df[config.new_treatment_col] = sample_treatment(ps=ps, seed=config.seed)

    logging.info("Evaluating assignment strength")
    auc = evaluate_assignment_strength(X_scaled, work_df[config.new_treatment_col].values)

    logging.info("Computing covariate balance table")
    balance_df = compute_balance_table(
        df=work_df,
        covariates=config.covariates,
        treatment_col=config.new_treatment_col,
    )

    logging.info("Saving transformed dataset to %s", config.output_csv)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    work_df.to_csv(config.output_csv, index=False)

    report = build_report(
        work_df=work_df,
        covariates=config.covariates,
        outcome_col=config.outcome_col,
        original_treatment_col=config.original_treatment_col,
        new_treatment_col=config.new_treatment_col,
        ps_col=config.ps_col,
        auc=auc,
        weights=weights,
        balance_df=balance_df,
        n_original_rows=n_original_rows,
    )

    logging.info("Saving report to %s", config.output_report)
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Transformation completed ===")
    print(f"Rows retained: {len(work_df)} / {n_original_rows}")
    print(f"Original treatment rate: {work_df[config.original_treatment_col].mean():.3f}")
    print(f"New treatment rate:      {work_df[config.new_treatment_col].mean():.3f}")
    print(f"Mean artificial PS:      {work_df[config.ps_col].mean():.3f}")
    print(f"AUC[Treat_obs ~ X]:      {auc:.3f}")
    print("\nTop covariate shifts by |SMD|:")
    print(balance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()