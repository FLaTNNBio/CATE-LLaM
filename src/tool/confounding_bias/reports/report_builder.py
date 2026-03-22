from __future__ import annotations

from typing import Any

import pandas as pd


def build_report(
    work_df: pd.DataFrame,
    covariates: list[str],
    outcome_col: str,
    original_treatment_col: str,
    new_treatment_col: str,
    ps_col: str,
    auc: float,
    weights: dict[str, float],
    balance_df: pd.DataFrame,
    n_original_rows: int,
) -> dict[str, Any]:
    """
    Build the JSON report for the confounding-bias transformation.

    Parameters
    ----------
    work_df : pd.DataFrame
        Final transformed dataframe.
    covariates : list[str]
        Covariates used in the artificial propensity score.
    outcome_col : str
        Outcome column name.
    original_treatment_col : str
        Original treatment column name.
    new_treatment_col : str
        New sampled pseudo-observational treatment column name.
    ps_col : str
        Propensity score column name.
    auc : float
        AUC of predicting new treatment from X.
    weights : dict[str, float]
        Artificial propensity-score weights.
    balance_df : pd.DataFrame
        Covariate balance table.
    n_original_rows : int
        Number of rows in the original input dataframe.

    Returns
    -------
    dict[str, Any]
        JSON-serializable report.
    """
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