from __future__ import annotations

import pandas as pd

from .evaluation import evaluate_assignment_strength
from .model import compute_artificial_propensity
from .sampling import sample_treatment
from .weights import build_default_weights


def run_propensity_pipeline(
    work_df: pd.DataFrame,
    X_scaled: pd.DataFrame,
    config,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the artificial propensity-score confounding pipeline.

    This function performs the following steps:
    - build default weights
    - compute raw and clipped artificial propensity score
    - sample new treatment
    - evaluate assignment strength with AUC

    Parameters
    ----------
    work_df : pd.DataFrame
        Cleaned working dataframe containing original columns.
    X_scaled : pd.DataFrame
        Standardized covariates used in the artificial propensity model.
    config : object
        Configuration object with the required attributes.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Annotated dataframe and pipeline metadata.
    """
    df_out = work_df.copy()

    weights = build_default_weights(config.covariates)

    ps_raw, ps = compute_artificial_propensity(
        X_scaled=X_scaled,
        weights=weights,
        intercept=config.intercept,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
    )

    df_out[f"{config.ps_col}_raw"] = ps_raw
    df_out[config.ps_col] = ps

    df_out[config.new_treatment_col] = sample_treatment(
        ps=ps,
        seed=config.seed,
    )

    auc = evaluate_assignment_strength(
        X_scaled=X_scaled,
        treatment=df_out[config.new_treatment_col].values,
    )

    metadata = {
        "weights_used": weights,
        "ps_raw_summary": {
            "min": float(df_out[f'{config.ps_col}_raw'].min()),
            "max": float(df_out[f'{config.ps_col}_raw'].max()),
            "mean": float(df_out[f'{config.ps_col}_raw'].mean()),
        },
        "ps_summary": {
            "min": float(df_out[config.ps_col].min()),
            "p01": float(df_out[config.ps_col].quantile(0.01)),
            "p05": float(df_out[config.ps_col].quantile(0.05)),
            "median": float(df_out[config.ps_col].median()),
            "p95": float(df_out[config.ps_col].quantile(0.95)),
            "p99": float(df_out[config.ps_col].quantile(0.99)),
            "max": float(df_out[config.ps_col].max()),
            "mean": float(df_out[config.ps_col].mean()),
        },
        "original_treatment_rate": float(df_out[config.original_treatment_col].mean()),
        "new_treatment_rate": float(df_out[config.new_treatment_col].mean()),
        "assignment_auc_predicting_new_treatment_from_X": float(auc),
    }

    return df_out, metadata