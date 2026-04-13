from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .balance import compute_balance_table, summarize_absolute_smd
from .summary import summarize_dataset


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-safe Python types.
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if pd.isna(obj):
        return None

    return obj


def build_selection_report(
    *,
    config,
    df_input: pd.DataFrame,
    df_annotated: pd.DataFrame,
    df_selected: pd.DataFrame,
    pipeline_metadata: dict,
) -> dict:
    """
    Build the final JSON-style report for a selection-bias run.

    Parameters
    ----------
    config : object
        Selection config with a to_dict() method and standard attributes.
    df_input : pd.DataFrame
        Original input dataframe.
    df_annotated : pd.DataFrame
        Eligible dataframe with selection columns appended.
    df_selected : pd.DataFrame
        Final selected dataframe.
    pipeline_metadata : dict
        Metadata returned by the selection pipeline.

    Returns
    -------
    dict
        JSON-serializable report dictionary.
    """
    input_summary = summarize_dataset(
        df_input,
        treatment_column=config.treatment_column,
        outcome_column=config.outcome_column,
        include_covariate_summaries=False,
    )

    eligible_summary = summarize_dataset(
        df_annotated,
        treatment_column=config.treatment_column,
        outcome_column=config.outcome_column,
        probability_column=config.probability_column,
        include_covariate_summaries=False,
    )

    selected_summary = summarize_dataset(
        df_selected,
        treatment_column=config.treatment_column,
        outcome_column=config.outcome_column,
        include_covariate_summaries=False,
    )

    balance_table = compute_balance_table(
        df_before=df_annotated,
        df_after=df_selected,
        columns=config.covariates,
    )

    report = {
        "config": config.to_dict(),
        "data_summary": {
            "input": input_summary,
            "eligible": eligible_summary,
            "selected": selected_summary,
        },
        "selection_process": pipeline_metadata,
        "balance": {
            "eligible_vs_selected": balance_table,
            "eligible_vs_selected_summary": summarize_absolute_smd(balance_table),
        },
    }

    return _sanitize_for_json(report)