from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def build_linear_predictor(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    feature_weights: Mapping[str, float] | None = None,
    *,
    treatment_column: str | None = None,
    treatment_weight: float = 0.0,
    output_column: str = "selection_linear_score",
) -> tuple[pd.DataFrame, dict]:
    """
    Build the linear predictor for the selection model.

    The linear predictor has the form:

        eta_i = sum_j w_j * x_ij + delta * t_i

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_columns : Sequence[str]
        Columns used as predictors in the selection model.
    feature_weights : Mapping[str, float] | None, default=None
        Optional per-feature weights. If None, all weights are set to 1.0.
    treatment_column : str | None, default=None
        Optional treatment column to include in the predictor.
    treatment_weight : float, default=0.0
        Weight associated with the treatment column.
    output_column : str, default="selection_linear_score"
        Name of the output column containing the linear predictor.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Dataframe with linear predictor column added, and metadata.

    Raises
    ------
    ValueError
        If required feature weights are missing.
    """
    df_out = df.copy()

    if not feature_columns:
        raise ValueError("feature_columns must contain at least one column.")

    if feature_weights is None:
        weights = {col: 1.0 for col in feature_columns}
    else:
        missing_weights = [col for col in feature_columns if col not in feature_weights]
        if missing_weights:
            raise ValueError(
                f"Missing feature weights for columns: {missing_weights}"
            )
        weights = {col: float(feature_weights[col]) for col in feature_columns}

    linear_score = pd.Series(0.0, index=df_out.index, dtype=float)

    for col in feature_columns:
        linear_score = linear_score + weights[col] * df_out[col]

    treatment_included = treatment_column is not None and treatment_weight != 0.0
    if treatment_included:
        linear_score = linear_score + float(treatment_weight) * df_out[treatment_column]

    df_out[output_column] = linear_score

    metadata = {
        "feature_columns": list(feature_columns),
        "feature_weights": weights,
        "treatment_column": treatment_column,
        "treatment_weight": float(treatment_weight),
        "treatment_included": treatment_included,
        "output_column": output_column,
        "linear_score_summary": {
            "mean": float(df_out[output_column].mean()),
            "std": float(df_out[output_column].std(ddof=0)),
            "min": float(df_out[output_column].min()),
            "max": float(df_out[output_column].max()),
        },
    }

    return df_out, metadata