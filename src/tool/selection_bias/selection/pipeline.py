from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocessing import (
    apply_missing_policy,
    standardize_columns,
    validate_selection_inputs,
)
from .calibration import calibrate_intercept
from .model import build_linear_predictor
from .sampling import (
    compute_selection_probabilities,
    sample_selection_indicator,
)


def run_selection_pipeline(
    df: pd.DataFrame,
    config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full pre-treatment selection bias pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    config : object
        Configuration object with the required attributes.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict]
        - annotated dataframe
        - selected dataframe
        - pipeline metadata / report payload
    """
    validate_selection_inputs(
        df=df,
        covariates=config.covariates,
        treatment_column=config.treatment_column,
        outcome_column=config.outcome_column,
        require_numeric_covariates=True,
        validate_treatment_as_binary=config.validate_treatment_as_binary,
        id_column=config.id_column,
    )

    df_eligible, missing_metadata = apply_missing_policy(
        df=df,
        columns=config.covariates,
        policy=config.missing_policy,
    )

    df_preprocessed, feature_columns, scaling_metadata = standardize_columns(
        df=df_eligible,
        columns=config.covariates,
        skip_binary=True,
        output_suffix=config.scaled_suffix,
    )

    treatment_column_for_model = (
        config.treatment_column if config.include_treatment_in_selection else None
    )
    treatment_weight = (
        float(config.treatment_weight) if config.include_treatment_in_selection else 0.0
    )

    resolved_feature_weights = config.resolved_feature_weights(feature_columns)

    df_scored, model_metadata = build_linear_predictor(
        df=df_preprocessed,
        feature_columns=feature_columns,
        feature_weights=resolved_feature_weights,
        treatment_column=treatment_column_for_model,
        treatment_weight=treatment_weight,
        output_column=config.linear_score_column,
    )

    intercept, calibration_metadata = calibrate_intercept(
        linear_score=df_scored[config.linear_score_column],
        target_inclusion_rate=config.target_inclusion_rate,
        strength=config.selection_strength,
    )

    df_probs, probability_metadata = compute_selection_probabilities(
        linear_score=df_scored[config.linear_score_column],
        intercept=intercept,
        strength=config.selection_strength,
        output_column=config.probability_column,
    )

    rng = np.random.default_rng(config.seed)

    df_sel, sampling_metadata = sample_selection_indicator(
        probabilities=df_probs[config.probability_column],
        rng=rng,
        output_column=config.indicator_column,
    )

    df_annotated = df_scored.join(df_probs).join(df_sel)

    df_selected = df_annotated.loc[
        df_annotated[config.indicator_column] == 1
    ].copy()

    if not getattr(config, "keep_selection_columns", True):
        columns_to_drop = [
            col for col in [
                config.linear_score_column,
                config.probability_column,
                config.indicator_column,
            ]
            if col in df_selected.columns
        ]
        df_selected = df_selected.drop(columns=columns_to_drop)

    metadata = {
        "n_rows_input": int(len(df)),
        "n_rows_eligible": int(len(df_eligible)),
        "n_rows_selected": int(len(df_selected)),
        "realized_inclusion_rate_among_eligible": (
            int(len(df_selected)) / int(len(df_eligible))
            if len(df_eligible) > 0
            else 0.0
        ),
        "feature_columns_used_in_model": list(feature_columns),
        "missing": missing_metadata,
        "scaling": scaling_metadata,
        "model": model_metadata,
        "calibration": calibration_metadata,
        "probabilities": probability_metadata,
        "sampling": sampling_metadata,
    }

    return df_annotated, df_selected, metadata