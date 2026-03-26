from __future__ import annotations

from typing import Sequence


def build_default_weights(covariates: Sequence[str]) -> dict[str, float]:
    """
    Build the default artificial propensity-score weights.

    Parameters
    ----------
    covariates : Sequence[str]
        Covariates requested by the user.

    Returns
    -------
    dict[str, float]
        Subset of default weights corresponding to the requested covariates.
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