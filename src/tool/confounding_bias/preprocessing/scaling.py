from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize_covariates(
    work_df: pd.DataFrame,
    covariates: Sequence[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize the specified covariates using scikit-learn StandardScaler.

    Parameters
    ----------
    work_df : pd.DataFrame
        Input dataframe containing the covariates.
    covariates : Sequence[str]
        Covariates to standardize.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        Scaled covariate dataframe and fitted scaler.
    """
    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(work_df[list(covariates)]),
        columns=list(covariates),
        index=work_df.index,
    )

    return X_scaled, scaler