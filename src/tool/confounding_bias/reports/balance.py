from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def standardized_mean_difference(
    x_treated: pd.Series,
    x_control: pd.Series,
) -> float:
    """
    Compute the standardized mean difference between treated and control groups.

    Parameters
    ----------
    x_treated : pd.Series
        Covariate values among treated units.
    x_control : pd.Series
        Covariate values among control units.

    Returns
    -------
    float
        Standardized mean difference.
    """
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
    covariates: Sequence[str],
    treatment_col: str,
) -> pd.DataFrame:
    """
    Compute a covariate balance table comparing treated vs control groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    covariates : Sequence[str]
        Covariates to include in the balance table.
    treatment_col : str
        Treatment column used to define treated and control groups.

    Returns
    -------
    pd.DataFrame
        Balance table sorted by descending absolute SMD.
    """
    rows = []
    treated_mask = df[treatment_col] == 1
    control_mask = df[treatment_col] == 0

    for col in covariates:
        treated = df.loc[treated_mask, col]
        control = df.loc[control_mask, col]

        rows.append(
            {
                "covariate": col,
                "treated_mean": treated.mean(),
                "control_mean": control.mean(),
                "mean_diff": treated.mean() - control.mean(),
                "smd": standardized_mean_difference(treated, control),
            }
        )

    balance_df = pd.DataFrame(rows).sort_values(
        by="smd",
        key=np.abs,
        ascending=False,
    )

    return balance_df