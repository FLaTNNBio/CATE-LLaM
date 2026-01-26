"""
Aim: feature column selection and processing.

Build 2 lists:
- num_cols (numerical)
- cat_cols (categorical: gender/race etc.)
- dropped (auto-excluded) columns.

What is excluded:
- ids and target/outcome cols (config): (stay_id, subject_id, hadm_id, t_vaso6h, y_hosp_mort)
- id-like columns: (end with _id, start with id_)
- date/time (intime)

Conversion:
- From object to numeric: (age)

Coercion:
- Conversion of numeric columns to numeric types: if not parserable -> NaN
  e.g. age with values like '> 89'
"""


import pandas as pd
from typing import Sequence

ID_LIKE_SUBSTR = ("_id", "id_", "stay_id", "subject_id", "hadm_id")

def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series)

def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )

def _is_object_but_numeric(series: pd.Series, sample_n: int = 2000, min_parse_rate: float = 0.95) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    s = series.dropna()
    if len(s) == 0:
        return False
    s = s.sample(n=min(sample_n, len(s)), random_state=0)
    parsed = pd.to_numeric(s, errors="coerce")
    return parsed.notna().mean() >= min_parse_rate


def default_feature_columns(
    df: pd.DataFrame,
    id_col: str,
    subject_col: str,
    treatment_col: str,
    outcome_col: str,
    drop_cols: Sequence[str] | None = None,
    drop_datetime: bool = True,
    drop_id_like: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns:
      - numeric feature cols
      - categorical feature cols
      - dropped cols (for logging/debug)
    """
    exclude = {id_col, subject_col, treatment_col, outcome_col}
    if drop_cols:
        exclude |= set(drop_cols)

    dropped = []
    num_cols = []
    cat_cols = []

    for c in df.columns:
        if c in exclude:
            dropped.append(c)
            continue

        if drop_id_like:
            cl = c.lower()
            if any(s in cl for s in ID_LIKE_SUBSTR):
                dropped.append(c)
                continue

        s = df[c]

        if _is_object_but_numeric(s):
            num_cols.append(c)
            continue

        if drop_datetime and _is_datetime(s):
            dropped.append(c)
            continue

        if _is_categorical(s):
            cat_cols.append(c)
        else:
            # numeric (or boolean)
            num_cols.append(c)

    return num_cols, cat_cols, dropped


def coerce_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Ensure numeric columns are numeric (age issues, etc.). Non-parsable values -> NaN.
    """
    out = df[cols].copy()
    for c in cols:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(float)
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str,
    outcome_col: str,
):
    """
    Build feature matrix X, treatment vector T, outcome vector Y.
    :param df: dataframe
    :param feature_cols: feature columns to use
    :param treatment_col: treatment columns to use
    :param outcome_col: outcome columns to use
    :return: tuple (X, T, Y)
    """
    return df[feature_cols], df[treatment_col].astype(int).values, df[outcome_col].astype(int).values
