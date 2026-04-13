import argparse
import pandas as pd

from src.baseline import default_feature_columns, BaselineConfig


def report_column_types(df: pd.DataFrame, cols: list[str], max_unique: int = 10) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c]
        dtype = str(s.dtype)
        n_null = int(s.isna().sum())
        frac_null = float(n_null) / len(s)
        sample = None
        try:
            sample = s.dropna().iloc[0]
        except Exception:
            sample = None

        nunique = int(s.nunique(dropna=True))
        uniq_preview = None
        if nunique <= max_unique:
            uniq_preview = list(s.dropna().unique()[:max_unique])
        rows.append({
            "col": c,
            "dtype": dtype,
            "null_frac": frac_null,
            "nunique": nunique,
            "sample": sample,
            "uniq_preview": uniq_preview
        })
    return pd.DataFrame(rows).sort_values(["dtype", "null_frac"], ascending=[True, False]).reset_index(drop=True)

def find_non_numeric_in_numeric(df: pd.DataFrame, num_cols: list[str]) -> list[str]:
    bad = []
    for c in num_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        bad.append(c)
    return bad

def find_datetime_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            out.append(c)
    return out


if __name__ == "__main__":
    # Argument --data <path> from cmd
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to analytic_v0_extended_prepared.parquet")
    args = ap.parse_args()
    df = pd.read_parquet(args.data)

    cfg = BaselineConfig()

    # feature list
    num_cols, cat_cols, dropped = default_feature_columns(
        df,
        id_col=cfg.id_col,
        subject_col=cfg.subject_col,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        drop_cols=cfg.drop_cols,
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(report_column_types(df, df.columns.tolist()))

    print("Non-numeric in numeric:", find_non_numeric_in_numeric(df, num_cols))
    print("Datetime columns:", find_datetime_cols(df, df.columns.tolist()))