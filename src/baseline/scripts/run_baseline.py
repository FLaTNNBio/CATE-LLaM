import argparse
import pandas as pd

import src
from src.baseline import BaselineConfig, split_by_subject, run_baseline
from src.baseline.features import default_feature_columns, coerce_numeric_columns

print("baseline package:", src.baseline.__file__)
print("estimator file:", src.baseline.estimator.__file__)
print("features file:", src.baseline.features.__file__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to analytic_v0_extended_prepared.parquet")
    ap.add_argument("--out_smd", default="baseline_smd.csv")
    args = ap.parse_args()

    cfg = BaselineConfig()

    df = pd.read_parquet(args.data)

    # feature list
    num_cols, cat_cols, dropped = default_feature_columns(
        df,
        id_col=cfg.id_col,
        subject_col=cfg.subject_col,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        drop_cols=cfg.drop_cols,
    )

    df[num_cols] = coerce_numeric_columns(df, num_cols)

    print(f"Dropped columns (auto): {dropped[:20]} ... total={len(dropped)}")
    print(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

    # split (train/val/test) - for now train vs test
    splits = split_by_subject(
        df,
        subject_col=cfg.subject_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    train_idx = splits["train"]
    test_idx = splits["test"]

    res = run_baseline(df=df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        train_idx=train_idx,
        test_idx=test_idx,
        n_folds=cfg.n_folds,
        ps_clip_range=cfg.ps_clip,
        weight_trim_q=cfg.weight_trim_quantiles,
        seed=cfg.random_state,
    )

    print("=== BASELINE RESULTS ===")
    print(f"PS AUC train: {res.ps_auc_train:.3f}")
    print(f"PS AUC test : {res.ps_auc_test:.3f}")
    print(f"ATE IPTW (test): {res.ate_iptw:.4f}")
    print(f"ESS IPTW (test): {res.ess_iptw:.1f}")
    print(f"ATE AIPW (test cross-fit): {res.ate_aipw:.4f}")

    res.smd.to_csv(args.out_smd, index=False)
    print(f"Saved SMD table to {args.out_smd}")

if __name__ == "__main__":
    main()
