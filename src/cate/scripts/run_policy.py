"""
    Script to evaluate treatment policies derived from CATE estimates.
    Uses doubly robust policy value estimation.

    Output:
        JSON file with 3 policy values:
        - V(treat-none) = expected mortality if no one treated
        - V(treat-all)  = expected mortality if everyone treated
        - V(policy)     = expected mortality under policy derived from tau_hat

    Interpretation:
        - Lower values are better since outcome is mortality (0/1).
        - Delta values show improvement vs baselines.
        - Bootstrap CIs provide uncertainty estimates.
        - Policy is useful if it reduces expected mortality vs baselines:
            -V(policy) < V(treat-all)
            -V(policy) < V(treat-none) (ideal)
        - Treatment rate under policy is also reported.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.baseline.config import BaselineConfig
from src.baseline.split import split_by_subject
from src.baseline.features import default_feature_columns, coerce_numeric_columns
from src.baseline.models import HGBConfig, make_hgb_pipeline
from src.baseline.summary import save_json

from src.config import get_config, CONFIGS

from src.cate.policy import (
    policy_from_tau,
    dr_policy_value,
    bootstrap_policy_value,
    PolicyValueConfig,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CONFIGS.keys()), default="rbc_v1", help="Which dataset config to use")
    ap.add_argument("--data", required=False, help="Path to analytic_v0_extended_prepared.parquet")
    ap.add_argument("--tau_pred", required=False, help="Path to artifacts/cate/dr_tau_test.parquet")
    ap.add_argument("--out_dir", help="Output directory")
    ap.add_argument("--policy", choices=["tau_gt_0", "top_frac"], default="tau_gt_0")
    ap.add_argument("--top_frac", type=float, default=0.2, help="Top fraction to treat if policy=top_frac")
    ap.add_argument("--n_boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = get_config(args.dataset)

    if args.data is None:
        args.data = cfg.data_path
    if args.out_dir is None:
        args.out_dir = cfg.out_dir
    if args.tau_pred is None:
        args.tau_pred = cfg.out_dir + "/dr_tau_test.parquet"

    seed = int(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load full data for train/test split + features
    df = pd.read_parquet(args.data)

    num_cols, cat_cols, dropped = default_feature_columns(
        df,
        id_col=cfg.id_col,
        subject_col=cfg.subject_col,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        drop_cols=cfg.drop_cols,
    )
    df[num_cols] = coerce_numeric_columns(df, num_cols)
    feat_cols = num_cols + cat_cols

    splits = split_by_subject(
        df,
        subject_col=cfg.subject_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=seed,
    )
    tr_idx, te_idx = splits["train"], splits["test"]

    X = df[feat_cols]
    T = df[cfg.treatment_col].astype(int).values
    Y = df[cfg.outcome_col].astype(int).values

    X_tr, T_tr, Y_tr = X.iloc[tr_idx], T[tr_idx], Y[tr_idx]
    X_te, T_te, Y_te = X.iloc[te_idx], T[te_idx], Y[te_idx]

    # --- load tau predictions (must be aligned with test rows/order used in run_dr)
    pred = pd.read_parquet(args.tau_pred)

    # safety: align by stay_id (robust against ordering differences)
    te_ids = df.iloc[te_idx][cfg.id_col].values
    te_map = pd.DataFrame({cfg.id_col: te_ids, "_pos": np.arange(len(te_ids))})
    pred2 = pred.merge(te_map, on=cfg.id_col, how="inner").sort_values("_pos")
    if len(pred2) != len(te_ids):
        raise ValueError("tau_pred does not match the test split size (alignment by stay_id failed).")

    tau_hat = pred2["tau_hat"].values.astype(float)

    # PS: prefer saved ps_hat in pred file, else recompute
    if "ps_hat" in pred2.columns:
        ps_hat = pred2["ps_hat"].values.astype(float)
    else:
        ps_cfg = HGBConfig(random_state=seed)
        ps_model = make_hgb_pipeline(num_cols, cat_cols, ps_cfg)
        ps_model.fit(X_tr, T_tr)
        ps_hat = ps_model.predict_proba(X_te)[:, 1]

    # --- outcome nuisance models m1, m0 fit on TRAIN
    out_cfg = HGBConfig(random_state=seed, max_depth=3, min_samples_leaf=50)
    out1 = make_hgb_pipeline(num_cols, cat_cols, out_cfg)
    out0 = make_hgb_pipeline(num_cols, cat_cols, out_cfg)

    mask1 = (T_tr == 1)
    mask0 = (T_tr == 0)
    out1.fit(X_tr[mask1], Y_tr[mask1])
    out0.fit(X_tr[mask0], Y_tr[mask0])

    mu1_hat = out1.predict_proba(X_te)[:, 1]
    mu0_hat = out0.predict_proba(X_te)[:, 1]

    # --- build policies
    pi_policy = policy_from_tau(tau_hat, kind=args.policy, top_frac=args.top_frac)
    pi_none = np.zeros_like(pi_policy)
    pi_all = np.ones_like(pi_policy)

    pvcfg = PolicyValueConfig(ps_clip=cfg.ps_clip)

    # --- DR policy values (expected mortality under policy)
    v_none = dr_policy_value(y=Y_te, t=T_te, pi=pi_none, ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat, cfg=pvcfg)
    v_all  = dr_policy_value(y=Y_te, t=T_te, pi=pi_all,  ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat, cfg=pvcfg)
    v_pi   = dr_policy_value(y=Y_te, t=T_te, pi=pi_policy, ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat, cfg=pvcfg)

    # bootstrap CIs (optional but useful)
    ci_none = bootstrap_policy_value(y=Y_te, t=T_te, pi=pi_none, ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat,
                                    n_boot=args.n_boot, seed=seed, cfg=pvcfg)
    ci_all  = bootstrap_policy_value(y=Y_te, t=T_te, pi=pi_all,  ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat,
                                    n_boot=args.n_boot, seed=seed, cfg=pvcfg)
    ci_pi   = bootstrap_policy_value(y=Y_te, t=T_te, pi=pi_policy, ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat,
                                    n_boot=args.n_boot, seed=seed, cfg=pvcfg)

    # treatment rates under policy
    treat_rate = float(np.mean(pi_policy))

    results = {
        "meta": {
            "n_test": int(len(Y_te)),
            "ps_clip": {"lo": float(cfg.ps_clip[0]), "hi": float(cfg.ps_clip[1])},
            "policy": args.policy,
            "top_frac": float(args.top_frac),
            "treat_rate_policy": treat_rate,
        },
        "values": {
            "treat_none": v_none,
            "treat_all": v_all,
            "policy": v_pi,
            "delta_policy_vs_none": float(v_pi - v_none),
            "delta_policy_vs_all": float(v_pi - v_all),
        },
        "bootstrap_ci": {
            "treat_none": ci_none,
            "treat_all": ci_all,
            "policy": ci_pi,
        },
    }

    out_path = out_dir / "policy_value.json"
    save_json(out_path, results)
    print(f"Saved policy value -> {out_path}")

    # headline print (lower is better for mortality)
    print("=== DR Policy Value (expected mortality; lower is better) ===")
    print(f"Treat-none: {v_none:.4f}")
    print(f"Treat-all : {v_all:.4f}")
    print(f"Policy    : {v_pi:.4f}  (treat_rate={treat_rate:.3f})")
    print(f"Δ Policy - None: {v_pi - v_none:+.4f}")
    print(f"Δ Policy - All : {v_pi - v_all:+.4f}")

if __name__ == "__main__":
    main()
