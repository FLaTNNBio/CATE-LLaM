# python
import argparse
import os

from tensorflow import keras
import pandas as pd
import numpy as np
from pathlib import Path

from src.baseline import default_feature_columns
from src.baseline.features import coerce_numeric_columns
from src.config import get_config, CONFIGS
from src.models.bcauss.models import make_bcauss
from src.baseline.split import split_by_subject
from tensorflow.keras.optimizers import Adam


def dataset_engineering_bcauss(
    df: pd.DataFrame,
    id_col: str,
    subject_col: str,
    treatment_col: str,
    outcome_col: str,
    drop_cols: list[str] = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    num_cols, cat_cols, dropped = default_feature_columns(
        df,
        id_col=id_col,
        subject_col=subject_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        drop_cols=drop_cols,
    )
    df[num_cols] = coerce_numeric_columns(df, num_cols)

    # One-hot encode categorical columns
    # Map known treatment labels to numeric (ensure HFNC=1, NIV=0)
    if df[treatment_col].dtype == object or df[treatment_col].dtype == "O":
        mapping = {"HFNC": 1, "NIV": 0}
        if df[treatment_col].isin(mapping.keys()).any():
            df[treatment_col] = df[treatment_col].map(mapping)
    # Only encode categorical columns that are not identifier/treatment/outcome
    cat_to_encode = [c for c in cat_cols if c not in {id_col, subject_col, treatment_col, outcome_col}]
    if cat_to_encode:
        df = pd.get_dummies(df, columns=cat_to_encode, drop_first=True)
    # Update cat_cols to reflect new one-hot encoded columns (everything that's not numeric or id/treatment/outcome)
    new_cat_cols = [col for col in df.columns if
                    col not in num_cols + [id_col, subject_col, treatment_col, outcome_col]]
    # Ensure new categorical columns are numeric
    if new_cat_cols:
        df[new_cat_cols] = df[new_cat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    cat_cols = new_cat_cols

    # Data Augmentation for N/A
    # Stratify by treatment and outcome and fill N/A with mean of that group (plus small noise)
    for col in num_cols:
        if df[col].isna().sum() > 0:
            for t_val in df[treatment_col].unique():
                for y_val in df[outcome_col].unique():
                    mask = (df[treatment_col] == t_val) & (df[outcome_col] == y_val)
                    na_mask = mask & df[col].isna()
                    if not na_mask.any():
                        continue
                    mean_val = df.loc[mask, col].mean()
                    # fallback to global column mean if group mean is NaN, then 0.0 if still NaN
                    if pd.isna(mean_val):
                        mean_val = df[col].mean()
                        if pd.isna(mean_val):
                            mean_val = 0.0
                    noise = np.random.normal(0, 1e-3, size=na_mask.sum())
                    df.loc[na_mask, col] = mean_val + noise

    # MinMax Scale of numeric columns    to[0,1]
    for col in num_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val) + 1e-12

    return df, num_cols, cat_cols

# python
from tensorflow.keras import backend as K

def stable_bcauss_loss(y_true_concat, y_pred_concat, alpha=1.0, beta=1.0):
    """
    Stable BCAUSS loss with gradient clipping and proper scaling.
    y_true_concat: [y_true, t_true] concatenated
    y_pred_concat: [y0_pred, y1_pred, t_pred, epsilon] concatenated
    """
    y_true = y_true_concat[:, 0:1]
    t_true = y_true_concat[:, 1:2]

    y0_pred = y_pred_concat[:, 0:1]
    y1_pred = y_pred_concat[:, 1:2]
    t_pred = y_pred_concat[:, 2:3]

    # Clip predictions to prevent NaN
    t_pred = K.clip(t_pred, 1e-6, 1.0 - 1e-6)
    y0_pred = K.clip(y0_pred, -1e6, 1e6)
    y1_pred = K.clip(y1_pred, -1e6, 1e6)

    # Binary cross-entropy for treatment prediction (stable)
    bce = - (t_true * K.log(t_pred) + (1.0 - t_true) * K.log(1.0 - t_pred))
    treatment_loss = K.mean(bce)

    # Regression loss for outcome prediction
    loss0 = (1.0 - t_true) * K.square(y_true - y0_pred)
    loss1 = t_true * K.square(y_true - y1_pred)
    regression_loss = K.mean(loss0 + loss1)

    return alpha * regression_loss + beta * treatment_loss


def train_bcauss(
    X: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    X_val: pd.DataFrame = None,
    t_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    lr: float = 1e-5,
    checkpoint_path: Path = None,
) -> keras.Model:
    # --- coerce types to numeric numpy arrays ---
    if isinstance(X, pd.DataFrame):
        X_np = X.to_numpy(dtype=np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)

    t_np = np.asarray(t, dtype=np.float32).reshape(-1, 1)
    y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    # Normalize y to prevent large gradients
    y_mean = np.mean(y_np)
    y_std = np.std(y_np) + 1e-8
    y_np_norm = (y_np - y_mean) / y_std

    X_val_np = t_val_np = y_val_np = None
    if X_val is not None and t_val is not None and y_val is not None:
        if isinstance(X_val, pd.DataFrame):
            X_val_np = X_val.to_numpy(dtype=np.float32)
        else:
            X_val_np = np.asarray(X_val, dtype=np.float32)
        t_val_np = np.asarray(t_val, dtype=np.float32).reshape(-1, 1)
        y_val_np = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
        y_val_np_norm = (y_val_np - y_mean) / y_std

    input_dim = X_np.shape[1]
    model = make_bcauss(input_dim=input_dim, use_bce=True)

    # Use lower learning rate with gradient clipping
    opt = Adam(learning_rate=lr if lr else 1e-4, clipnorm=1.0, clipvalue=0.5)
    print("Compiling BCAUSS model with stable loss...")

    # Custom loss wrapper
    def bcauss_loss_fn(y_true, y_pred):
        return stable_bcauss_loss(y_true, y_pred, alpha=1.0, beta=1.0)

    model.compile(optimizer=opt, loss=bcauss_loss_fn)

    callbacks = []
    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                   restore_best_weights=True))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                       patience=max(2, patience // 2), min_lr=1e-7))

    if checkpoint_path is not None:
        callbacks.insert(0, keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True,
                                                            monitor='val_loss', mode='min'))

    # Concatenate y and t as target for loss function
    train_inputs = [X_np, y_np_norm, t_np]
    train_target = np.concatenate([y_np_norm, t_np], axis=1)

    if X_val_np is not None:
        val_inputs = [X_val_np, y_val_np_norm, t_val_np]
        val_target = np.concatenate([y_val_np_norm, t_val_np], axis=1)

        history = model.fit(
            train_inputs,
            train_target,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, val_target),
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )
    else:
        history = model.fit(
            train_inputs,
            train_target,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )
    return model

def predict_cate_bcauss(model: keras.Model, X: pd.DataFrame) -> np.ndarray:
    # Coerce X to float32 numpy array
    if isinstance(X, pd.DataFrame):
        X_np = X.to_numpy(dtype=np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)

    n_samples = X_np.shape[0]

    # Model expects [X, y_true, t_true] — provide dummy values for inference
    dummy_y = np.zeros((n_samples, 1), dtype=np.float32)
    dummy_t = np.zeros((n_samples, 1), dtype=np.float32)

    preds = model.predict([X_np, dummy_y, dummy_t], verbose=0)

    # Output format: [y0_pred, y1_pred, t_pred, epsilon] concatenated
    # CATE = E[Y(1)] - E[Y(0)] = y1_pred - y0_pred
    if preds.ndim == 2 and preds.shape[1] >= 2:
        y0_pred = preds[:, 0]
        y1_pred = preds[:, 1]
        cate = y1_pred - y0_pred
    else:
        raise ValueError("Unexpected model.predict output format for CATE computation.")

    return cate.reshape(-1)


# -------------------------
# Evalution Policy Scripts
# -------------------------

# python
def evaluate_policy_value(cate_estimates: np.ndarray, y: np.ndarray, t: np.ndarray,
                          ps: np.ndarray = None, threshold: float = 0.0) -> float:
    """
    Compute doubly-robust policy value for a given threshold.
    Treats if cate_estimate > threshold.
    """
    if ps is None:
        # Simple propensity estimate
        ps = np.full_like(y, fill_value=t.mean(), dtype=np.float32)
    ps = np.clip(ps, 0.01, 0.99)

    # Policy: treat if CATE > threshold
    pi = (cate_estimates > threshold).astype(np.float32)

    # IPW-style policy value
    w1 = t / ps
    w0 = (1 - t) / (1 - ps)

    value = np.mean(pi * w1 * y + (1 - pi) * w0 * y)
    return value


def bootstrap_policy_value(cate_estimates: np.ndarray, y: np.ndarray, t: np.ndarray,
                           ps: np.ndarray = None, threshold: float = 0.0,
                           n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Bootstrap confidence intervals for policy value."""
    n = len(y)
    boot_values = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        val = evaluate_policy_value(
            cate_estimates[idx], y[idx], t[idx],
            ps[idx] if ps is not None else None, threshold
        )
        boot_values.append(val)

    boot_values = np.array(boot_values)
    alpha = 1 - ci

    return {
        'mean': np.mean(boot_values),
        'sd': np.std(boot_values),
        'ci_lo': np.percentile(boot_values, 100 * alpha / 2),
        'ci_hi': np.percentile(boot_values, 100 * (1 - alpha / 2)),
    }


def compare_with_baselines(cate_bcauss: np.ndarray, y_test: np.ndarray, t_test: np.ndarray,
                           dr_summary_path: Path = None, policy_curve_path: Path = None,
                           n_bootstrap: int = 1000) -> pd.DataFrame:
    """
    Compare BCAUSS predictions with DR-learner and treat-all/treat-none baselines.
    """
    import json

    # Estimate propensity scores (simple)
    ps_test = np.full_like(y_test, fill_value=t_test.mean(), dtype=np.float32)

    results = []

    # BCAUSS policy value at threshold=0
    bcauss_boot = bootstrap_policy_value(cate_bcauss, y_test, t_test, ps_test,
                                         threshold=0.0, n_bootstrap=n_bootstrap)
    results.append({
        'method': 'BCAUSS (threshold=0)',
        'value_mean': bcauss_boot['mean'],
        'value_sd': bcauss_boot['sd'],
        'ci_lo': bcauss_boot['ci_lo'],
        'ci_hi': bcauss_boot['ci_hi'],
    })

    # Treat-all baseline
    treat_all_value = np.mean(t_test * y_test / ps_test)
    boot_treat_all = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
        val = np.mean(t_test[idx] * y_test[idx] / ps_test[idx])
        boot_treat_all.append(val)
    boot_treat_all = np.array(boot_treat_all)
    results.append({
        'method': 'Treat-All',
        'value_mean': np.mean(boot_treat_all),
        'value_sd': np.std(boot_treat_all),
        'ci_lo': np.percentile(boot_treat_all, 2.5),
        'ci_hi': np.percentile(boot_treat_all, 97.5),
    })

    # Treat-none baseline
    treat_none_value = np.mean((1 - t_test) * y_test / (1 - ps_test))
    boot_treat_none = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
        val = np.mean((1 - t_test[idx]) * y_test[idx] / (1 - ps_test[idx]))
        boot_treat_none.append(val)
    boot_treat_none = np.array(boot_treat_none)
    results.append({
        'method': 'Treat-None',
        'value_mean': np.mean(boot_treat_none),
        'value_sd': np.std(boot_treat_none),
        'ci_lo': np.percentile(boot_treat_none, 2.5),
        'ci_hi': np.percentile(boot_treat_none, 97.5),
    })

    # Load DR-learner results if available
    if dr_summary_path and dr_summary_path.exists():
        with open(dr_summary_path, 'r') as f:
            dr_summary = json.load(f)
        tau_hat = dr_summary.get('tau_hat_test', {})
        results.append({
            'method': 'DR-Learner (from file)',
            'value_mean': tau_hat.get('mean', np.nan),
            'value_sd': tau_hat.get('sd', np.nan),
            'ci_lo': tau_hat.get('p01', np.nan),
            'ci_hi': tau_hat.get('p99', np.nan),
        })

    # Load policy curve if available
    if policy_curve_path and policy_curve_path.exists():
        policy_df = pd.read_csv(policy_curve_path)
        # Get optimal threshold (max value_dr)
        best_row = policy_df.loc[policy_df['value_dr'].idxmax()]
        results.append({
            'method': f"DR-Learner (optimal threshold={best_row['threshold']:.3f})",
            'value_mean': best_row['value_boot_mean'],
            'value_sd': best_row['value_boot_sd'],
            'ci_lo': best_row['value_boot_ci_lo'],
            'ci_hi': best_row['value_boot_ci_hi'],
        })

    return pd.DataFrame(results)


def generate_bcauss_policy_curve(cate_estimates: np.ndarray, y: np.ndarray, t: np.ndarray,
                                 thresholds: np.ndarray = None, n_bootstrap: int = 500) -> pd.DataFrame:
    """Generate policy curve for BCAUSS similar to DR-learner format."""
    if thresholds is None:
        thresholds = np.linspace(cate_estimates.min(), cate_estimates.max(), 50)

    ps = np.full_like(y, fill_value=t.mean(), dtype=np.float32)
    rows = []

    for thresh in thresholds:
        treat_rate = np.mean(cate_estimates > thresh)
        boot = bootstrap_policy_value(cate_estimates, y, t, ps, thresh, n_bootstrap)

        # Compute treat-none and treat-all for deltas
        val_none = np.mean((1 - t) * y / (1 - ps))
        val_all = np.mean(t * y / ps)

        rows.append({
            'threshold': thresh,
            'treat_rate': treat_rate,
            'value_dr': boot['mean'],
            'delta_vs_none': boot['mean'] - val_none,
            'delta_vs_all': boot['mean'] - val_all,
            'value_boot_mean': boot['mean'],
            'value_boot_sd': boot['sd'],
            'value_boot_ci_lo': boot['ci_lo'],
            'value_boot_ci_hi': boot['ci_hi'],
        })

    return pd.DataFrame(rows)


# -------------------------

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", required=True, type=str, choices="".join(CONFIGS.keys()),
                            help="Dataset config to use")
    arg_parser.add_argument("--epochs", type=int, default=100)
    arg_parser.add_argument("--batch-size", type=int, default=32)
    arg_parser.add_argument("--patience", type=int, default=10)
    arg_parser.add_argument("--lr", type=float, default=None)

    args = arg_parser.parse_args()
    cfg = get_config(args.dataset)

    OUT_DIR = os.path.join(cfg.out_dir,"bcauss")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(cfg.data_path)

    df, num_cols, cat_cols = dataset_engineering_bcauss(df, id_col=cfg.id_col,
                                    subject_col=cfg.subject_col,
                                    treatment_col=cfg.treatment_col,
                                    outcome_col=cfg.outcome_col,
                                    drop_cols=cfg.drop_cols)

    feat_cols = num_cols + cat_cols

    drop_also = ["intime", "t0_time", "hadm_id", "icustay_id", "subject_id", "stay_id"]
    feat_cols = [c for c in feat_cols if c not in drop_also]

    splits = split_by_subject(
        df,
        subject_col=cfg.subject_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    tr_idx = splits["train"]
    te_idx = splits["test"]
    val_idx = splits.get("val", None)

    X = df[feat_cols]
    T = df[cfg.treatment_col].astype(int).values
    Y = df[cfg.outcome_col].astype(int).values

    print(f"Data shape after engineering: {X.shape}")

    with pd.option_context('display.max_columns', None):
        print("Sample of processed data:")
        print(X.head(3))

    X_tr, T_tr, Y_tr = X.iloc[tr_idx], T[tr_idx], Y[tr_idx]
    X_te, T_te, Y_te = X.iloc[te_idx], T[te_idx], Y[te_idx]

    X_val = T_val = Y_val = None
    if val_idx is not None:
        X_val, T_val, Y_val = X.iloc[val_idx], T[val_idx], Y[val_idx]

    print(f"Training samples: {X_tr.shape[0]} \n "
          f"Validation samples: {X_val.shape[0] if X_val is not None else 0} \n "
          f"Test samples: {X_te.shape[0]}")
    checkpoint_file = os.path.join(OUT_DIR, "bcauss_best.h5")
    trained_model = train_bcauss(
        X=X_tr,
        t=T_tr,
        y=Y_tr,
        X_val=X_val,
        t_val=T_val,
        y_val=Y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        checkpoint_path=Path(checkpoint_file),
    )
    print("BCAUSS model trained.")

    print("\nPredicting CATE on test set...")
    cate_estimates = predict_cate_bcauss(trained_model, X_te)

    # Save CATE estimates with id and treatment for correct grouping
    cate_df = pd.DataFrame({
        cfg.id_col: df.iloc[te_idx][cfg.id_col].values,
        'treatment': T_te,
        'cate_estimate': cate_estimates.flatten()
    })

    parquet_path = Path(OUT_DIR) / "cate_estimates.parquet"
    cate_df.to_parquet(parquet_path, index=False)
    print(f"CATE estimates saved to {parquet_path}")

    # Print summary statistics of CATE estimates
    print("CATE Estimates Summary:")
    print(cate_df['cate_estimate'].describe())

    # Divide cate per treatment group correctly
    print("\nCATE Estimates Summary by Treatment Group (0=Control, 1=Treated) :")
    for group in [0, 1]:
        group_df = cate_df[cate_df['treatment'] == group]
        print(f"\nCATE Estimates Summary Group {group}:")
        print(group_df['cate_estimate'].describe())
        print("\nOther Stats:")
        print(f"Mean CATE: {group_df['cate_estimate'].mean()}")
        print(f"Std CATE: {group_df['cate_estimate'].std()}")
        print(f"Min CATE: {group_df['cate_estimate'].min()}")
        print(f"Max CATE: {group_df['cate_estimate'].max()}")

    print("\n Starting Policy Evaluation...")

    # python
    # -------------------------
    # Compare with DR-learner and baselines
    # -------------------------
    print("\n" + "=" * 60)
    print("Comparing BCAUSS with DR-Learner and Baselines")
    print("=" * 60)

    # Paths to DR-learner artifacts (adjust based on your config)
    dr_summary_path = Path(cfg.out_dir) / "dr_summary.json"
    policy_curve_path = Path(cfg.out_dir) / "policy_threshold_curve.csv"

    comparison_df = compare_with_baselines(
        cate_bcauss=cate_estimates,
        y_test=Y_te,
        t_test=T_te,
        dr_summary_path=dr_summary_path,
        policy_curve_path=policy_curve_path,
        n_bootstrap=100
    )

    print("\nPolicy Value Comparison (with 95% CI):")
    print(comparison_df.to_string(index=False))

    # Save comparison results
    comparison_path = Path(OUT_DIR) / "comparison_with_dr_learner.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to {comparison_path}")

    # Generate BCAUSS policy curve (similar format to DR-learner)
    print("\nGenerating BCAUSS policy threshold curve...")
    bcauss_policy_curve = generate_bcauss_policy_curve(
        cate_estimates=cate_estimates,
        y=Y_te,
        t=T_te,
        n_bootstrap=50
    )

    policy_curve_out = Path(OUT_DIR) / "bcauss_policy_threshold_curve.csv"
    bcauss_policy_curve.to_csv(policy_curve_out, index=False)
    print(f"BCAUSS policy curve saved to {policy_curve_out}")

    # Print optimal threshold
    best_idx = bcauss_policy_curve['value_dr'].idxmax()
    best_row = bcauss_policy_curve.iloc[best_idx]
    print(f"\nOptimal BCAUSS threshold: {best_row['threshold']:.4f}")
    print(f"  Treat rate: {best_row['treat_rate']:.2%}")
    print(
        f"  Policy value: {best_row['value_dr']:.4f} [{best_row['value_boot_ci_lo']:.4f}, {best_row['value_boot_ci_hi']:.4f}]")
