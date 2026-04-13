# python
import argparse
import os, json
from bdb import effective

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
    """
    Engineer dataset for BCAUSS training with aggressive data cleaning.
    
    Key steps:
    1. Identify and coerce numeric columns
    2. Convert treatment/outcome to numeric
    3. One-hot encode categorical columns
    4. Impute missing values with stratified means
    5. Min-max scale with inf guard
    6. Final comprehensive NaN/Inf cleanup
    """
    drop_cols = drop_cols or []
    
    # Step 0: Make a copy to avoid modif original (?)
    df = df.copy()

    num_cols, cat_cols, dropped = default_feature_columns(
        df,
        id_col=id_col,
        subject_col=subject_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        drop_cols=drop_cols,
    )
    
    # Step 1: Coerce numeric columns to float32, keep non-convertible as-is for later processing
    df[num_cols] = coerce_numeric_columns(df, num_cols)
    
    # Step 2: Convert treatment and outcome to numeric FIRST (before any operations)
    # This is critical: convert to numeric, replace inf with nan, fill nan with 0/1 default
    df[treatment_col] = pd.to_numeric(df[treatment_col], errors="coerce")
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    
    # Replace inf with NaN temporarily for treatment/outcome
    df[treatment_col] = df[treatment_col].replace([np.inf, -np.inf], np.nan)
    df[outcome_col] = df[outcome_col].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN in treatment/outcome: use mode/0 as fallback
    for col in [treatment_col, outcome_col]:
        if df[col].isna().any():
            fill_value = df[col].mode()
            if len(fill_value) > 0:
                df[col] = df[col].fillna(fill_value[0])
            else:
                df[col] = df[col].fillna(0)
    
    # Ensure treatment/outcome are int
    df[treatment_col] = df[treatment_col].astype(int)
    df[outcome_col] = df[outcome_col].astype(int)
    
    # Step 3: Handle categorical columns - convert to numeric first, then OHE
    # This prevents OHE from failing on weird values
    cat_to_encode = [
        c for c in cat_cols 
        if c not in {id_col, subject_col, treatment_col, outcome_col}
    ]

    #
    if cat_to_encode:
        # Convert categorical columns to string, then apply mapping if available
        for col in cat_to_encode:
            # Try mapping (e.g., HFNC=1, NIV=0)
            if col == treatment_col:
                mapping = {"HFNC": 1, "NIV": 0}
                df[col] = df[col].map(mapping).fillna(df[col])
            else:
                # For other categoricals, just convert to string then one-hot
                df[col] = df[col].astype(str)
        
        # One-hot encode
        df = pd.get_dummies(df, columns=cat_to_encode, drop_first=True, dummy_na=False)
    
    # Step 4: Identify new columns after OHE
    new_cat_cols = [
        col for col in df.columns 
        if col not in num_cols + [id_col, subject_col, treatment_col, outcome_col]
    ]
    
    # Ensure new categorical columns are numeric and finite
    if new_cat_cols:
        for col in new_cat_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)
    
    cat_cols = new_cat_cols
    
    # Step 5: Impute missing values in numeric columns
    # Use stratified mean by (treatment, outcome) group
    for col in num_cols:
        if df[col].isna().any():
            # First pass: fill with stratified group mean
            for t_val in sorted(df[treatment_col].dropna().unique()):
                for y_val in sorted(df[outcome_col].dropna().unique()):
                    mask = (df[treatment_col] == t_val) & (df[outcome_col] == y_val)
                    na_mask = mask & df[col].isna()
                    
                    if not na_mask.any():
                        continue
                    
                    # Get group mean
                    group_mean = df.loc[mask, col].mean()
                    
                    # If group mean is NaN (all NaN in group), use global mean
                    if pd.isna(group_mean):
                        group_mean = df[col].mean()
                    
                    # If still NaN (entire column is NaN), use 0
                    if pd.isna(group_mean):
                        group_mean = 0.0
                    
                    # Fill with mean + small noise
                    noise = np.random.normal(0, 1e-4, size=int(na_mask.sum()))
                    df.loc[na_mask, col] = group_mean + noise
            
            # Second pass: fill any remaining NaN with global mean or 0
            remaining_na = df[col].isna().sum()
            if remaining_na > 0:
                fill_val = df[col].mean()
                fill_val = fill_val if not pd.isna(fill_val) else 0.0
                df[col] = df[col].fillna(fill_val)
    
    # Step 6: Replace inf/nan with 0 in all numeric columns (safety net)
    for col in num_cols + cat_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0.0)
    
    # Step 7: Min-max scale numeric columns to [0,1]
    # CRITICAL: Check for inf BEFORE scaling
    for col in num_cols:
        if col not in df.columns:
            continue
        
        # Ensure no inf before scaling
        df[col] = df[col].replace([np.inf, -np.inf], 0.0)
        
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Safety: if min/max are inf or NaN, set column to 0
        if pd.isna(col_min) or pd.isna(col_max) or np.isinf(col_min) or np.isinf(col_max):
            df[col] = 0.0
            continue
        
        denom = col_max - col_min
        
        # Guard for constant columns
        if abs(float(denom)) < 1e-12:
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / (denom + 1e-12)
            # Clip to [0, 1] in case of floating point errors
            df[col] = np.clip(df[col], 0.0, 1.0)
    
    # Step 8: Final comprehensive cleanup - remove ANY inf/nan from features
    feature_cols = [c for c in num_cols + cat_cols if c in df.columns]
    
    for col in feature_cols:
        # Replace inf first
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN
        df[col] = df[col].fillna(0.0)
        # Convert to float32
        df[col] = df[col].astype(np.float32)
        
        # Final check: if still any non-finite, set to 0
        non_finite = ~np.isfinite(df[col].values)
        if non_finite.any():
            df.loc[non_finite, col] = 0.0
    
    # Step 9: Ensure treatment/outcome are clean binary and finite
    df[treatment_col] = df[treatment_col].astype(int)
    df[outcome_col] = df[outcome_col].astype(int)
    
    # Step 10: CRITICAL - drop any columns that are not: features, id, subject, treatment, outcome
    # This prevents leakage columns from being passed to the model
    valid_cols = set(feature_cols + [id_col, subject_col, treatment_col, outcome_col])
    cols_to_drop = [c for c in df.columns if c not in valid_cols]
    if cols_to_drop:
        print(f"[INFO] Dropping {len(cols_to_drop)} non-feature columns: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
        df = df.drop(columns=cols_to_drop)
    
    # Final validation: ensure NO inf/nan anywhere
    all_feature_cols = feature_cols + [treatment_col, outcome_col]
    for col in all_feature_cols:
        if col in df.columns:
            assert np.isfinite(df[col]).all(), f"Column {col} still has non-finite values!"
    
    print(f"[INFO] Dataset engineering complete: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"[INFO] Feature columns: {len(feature_cols)} (numeric: {len(num_cols)}, categorical: {len(cat_cols)})")
    
    return df, num_cols, cat_cols

# python
from tensorflow.keras import backend as K

def stable_bcauss_loss(y_true_concat, y_pred_concat, alpha=1.0, beta=1.0):
    """
    Stable BCAUSS loss using Keras native BinaryCrossentropy for numerical stability.
    
    y_true_concat: [y_true, t_true] concatenated
    y_pred_concat: [y0_pred, y1_pred, t_pred] (logits, not probabilities)
    """
    from tensorflow.keras.losses import BinaryCrossentropy
    
    y_true = y_true_concat[:, 0:1]
    t_true = y_true_concat[:, 1:2]

    y0_pred = y_pred_concat[:, 0:1]
    y1_pred = y_pred_concat[:, 1:2]
    t_pred = y_pred_concat[:, 2:3]

    # Use native Keras BinaryCrossentropy with from_logits=True
    # This handles numerical stability internally
    bce_fn = BinaryCrossentropy(from_logits=True, reduction='none')

    # Treatment loss: BCE from logits
    bce_t = bce_fn(t_true, t_pred)
    # Clip BCE values to prevent infinities
    bce_t = K.clip(bce_t, -1e3, 1e3)
    treatment_loss = K.mean(bce_t)

    # Outcome loss: BCE from logits (factual, not counterfactual)
    bce0 = bce_fn(y_true, y0_pred)
    bce1 = bce_fn(y_true, y1_pred)
    
    # Clip BCE values to prevent infinities
    bce0 = K.clip(bce0, -1e3, 1e3)
    bce1 = K.clip(bce1, -1e3, 1e3)

    # Weigh by observed treatment with epsilon to prevent NaN when batch is unbalanced
    # (i.e., all t=0 or all t=1 would make weight exactly 0, causing NaN in gradient)
    eps = 1e-6
    weight0 = K.maximum(1.0 - t_true, eps)  # Ensure min weight is eps, not 0
    weight1 = K.maximum(t_true, eps)         # Ensure min weight is eps, not 0
    
    # Normalize weights so they stay reasonable scale
    norm = weight0 + weight1 + eps
    weight0 = weight0 / norm
    weight1 = weight1 / norm
    
    regression_loss = K.mean(weight0 * bce0 + weight1 * bce1)

    total_loss = alpha * regression_loss + beta * treatment_loss
    # Ensure final loss is finite
    total_loss = K.clip(total_loss, -1e3, 1e3)
    
    return total_loss


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
    lr: float = 5e-5,
    checkpoint_path: Path = None,
) -> keras.Model:
    # --- coerce types to numeric numpy arrays ---
    if isinstance(X, pd.DataFrame):
        X_np = X.to_numpy(dtype=np.float32)
    else:
        X_np = np.asarray(X, dtype=np.float32)

    t_np = np.asarray(t, dtype=np.float32).reshape(-1, 1)
    y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    # Keep tensors finite before normalization/loss computation
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    t_np = np.nan_to_num(t_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    t_np = np.clip(t_np, 0.0, 1.0)
    y_np = np.clip(y_np, 0.0, 1.0)

    # Normalize y to prevent large gradients
    y_mean = np.mean(y_np)
    y_std = np.std(y_np) + 1e-8
    y_np_norm = (y_np - y_mean) / y_std

    X_val_np = t_val_np = y_val_np = y_val_np_norm = None
    if X_val is not None and t_val is not None and y_val is not None:
        if isinstance(X_val, pd.DataFrame):
            X_val_np = X_val.to_numpy(dtype=np.float32)
        else:
            X_val_np = np.asarray(X_val, dtype=np.float32)
        t_val_np = np.asarray(t_val, dtype=np.float32).reshape(-1, 1)
        y_val_np = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

        X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        t_val_np = np.nan_to_num(t_val_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        y_val_np = np.nan_to_num(y_val_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        t_val_np = np.clip(t_val_np, 0.0, 1.0)
        y_val_np = np.clip(y_val_np, 0.0, 1.0)

        y_val_np_norm = (y_val_np - y_mean) / y_std

    input_dim = X_np.shape[1]
    model = make_bcauss(input_dim=input_dim, use_bce=True)

    # Use lower learning rate with aggressive gradient clipping
    opt = Adam(learning_rate=lr if lr else 5e-5, clipnorm=0.5, clipvalue=0.1)
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

    # Use validation_split: Keras handles internal splitting more robustly than explicit validation_data
    # This avoids NaN logging issues with custom multi-output loss functions
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
    arg_parser.add_argument("--dataset", required=True, type=str, choices=list(CONFIGS.keys()),
                            help="Dataset config to use")
    arg_parser.add_argument("--epochs", type=int, default=100)
    arg_parser.add_argument("--batch-size", type=int, default=32)
    arg_parser.add_argument("--patience", type=int, default=10)
    arg_parser.add_argument("--lr", type=float, default=None)
    arg_parser.add_argument("--save-plots", type=bool, default=True)

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

    drop_also = ["intime", "t0_time", "hadm_id", "icustay_id", "subject_id", "stay_id", "first_diur_time"]
    also_dropped = [c for c in feat_cols if c in drop_also]
    feat_cols = [c for c in feat_cols if c not in drop_also]
    feat_cols = [c for c in feat_cols if c not in cfg.drop_cols]

    # Safety: ensure feat_cols only contains columns that actually exist in df after engineering
    feat_cols = [c for c in feat_cols if c in df.columns]

    effective_drop = cfg.drop_cols + also_dropped

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
        print("\n ==== X Info After data engineering ====")
        print(X.info())
        print("\n ==== End X Info After data engineering ====")


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

    summary = {
        "dataset": args.dataset,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx) if val_idx is not None else 0),
        "n_test": int(len(te_idx)),
        "outcome_semantics": "1=failure, 0=censoring",
        "cate_test_mean": float(np.mean(cate_estimates)),
        "cate_test_std": float(np.std(cate_estimates)),
        "cate_test_p05": float(np.percentile(cate_estimates, 5)),
        "cate_test_p50": float(np.percentile(cate_estimates, 50)),
        "cate_test_p95": float(np.percentile(cate_estimates, 95)),
        "dropped_columns_for_leakage": effective_drop,
        "used_feature_count": int(len(feat_cols)),
    }
    out_dir = Path(OUT_DIR)
    with open(out_dir / "bcauss_summary.json", "w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved: {out_dir / 'cate_estimates.parquet'}")
    print(f"[OK] Saved: {out_dir / 'bcauss_summary.json'}")
    print(f"[INFO] Features used: {len(feat_cols)}")

    if(args.save_plots):
        # Run main in plot_bcauss to save plots
        # run as module to get all arguments parsed correctly:
        os.system(f"python -m src.cate.scripts.plot_bcauss --dataset {args.dataset}")