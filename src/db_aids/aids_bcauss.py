import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from src.baseline import default_feature_columns
from src.baseline.features import coerce_numeric_columns
from src.baseline.split import split_by_subject
from src.config import get_config
from src.models.bcauss.models import make_bcauss


def dataset_engineering_bcauss(
    df: pd.DataFrame,
    id_col: str,
    subject_col: str,
    treatment_col: str,
    outcome_col: str,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    drop_cols = drop_cols or []

    num_cols, cat_cols, _ = default_feature_columns(
        df,
        id_col=id_col,
        subject_col=subject_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        drop_cols=drop_cols,
    )

    # Enforce numeric where expected
    df[num_cols] = coerce_numeric_columns(df, num_cols)

    # Force binary numeric treatment/outcome
    df[treatment_col] = pd.to_numeric(df[treatment_col], errors="coerce").fillna(0).astype(int)
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0).astype(int)

    # One-hot encode non-protected categorical columns
    protected = {id_col, subject_col, treatment_col, outcome_col}
    cat_to_encode = [c for c in cat_cols if c not in protected and c not in drop_cols]
    if cat_to_encode:
        df = pd.get_dummies(df, columns=cat_to_encode, drop_first=True)

    # Recompute categorical cols after OHE
    new_cat_cols = [c for c in df.columns if c not in set(num_cols) | protected | set(drop_cols)]
    for c in new_cat_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.float32)
    if new_cat_cols:
        df[new_cat_cols] = df[new_cat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    # Group-wise imputation for numeric covariates (by treatment/outcome)
    for col in num_cols:
        if col in drop_cols:
            continue
        if df[col].isna().any():
            for t_val in df[treatment_col].dropna().unique():
                for y_val in df[outcome_col].dropna().unique():
                    mask = (df[treatment_col] == t_val) & (df[outcome_col] == y_val)
                    na_mask = mask & df[col].isna()
                    if not na_mask.any():
                        continue
                    mean_val = df.loc[mask, col].mean()
                    if pd.isna(mean_val):
                        mean_val = df[col].mean()
                    if pd.isna(mean_val):
                        mean_val = 0.0
                    noise = np.random.normal(0.0, 1e-3, size=int(na_mask.sum()))
                    df.loc[na_mask, col] = mean_val + noise

    # Min-max scale numeric covariates
    for col in num_cols:
        if col in drop_cols:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        denom = (col_max - col_min)
        if pd.isna(denom) or abs(float(denom)) < 1e-12:
            # Constant column: just set to 0
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / (denom + 1e-12)

    # Final numeric safety net: remove +/-inf and NaN from covariates
    feature_cols = [c for c in df.columns if c not in {id_col, subject_col, treatment_col, outcome_col}]
    if feature_cols:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    # Ensure treatment/outcome are clean binary (force Series ops to avoid type issues)
    df[treatment_col] = (
        pd.Series(df[treatment_col])
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    df[outcome_col] = (
        pd.Series(df[outcome_col])
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )

    return df, num_cols, new_cat_cols


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

    # Ensure final loss is finite
    total_loss = alpha * regression_loss + beta * treatment_loss
    total_loss = K.clip(total_loss, -1e3, 1e3)

    return total_loss


def train_bcauss(
    X,
    t,
    y,
    X_val=None,
    t_val=None,
    y_val=None,
    epochs: int = 120,
    batch_size: int = 64,
    patience: int = 12,
    lr: float = 1e-4,
    checkpoint_path: Path | None = None,
) -> keras.Model:
    """
    Train BCAUSS model with fail-fast checks for non-finite data.
    """
    def _assert_finite(name: str, arr: np.ndarray) -> None:
        """Raise ValueError if array contains NaN or Inf."""
        if arr.size == 0:
            raise ValueError(f"{name} is empty; cannot train/evaluate.")
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise ValueError(f"{name} contains non-finite values: NaN={n_nan}, Inf={n_inf}")

    X_np = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
    t_np = np.asarray(t, dtype=np.float32).reshape(-1, 1)
    y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    t_np = np.nan_to_num(t_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    # enforce binary range for treatment/outcome
    t_np = np.clip(t_np, 0.0, 1.0)
    y_np = np.clip(y_np, 0.0, 1.0)

    # Fail-fast: ensure train data is finite before normalization
    _assert_finite("X_train", X_np)
    _assert_finite("t_train", t_np)
    _assert_finite("y_train", y_np)

    # Binary outcome: keep in {0,1}
    y_np_norm = y_np

    X_val_np = t_val_np = y_val_np_norm = None
    if X_val is not None and t_val is not None and y_val is not None:
        X_val_np = X_val.to_numpy(dtype=np.float32) if isinstance(X_val, pd.DataFrame) else np.asarray(X_val,
                                                                                                       dtype=np.float32)
        t_val_np = np.asarray(t_val, dtype=np.float32).reshape(-1, 1)
        y_val_np = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

        # Safety net anche sulla validation (qui ti nasceva il NaN)
        X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        t_val_np = np.nan_to_num(t_val_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        y_val_np = np.nan_to_num(y_val_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        t_val_np = np.clip(t_val_np, 0.0, 1.0)
        y_val_np = np.clip(y_val_np, 0.0, 1.0)

        # Fail-fast: ensure val data is finite
        _assert_finite("X_val", X_val_np)
        _assert_finite("t_val", t_val_np)
        _assert_finite("y_val", y_val_np)

        y_val_np_norm = y_val_np

    model = make_bcauss(input_dim=X_np.shape[1], use_bce=True)
    opt = Adam(learning_rate=lr, clipnorm=0.5, clipvalue=0.1)  # Aggressive gradient clipping

    def bcauss_loss_fn(y_true, y_pred):
        return stable_bcauss_loss(y_true, y_pred, alpha=1.0, beta=1.0)

    model.compile(optimizer=opt, loss=bcauss_loss_fn)

    # Custom callback to detect NaN in val_loss immediately
    class NaNCheckCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get('val_loss')
            if val_loss is not None and np.isnan(val_loss):
                print(f"\n[WARNING] Epoch {epoch + 1}: val_loss is NaN! train_loss={logs.get('loss')}")

    callbacks = [
        NaNCheckCallback(),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-7),
    ]
    if checkpoint_path is not None:
        callbacks.insert(
            1,
            keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor="val_loss", mode="min"),
        )

    train_inputs = [X_np, y_np_norm, t_np]
    train_target = np.concatenate([y_np_norm, t_np], axis=1)

    # Use validation_split: Keras handles internal splitting more robustly than explicit validation_data
    # This avoids NaN logging issues with custom multi-output loss functions
    model.fit(
        train_inputs,
        train_target,
        validation_split=0.2,  # Use 20% for validation, same proportion as val_size=0.15
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    return model


def predict_cate_bcauss(model: keras.Model, X) -> np.ndarray:
    X_np = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
    n = X_np.shape[0]
    dummy_y = np.zeros((n, 1), dtype=np.float32)
    dummy_t = np.zeros((n, 1), dtype=np.float32)

    preds = model.predict([X_np, dummy_y, dummy_t], verbose=0)
    if preds.ndim != 2 or preds.shape[1] < 2:
        raise ValueError("Unexpected prediction shape from BCAUSS model.")
    y0_pred = preds[:, 0]
    y1_pred = preds[:, 1]
    return (y1_pred - y0_pred).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aids_v1", choices=["aids_v1"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    cfg = get_config(args.dataset)
    out_dir = Path(cfg.out_dir) / "bcauss"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.data_path)

    # Strict anti-leakage: drop leakage cols but keep mandatory pipeline columns
    protected_cols = {cfg.id_col, cfg.subject_col, cfg.treatment_col, cfg.outcome_col}
    forced_drop = set(cfg.drop_cols or [])
    effective_drop = sorted([c for c in forced_drop if c not in protected_cols])

    present_forced_drop = [c for c in effective_drop if c in df.columns]
    if present_forced_drop:
        df = df.drop(columns=present_forced_drop)

    df, num_cols, cat_cols = dataset_engineering_bcauss(
        df=df,
        id_col=cfg.id_col,
        subject_col=cfg.subject_col,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        drop_cols=effective_drop,
    )

    feat_cols = [c for c in (num_cols + cat_cols) if c not in set(effective_drop)]
    feat_cols = [c for c in feat_cols if c not in protected_cols and c in df.columns]

    splits = split_by_subject(
        df=df,
        subject_col=cfg.subject_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    tr_idx = splits["train"]
    te_idx = splits["test"]
    val_idx = splits.get("val")

    X = df[feat_cols]
    T = df[cfg.treatment_col].astype(int).values
    Y = df[cfg.outcome_col].astype(int).values  # 1=failure, 0=censoring

    X_tr, T_tr, Y_tr = X.iloc[tr_idx], T[tr_idx], Y[tr_idx]
    X_te, T_te, Y_te = X.iloc[te_idx], T[te_idx], Y[te_idx]

    X_val = T_val = Y_val = None
    if val_idx is not None:
        X_val, T_val, Y_val = X.iloc[val_idx], T[val_idx], Y[val_idx]

    checkpoint_file = out_dir / "bcauss_best.h5"
    model = train_bcauss(
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
        checkpoint_path=checkpoint_file,
    )

    cate_test = predict_cate_bcauss(model, X_te)

    cate_df = pd.DataFrame(
        {
            cfg.id_col: df.iloc[te_idx][cfg.id_col].values,
            "treatment": T_te,
            "outcome_failure": Y_te,
            "cate_estimate": cate_test,
        }
    )
    cate_df.to_parquet(out_dir / "cate_estimates.parquet", index=False)

    summary = {
        "dataset": args.dataset,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx) if val_idx is not None else 0),
        "n_test": int(len(te_idx)),
        "outcome_semantics": "1=failure, 0=censoring",
        "cate_test_mean": float(np.mean(cate_test)),
        "cate_test_std": float(np.std(cate_test)),
        "cate_test_p05": float(np.percentile(cate_test, 5)),
        "cate_test_p50": float(np.percentile(cate_test, 50)),
        "cate_test_p95": float(np.percentile(cate_test, 95)),
        "dropped_columns_for_leakage": effective_drop,
        "used_feature_count": int(len(feat_cols)),
    }
    with open(out_dir / "bcauss_summary.json", "w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved: {out_dir / 'cate_estimates.parquet'}")
    print(f"[OK] Saved: {out_dir / 'bcauss_summary.json'}")
    print(f"[INFO] Features used: {len(feat_cols)}")


if __name__ == "__main__":
    main()
