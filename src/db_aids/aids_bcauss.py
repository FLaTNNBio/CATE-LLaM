import argparse
import json
import os
from pathlib import Path

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
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / (denom + 1e-12)

    # Final numeric safety net: remove +/-inf and NaN from covariates
    feature_cols = [c for c in df.columns if c not in {id_col, subject_col, treatment_col, outcome_col}]
    if feature_cols:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    # Ensure treatment/outcome are clean binary
    df[treatment_col] = pd.to_numeric(df[treatment_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(
        0).astype(np.float32)
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(
        0).astype(np.float32)

    return df, num_cols, new_cat_cols


def stable_bcauss_loss(y_true_concat, y_pred_concat, alpha=1.0, beta=1.0):
    y_true = y_true_concat[:, 0:1]
    t_true = y_true_concat[:, 1:2]

    y0_pred = y_pred_concat[:, 0:1]
    y1_pred = y_pred_concat[:, 1:2]
    t_pred = y_pred_concat[:, 2:3]

    t_pred = K.clip(t_pred, 1e-6, 1.0 - 1e-6)
    y0_pred = K.clip(y0_pred, -1e6, 1e6)
    y1_pred = K.clip(y1_pred, -1e6, 1e6)

    bce = -(t_true * K.log(t_pred) + (1.0 - t_true) * K.log(1.0 - t_pred))
    treatment_loss = K.mean(bce)

    # Outcome binario: BCE factual (molto piu stabile di MSE su y normalizzata)
    y0_prob = K.clip(K.sigmoid(y0_pred), 1e-6, 1.0 - 1e-6)
    y1_prob = K.clip(K.sigmoid(y1_pred), 1e-6, 1.0 - 1e-6)

    bce0 = -(y_true * K.log(y0_prob) + (1.0 - y_true) * K.log(1.0 - y0_prob))
    bce1 = -(y_true * K.log(y1_prob) + (1.0 - y_true) * K.log(1.0 - y1_prob))
    regression_loss = K.mean((1.0 - t_true) * bce0 + t_true * bce1)

    return alpha * regression_loss + beta * treatment_loss


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
    X_np = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
    t_np = np.asarray(t, dtype=np.float32).reshape(-1, 1)
    y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    t_np = np.nan_to_num(t_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    # enforce binary range for treatment/outcome
    t_np = np.clip(t_np, 0.0, 1.0)
    y_np = np.clip(y_np, 0.0, 1.0)

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

        y_val_np_norm = y_val_np

    model = make_bcauss(input_dim=X_np.shape[1], use_bce=True)
    opt = Adam(learning_rate=lr, clipnorm=1.0, clipvalue=0.25)

    def bcauss_loss_fn(y_true, y_pred):
        return stable_bcauss_loss(y_true, y_pred, alpha=1.0, beta=1.0)

    model.compile(optimizer=opt, loss=bcauss_loss_fn)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-7),
    ]
    if checkpoint_path is not None:
        callbacks.insert(
            0,
            keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor="val_loss", mode="min"),
        )

    train_inputs = [X_np, y_np_norm, t_np]
    train_target = np.concatenate([y_np_norm, t_np], axis=1)

    if X_val_np is not None:
        val_inputs = [X_val_np, y_val_np_norm, t_val_np]
        val_target = np.concatenate([y_val_np_norm, t_val_np], axis=1)
        model.fit(
            train_inputs,
            train_target,
            validation_data=(val_inputs, val_target),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )
    else:
        model.fit(
            train_inputs,
            train_target,
            validation_split=0.2,
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
