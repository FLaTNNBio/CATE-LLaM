"""
Baseline model pipelines and utilities.
Pipeline sklearn:

- ColumnTransformer for preprocessing:
    - Numeric features: passthrough (keep as is, including NaN)
    - Categorical features: OneHotEncoder(handle_unknown="ignore")

- HistGradientBoostingClassifier as the model:
    - Works well on tabular clinical data
    - Handles non-linearities and interactions well
    - Crucially, tolerates NaN in numeric features

HistGradientBoostingClassifier is a modern gradient boosting implementation in sklearn
that natively supports missing values in numeric features,
making it suitable for clinical datasets where missingness is common.
It is similar in spirit to XGBoost or LightGBM but is part of sklearn,
simplifying integration into sklearn pipelines.

"""
from dataclasses import dataclass
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

@dataclass
class HGBConfig:
    max_depth: int = 3
    max_iter: int = 400
    learning_rate: float = 0.05
    min_samples_leaf: int = 50
    l2_regularization: float = 0.0
    random_state: int = 42

def make_hgb_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    cfg: HGBConfig
) -> Pipeline:
    """
    Pipeline that:
      - leaves numeric as-is (NaN allowed)
      - one-hot encodes categoricals (missing treated as its own category)
      - fits HGBClassifier
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = HistGradientBoostingClassifier(
        max_depth=cfg.max_depth,
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        random_state=cfg.random_state,
    )

    return Pipeline([("pre", pre), ("clf", clf)])

def clip_ps(ps: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(ps, lo, hi)
