from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def evaluate_assignment_strength(
    X_scaled: pd.DataFrame,
    treatment: np.ndarray,
) -> float:
    """
    Evaluate how strongly X predicts the new treatment assignment.

    This is done by fitting a logistic regression model to predict the treatment from X_scaled and computing the ROC AUC:
    LogisticRegression(max_iter=5000) + ROC AUC.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        Standardized covariates.
    treatment : np.ndarray
        Sampled binary treatment assignment.

    Returns
    -------
    float
        AUC for predicting treatment from X.
    """
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_scaled, treatment)
    pred = clf.predict_proba(X_scaled)[:, 1]
    return roc_auc_score(treatment, pred)