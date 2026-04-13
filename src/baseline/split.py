"""
Module for splitting a DataFrame into train/val/test sets by unique subject IDs.
Ensures that no subject appears in multiple splits.
Split  per subject_id:
- train: 70%
- val: 15%
- test: 15%
"""

import numpy as np
import pandas as pd

def split_by_subject(
    df: pd.DataFrame,
    subject_col: str,
    test_size: float,
    val_size: float,
    random_state: int
) -> dict[str, np.ndarray]:
    """
    Returns indices for train/val/test splits by unique subject_id.
    No subject appears in multiple splits.
    """
    rng = np.random.default_rng(random_state)
    subjects = df[subject_col].dropna().unique()
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val

    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:n_train + n_val])
    test_subj = set(subjects[n_train + n_val:])

    subj = df[subject_col].values
    train_idx = np.where(np.isin(subj, list(train_subj)))[0]
    val_idx = np.where(np.isin(subj, list(val_subj)))[0]
    test_idx = np.where(np.isin(subj, list(test_subj)))[0]

    # sanity checks
    assert len(set(df.iloc[train_idx][subject_col]).intersection(set(df.iloc[val_idx][subject_col]))) == 0
    assert len(set(df.iloc[train_idx][subject_col]).intersection(set(df.iloc[test_idx][subject_col]))) == 0
    assert len(set(df.iloc[val_idx][subject_col]).intersection(set(df.iloc[test_idx][subject_col]))) == 0

    return {"train": train_idx, "val": val_idx, "test": test_idx}
