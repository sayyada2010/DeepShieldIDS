
from __future__ import annotations
import numpy as np
import pandas as pd


def apply_smote(X_train: pd.DataFrame, y_train, random_state: int = 42):
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as exc:
        raise ImportError("Install imbalanced-learn to use SMOTE: pip install imbalanced-learn") from exc
    unique, counts = np.unique(y_train, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        return X_train, y_train
    k_neighbors = int(max(1, min(5, counts.min() - 1)))
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), y_res
