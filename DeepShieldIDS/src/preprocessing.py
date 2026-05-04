
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from .config import LABEL_CANDIDATES, IRRELEVANT_COLUMN_HINTS, TARGET_ATTACK_LABELS, RANDOM_STATE


@dataclass
class PreparedSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_encoder: Optional[LabelEncoder]
    scaler: MinMaxScaler
    feature_names: list[str]


def infer_label_column(df: pd.DataFrame, label_col: Optional[str] = None) -> str:
    if label_col and label_col in df.columns:
        return label_col
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer label column. Available columns: {list(df.columns)[:30]}...")


def remove_irrelevant_columns(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        if col == label_col:
            continue
        normalized = col.strip().lower()
        if any(hint in normalized for hint in IRRELEVANT_COLUMN_HINTS):
            drop_cols.append(col)
    return df.drop(columns=list(set(drop_cols)), errors="ignore")


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.replace([np.inf, -np.inf, "Infinity", "inf", "NaN", "nan", ""], np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df.dropna(axis=0, how="all")


def make_labels(y: pd.Series, binary: bool = True) -> tuple[np.ndarray, Optional[LabelEncoder]]:
    y = y.astype(str).str.strip()
    if binary:
        normal_set = {str(v).strip().lower() for v in TARGET_ATTACK_LABELS}
        labels = np.array([0 if str(v).strip().lower() in normal_set else 1 for v in y], dtype=np.int64)
        return labels, None
    le = LabelEncoder()
    return le.fit_transform(y), le


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(df.median(numeric_only=True)).fillna(0)
    return df.astype("float32")


def prepare_single_dataset(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    binary: bool = True,
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = RANDOM_STATE,
) -> PreparedSplits:
    df = clean_values(df)
    label_col = infer_label_column(df, label_col)
    df = remove_irrelevant_columns(df, label_col)
    y_raw = df[label_col]
    X_raw = df.drop(columns=[label_col])
    y, le = make_labels(y_raw, binary=binary)
    X = encode_features(X_raw)

    stratify = y if len(np.unique(y)) > 1 else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    val_fraction = val_size / (1.0 - test_size)
    stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=stratify_temp
    )
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return PreparedSplits(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, le, scaler, list(X.columns)
    )


def align_columns(train_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    return other_df.reindex(columns=train_df.columns, fill_value=0)
