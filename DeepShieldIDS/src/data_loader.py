
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd


def load_csv(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place the CSV in data/raw or pass --cic/--unsw paths."
        )
    try:
        return pd.read_csv(path, low_memory=False, nrows=nrows)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False, nrows=nrows)


def load_datasets(cic_path: str | Path, unsw_path: str | Path, nrows: Optional[int] = None):
    return load_csv(cic_path, nrows=nrows), load_csv(unsw_path, nrows=nrows)
