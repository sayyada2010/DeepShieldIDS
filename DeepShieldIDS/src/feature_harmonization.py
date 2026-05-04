
from __future__ import annotations
import re
import pandas as pd


def canonical_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    synonyms = {
        "dur": "duration", "flow_duration": "duration", "tot_fwd_pkts": "fwd_packets",
        "total_fwd_packets": "fwd_packets", "tot_bwd_pkts": "bwd_packets",
        "total_backward_packets": "bwd_packets", "proto": "protocol", "service": "service",
    }
    return synonyms.get(name, name)


def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    seen = {}
    for col in df.columns:
        base = canonical_name(col)
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        new_cols.append(base)
    df.columns = new_cols
    return df


def get_common_features(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    common = sorted(set(df_a.columns).intersection(set(df_b.columns)))
    if not common:
        raise ValueError("No common features after harmonization. Check dataset column names or use independent mode.")
    return common


def harmonize_pair(X_a: pd.DataFrame, X_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    Xa = rename_to_canonical(X_a)
    Xb = rename_to_canonical(X_b)
    common = get_common_features(Xa, Xb)
    return Xa[common].copy(), Xb[common].copy(), common
