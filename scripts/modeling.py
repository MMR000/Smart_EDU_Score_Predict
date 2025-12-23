import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from utils import ensure_dir, safe_float

@dataclass
class Encoders:
    numeric_cols: List[str]
    cat_cols: List[str]
    cat_maps: Dict[str, Dict[str, int]]
    num_means: Dict[str, float]
    num_stds: Dict[str, float]

def build_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # robust: use any numeric-ish engineered cols that exist
    numeric_candidates = [
        "avg_mark", "std_mark", "min_mark", "max_mark",
        "n_subjects", "n_marks",
        "avg_mark_lag1", "delta_avg",
        "rolling_avg_2", "rolling_std_2",
        "age"
    ]
    numeric = [c for c in numeric_candidates if c in df.columns]
    # any extra numeric columns:
    for c in df.columns:
        if c in numeric or c in ["y_reg", "y_clf", "target_next_avg"]:
            continue
        if str(c).startswith("num__"):
            numeric.append(c)

    # categorical: keep small-cardinality object columns
    cat = []
    for c in df.columns:
        if c in numeric or c in ["y_reg", "y_clf", "target_next_avg"]:
            continue
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
            # avoid huge text
            if df[c].nunique(dropna=True) <= 2000:
                cat.append(c)
    return numeric, cat

def fit_encoders(train_df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]) -> Encoders:
    cat_maps = {}
    for c in cat_cols:
        vals = train_df[c].astype(str).fillna("UNK").unique().tolist()
        cat_maps[c] = {v: i + 1 for i, v in enumerate(vals)}  # 0 for unseen

    num_means = {}
    num_stds = {}
    for c in numeric_cols:
        s = pd.to_numeric(train_df[c], errors="coerce")
        mu = float(np.nanmean(s)) if np.isfinite(np.nanmean(s)) else 0.0
        sd = float(np.nanstd(s)) if np.isfinite(np.nanstd(s)) and np.nanstd(s) > 0 else 1.0
        num_means[c] = mu
        num_stds[c] = sd
    return Encoders(numeric_cols=numeric_cols, cat_cols=cat_cols, cat_maps=cat_maps, num_means=num_means, num_stds=num_stds)

def transform(df: pd.DataFrame, enc: Encoders, standardize: bool) -> Tuple[np.ndarray, np.ndarray]:
    Xn = []
    for c in enc.numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").fillna(enc.num_means[c]).astype(float).values
        if standardize:
            s = (s - enc.num_means[c]) / enc.num_stds[c]
        Xn.append(s)
    Xn = np.stack(Xn, axis=1) if Xn else np.zeros((len(df), 0), dtype=np.float32)

    Xc = []
    for c in enc.cat_cols:
        m = enc.cat_maps[c]
        v = df[c].astype(str).fillna("UNK").map(m).fillna(0).astype(int).values
        Xc.append(v)
    Xc = np.stack(Xc, axis=1) if Xc else np.zeros((len(df), 0), dtype=np.int64)
    return Xn.astype(np.float32), Xc.astype(np.int64)

def save_encoders(enc: Encoders, path: str):
    ensure_dir(os.path.dirname(path))
    obj = {
        "numeric_cols": enc.numeric_cols,
        "cat_cols": enc.cat_cols,
        "cat_maps": enc.cat_maps,
        "num_means": enc.num_means,
        "num_stds": enc.num_stds
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_encoders(path: str) -> Encoders:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return Encoders(**obj)
