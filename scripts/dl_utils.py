import json, os, time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_split(path: str) -> pd.DataFrame:
    # auto fallback parquet -> csv
    if os.path.exists(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        if path.endswith(".csv"):
            return pd.read_csv(path)
    # try fallback with same base name
    if path.endswith(".parquet"):
        alt = path[:-8] + ".csv"
        if os.path.exists(alt):
            return pd.read_csv(alt)
    if path.endswith(".csv"):
        alt = path[:-4] + ".parquet"
        if os.path.exists(alt):
            return pd.read_parquet(alt)
    raise FileNotFoundError(f"Cannot find split file: {path} (or csv/parquet fallback)")

def infer_device(device_cfg: str) -> torch.device:
    if device_cfg == "cuda":
        return torch.device("cuda")
    if device_cfg == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def auto_feature_types(df: pd.DataFrame, targets: List[str], id_cols: List[str]):
    cols = [c for c in df.columns if c not in targets and c not in id_cols]
    cat_cols = []
    num_cols = []
    for c in cols:
        s = df[c]
        if s.dtype == "object" or str(s.dtype).startswith("category"):
            cat_cols.append(c)
        else:
            # heuristics: small unique integer-ish => categorical
            nunique = s.nunique(dropna=True)
            if nunique <= 50 and (pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s)):
                cat_cols.append(c)
            else:
                num_cols.append(c)
    return num_cols, cat_cols

@dataclass
class Encoders:
    cat_maps: Dict[str, Dict]
    num_mean: Dict[str, float]
    num_std: Dict[str, float]

def fit_encoders(df_train: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Encoders:
    cat_maps = {}
    for c in cat_cols:
        vals = df_train[c].astype("object").fillna("__NA__").unique().tolist()
        # reserve 0 for unknown
        mp = {v: i+1 for i, v in enumerate(sorted(vals))}
        cat_maps[c] = mp
    num_mean = {c: float(df_train[c].astype(float).mean()) for c in num_cols}
    num_std = {c: float(df_train[c].astype(float).std(ddof=0) if df_train[c].astype(float).std(ddof=0) > 1e-9 else 1.0) for c in num_cols}
    return Encoders(cat_maps=cat_maps, num_mean=num_mean, num_std=num_std)

def save_encoders(enc: Encoders, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "cat_maps": enc.cat_maps,
            "num_mean": enc.num_mean,
            "num_std": enc.num_std,
        }, f, ensure_ascii=False)

def encode_frame(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], enc: Encoders):
    # numeric -> float32 standardized
    X_num = np.zeros((len(df), len(num_cols)), dtype=np.float32)
    for j, c in enumerate(num_cols):
        v = df[c].astype(float).fillna(enc.num_mean[c]).to_numpy()
        X_num[:, j] = ((v - enc.num_mean[c]) / enc.num_std[c]).astype(np.float32)

    # categorical -> int64 indices
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        mp = enc.cat_maps[c]
        v = df[c].astype("object").fillna("__NA__").to_numpy()
        X_cat[:, j] = np.array([mp.get(x, 0) for x in v], dtype=np.int64)

    return X_num, X_cat

class TabularTorchDataset(torch.utils.data.Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.from_numpy(X_num)
        self.X_cat = torch.from_numpy(X_cat)
        self.y = torch.from_numpy(y)
    def __len__(self): return self.y.shape[0]
    def __getitem__(self, i):
        return self.X_num[i], self.X_cat[i], self.y[i]

def make_loaders(Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, y_va, Xn_te, Xc_te, y_te, batch_size, num_workers):
    tr = TabularTorchDataset(Xn_tr, Xc_tr, y_tr)
    va = TabularTorchDataset(Xn_va, Xc_va, y_va)
    te = TabularTorchDataset(Xn_te, Xc_te, y_te)

    tr_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    va_loader = torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    te_loader = torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr_loader, va_loader, te_loader

def now():
    return time.time()
