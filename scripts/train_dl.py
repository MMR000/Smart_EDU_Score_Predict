#!/usr/bin/env python3
import argparse, json, os, time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from scripts.dl_utils import (
    ensure_dir, set_seed, load_split, infer_device,
    auto_feature_types, fit_encoders, save_encoders,
    encode_frame, make_loaders, now
)
from scripts.dl_models import MLPEmb, FTTransformer, AutoInt, DeepCross
from scripts.metrics_dl import regression_metrics, classification_metrics

# TabNet (external)
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier

def write_progress(path: str, payload: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def get_targets(df: pd.DataFrame, cfg: Dict[str, Any]):
    """
    超稳目标列选择：
    1) 用 config 指定的 regression_target（支持大小写无关）
    2) 自动猜测：包含 mark/score/grade 的数值列
    3) 兜底：在数值列里选方差最大的（通常是分数）
    """
    reg_t_cfg = cfg["schema"]["regression_target"]
    cls_t = cfg["schema"]["classification_target"]
    pass_mark = float(cfg["schema"].get("pass_mark", 50))

    cols = list(df.columns)
    lowers = {c.lower(): c for c in cols}

    # 1) exact / case-insensitive
    reg_t = None
    if reg_t_cfg in cols:
        reg_t = reg_t_cfg
    elif reg_t_cfg.lower() in lowers:
        reg_t = lowers[reg_t_cfg.lower()]

    # 2) fuzzy: mark/score/grade
    if reg_t is None:
        for key in ["mark", "score", "grade"]:
            hits = [lowers[c] for c in lowers if key in c]
            # 尽量选数值列
            for h in hits:
                try:
                    x = pd.to_numeric(df[h], errors="coerce")
                    if x.notna().mean() > 0.8:
                        reg_t = h
                        break
                except Exception:
                    pass
            if reg_t is not None:
                break

    # 3) fallback: numeric col with max variance
    if reg_t is None:
        best = None
        for c in cols:
            try:
                x = pd.to_numeric(df[c], errors="coerce")
                if x.notna().mean() > 0.8:
                    v = float(x.var())
                    if best is None or v > best[1]:
                        best = (c, v)
            except Exception:
                pass
        if best is None:
            raise ValueError(f"Cannot infer regression target. Available columns: {cols[:30]} ...")
        reg_t = best[0]

    if reg_t not in df.columns:
        raise ValueError(f"Regression target '{reg_t}' not found in prepared data columns.")

    if cls_t not in df.columns:
        df[cls_t] = (pd.to_numeric(df[reg_t], errors="coerce").fillna(pass_mark) < pass_mark).astype(int)

    print(f"[targets] regression_target={reg_t} | classification_target={cls_t} | pass_mark={pass_mark}")
    return reg_t, cls_t

def torch_fit(
    model: nn.Module,
    task: str,
    tr_loader,
    va_loader,
    device,
    max_epochs: int,
    patience: int,
    precision: str,
):
    if task == "reg":
        loss_fn = nn.MSELoss()
        best_key = "RMSE"
        better = lambda a, b: a < b
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        best_key = "PR_AUC"
        better = lambda a, b: a > b

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "amp" and device.type == "cuda"))

    best = None
    best_state = None
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        pbar = tqdm(tr_loader, desc=f"epoch {epoch}/{max_epochs}", leave=False)
        for x_num, x_cat, y in pbar:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x_num, x_cat).squeeze(-1)
                if task == "cls":
                    y = y.float()
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # val
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x_num, x_cat, y in va_loader:
                x_num = x_num.to(device, non_blocking=True)
                x_cat = x_cat.to(device, non_blocking=True)
                out = model(x_num, x_cat).squeeze(-1)
                ys.append(y.detach().cpu().numpy())
                if task == "cls":
                    ps.append(torch.sigmoid(out).detach().cpu().numpy())
                else:
                    ps.append(out.detach().cpu().numpy())
        yv = np.concatenate(ys)
        pv = np.concatenate(ps)
        if task == "reg":
            m = regression_metrics(yv, pv)
        else:
            m = classification_metrics(yv, pv, threshold=0.5)

        cur = m.get(best_key, float("inf") if task == "reg" else float("-inf"))
        if best is None or better(cur, best):
            best = cur
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def torch_predict(model, loader, device, task: str):
    model.eval()
    ys, ps = [], []
    for x_num, x_cat, y in loader:
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        out = model(x_num, x_cat).squeeze(-1)
        ys.append(y.detach().cpu().numpy())
        if task == "cls":
            ps.append(torch.sigmoid(out).detach().cpu().numpy())
        else:
            ps.append(out.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

def run_tabnet(df_tr, df_va, df_te, num_cols, cat_cols, enc, task: str, cfg: Dict[str, Any], device):
    from scripts.dl_utils import encode_frame
    Xn_tr, Xc_tr = encode_frame(df_tr, num_cols, cat_cols, enc)
    Xn_va, Xc_va = encode_frame(df_va, num_cols, cat_cols, enc)
    Xn_te, Xc_te = encode_frame(df_te, num_cols, cat_cols, enc)

    X_tr = np.concatenate([Xn_tr, Xc_tr.astype(np.float32)], axis=1)
    X_va = np.concatenate([Xn_va, Xc_va.astype(np.float32)], axis=1)
    X_te = np.concatenate([Xn_te, Xc_te.astype(np.float32)], axis=1)

    # TabNet uses categorical indices; we pass them as additional info via cat_idxs/cat_dims
    cat_idxs = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    cat_dims = [max(enc.cat_maps[c].values(), default=0) + 1 for c in cat_cols]  # number of categories incl unknown
    # For safety, cap absurd dims
    cat_dims = [int(min(d, 50000)) for d in cat_dims]

    p = cfg["models"]["tabnet"]
    if task == "reg":
        y_tr = df_tr[cfg["schema"]["regression_target"]].astype(float).to_numpy().reshape(-1, 1)
        y_va = df_va[cfg["schema"]["regression_target"]].astype(float).to_numpy().reshape(-1, 1)
        y_te = df_te[cfg["schema"]["regression_target"]].astype(float).to_numpy().reshape(-1, 1)

        m = TabNetRegressor(
            n_d=p["n_d"], n_a=p["n_a"], n_steps=p["n_steps"], gamma=p["gamma"],
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=p["cat_emb_dim"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=p["lr"]),
            device_name="cuda" if device.type == "cuda" else "cpu",
        )
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=["rmse"],
            max_epochs=int(p["max_epochs"]),
            patience=int(p["patience"]),
            batch_size=int(cfg["training"]["batch_size"]),
            virtual_batch_size=min(1024, int(cfg["training"]["batch_size"])),
            num_workers=int(cfg["training"]["num_workers"]),
            drop_last=False,
        )
        pred = m.predict(X_te).reshape(-1)
        return y_te.reshape(-1), pred, m
    else:
        cls_t = cfg["schema"]["classification_target"]
        y_tr = df_tr[cls_t].astype(int).to_numpy()
        y_va = df_va[cls_t].astype(int).to_numpy()
        y_te = df_te[cls_t].astype(int).to_numpy()

        m = TabNetClassifier(
            n_d=p["n_d"], n_a=p["n_a"], n_steps=p["n_steps"], gamma=p["gamma"],
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=p["cat_emb_dim"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=p["lr"]),
            device_name="cuda" if device.type == "cuda" else "cpu",
        )
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=["auc"],
            max_epochs=int(p["max_epochs"]),
            patience=int(p["patience"]),
            batch_size=int(cfg["training"]["batch_size"]),
            virtual_batch_size=min(1024, int(cfg["training"]["batch_size"])),
            num_workers=int(cfg["training"]["num_workers"]),
            drop_last=False,
        )
        prob = m.predict_proba(X_te)[:, 1]
        return y_te, prob, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dl.yaml")
    ap.add_argument("--progress", default="logs/progress_dl.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg["training"]["seed"]))

    device = infer_device(cfg["training"]["device"])
    precision = cfg["training"].get("precision", "amp")

    # load splits
    df_tr = load_split(cfg["data"]["train"])
    df_va = load_split(cfg["data"]["val"])
    df_te = load_split(cfg["data"]["test"])

    reg_t, cls_t = get_targets(df_tr, cfg)
    # ensure label exists in val/test too
    if cls_t not in df_va.columns:
        df_va[cls_t] = (df_va[reg_t].astype(float) < float(cfg["schema"].get("pass_mark", 50))).astype(int)
    if cls_t not in df_te.columns:
        df_te[cls_t] = (df_te[reg_t].astype(float) < float(cfg["schema"].get("pass_mark", 50))).astype(int)

    id_cols = cfg["schema"].get("id_cols", [])
    targets = [reg_t, cls_t]

    num_cols, cat_cols = auto_feature_types(df_tr, targets=targets, id_cols=id_cols)

    enc = fit_encoders(df_tr, num_cols, cat_cols)
    save_encoders(enc, "artifacts/encoders_dl/encoders_dl.json")

    # encode
    Xn_tr, Xc_tr = encode_frame(df_tr, num_cols, cat_cols, enc)
    Xn_va, Xc_va = encode_frame(df_va, num_cols, cat_cols, enc)
    Xn_te, Xc_te = encode_frame(df_te, num_cols, cat_cols, enc)

    # y
    yreg_tr = df_tr[reg_t].astype(float).to_numpy().astype(np.float32)
    yreg_va = df_va[reg_t].astype(float).to_numpy().astype(np.float32)
    yreg_te = df_te[reg_t].astype(float).to_numpy().astype(np.float32)

    ycls_tr = df_tr[cls_t].astype(int).to_numpy().astype(np.int64)
    ycls_va = df_va[cls_t].astype(int).to_numpy().astype(np.int64)
    ycls_te = df_te[cls_t].astype(int).to_numpy().astype(np.int64)

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])
    tr_reg, va_reg, te_reg = make_loaders(Xn_tr, Xc_tr, yreg_tr, Xn_va, Xc_va, yreg_va, Xn_te, Xc_te, yreg_te, batch_size, num_workers)
    tr_cls, va_cls, te_cls = make_loaders(Xn_tr, Xc_tr, ycls_tr, Xn_va, Xc_va, ycls_va, Xn_te, Xc_te, ycls_te, batch_size, num_workers)

    enabled = cfg["models"]["enabled"]
    tasks = []
    for m in enabled:
        tasks.append({"name": f"{m}|reg", "status": "pending"})
        tasks.append({"name": f"{m}|cls", "status": "pending"})
    progress = {"total_tasks": len(tasks), "done_tasks": 0, "tasks": tasks}
    write_progress(args.progress, progress)

    rows = []

    def mark_done(idx, dur):
        progress["tasks"][idx]["status"] = "done"
        progress["tasks"][idx]["duration_sec"] = float(dur)
        progress["done_tasks"] = int(sum(1 for t in progress["tasks"] if t["status"] == "done"))
        write_progress(args.progress, progress)

    # helper for saving torch models
    def save_torch(model, path):
        ensure_dir(os.path.dirname(path))
        torch.save({"state_dict": model.state_dict(), "meta": {"num_cols": num_cols, "cat_cols": cat_cols}}, path)

    # run
    task_i = 0
    for name in enabled:
        # ---------------------- regression ----------------------
        t0 = now()
        if name == "tabnet":
            y_true, y_pred, model = run_tabnet(df_tr, df_va, df_te, num_cols, cat_cols, enc, "reg", cfg, device)
            met = regression_metrics(y_true, y_pred)
            train_sec = now() - t0
            ensure_dir("artifacts/models_dl")
            model.save_model(f"artifacts/models_dl/{name}_reg.tabnet")
        else:
            out_dim = 1
            if name == "mlp_emb":
                p = cfg["models"]["mlp_emb"]
                model = MLPEmb(len(num_cols), [max(enc.cat_maps[c].values(), default=0) for c in cat_cols], p["hidden"], p["dropout"], out_dim)
            elif name == "ft_transformer":
                p = cfg["models"]["ft_transformer"]
                model = FTTransformer(len(num_cols), [max(enc.cat_maps[c].values(), default=0) for c in cat_cols],
                                     p["d_token"], p["n_blocks"], p["n_heads"], p["dropout"], p["ffn_factor"], out_dim)
            elif name == "autoint":
                p = cfg["models"]["autoint"]
                model = AutoInt(len(num_cols), [max(enc.cat_maps[c].values(), default=0) for c in cat_cols],
                                p["d_token"], p["n_blocks"], p["n_heads"], p["dropout"], out_dim)
            elif name == "deep_cross":
                p = cfg["models"]["deep_cross"]
                model = DeepCross(len(num_cols), [max(enc.cat_maps[c].values(), default=0) for c in cat_cols],
                                  p["deep_hidden"], p["cross_layers"], p["dropout"], out_dim)
            else:
                raise ValueError(f"Unknown model: {name}")

            model = model.to(device)
            model = torch_fit(model, "reg", tr_reg, va_reg, device, int(cfg["training"]["max_epochs"]), int(cfg["training"]["patience"]), precision)
            y_true, y_pred = torch_predict(model, te_reg, device, "reg")
            met = regression_metrics(y_true, y_pred)
            train_sec = now() - t0
            save_torch(model, f"artifacts/models_dl/{name}_reg.pt")

        rows.append({"model": name, "task": "reg", "split": "test", **met, "train_sec": train_sec})
        mark_done(task_i, train_sec); task_i += 1

        # ---------------------- classification ----------------------
        t0 = now()
        if name == "tabnet":
            y_true, y_prob, model = run_tabnet(df_tr, df_va, df_te, num_cols, cat_cols, enc, "cls", cfg, device)
            met = classification_metrics(y_true, y_prob, threshold=0.5)
            train_sec = now() - t0
            ensure_dir("artifacts/models_dl")
            model.save_model(f"artifacts/models_dl/{name}_cls.tabnet")
        else:
            out_dim = 1
            if name == "mlp_emb":
                p = cfg["models"]["mlp_emb"]
                model = MLPEmb(len(num_cols), [max(enc.cat_maps[c].values(), default=0) for c in cat_cols], p["hidden"], p["dropout"], out_dim)
            elif name == "ft_transformer":
                p = cfg["models"]["ft_transformer"]
                model = FTTransformer(len(num_cols), [max(enc.cat_maps[c].values(), default=0) for c in cat_cols],
                                     p["d_token"], p["n_blocks"], p["n_heads"], p["dropout"], p["ffn_factor"], out_dim)
            elif name == "autoint":
                p = cfg["models"]["autoint"]
                model = AutoInt(len(num_cols), [max(enc.cat_maps[c].values(), default=0) for c in cat_cols],
                                p["d_token"], p["n_blocks"], p["n_heads"], p["dropout"], out_dim)
            elif name == "deep_cross":
                p = cfg["models"]["deep_cross"]
                model = DeepCross(len(num_cols), [max(enc.cat_maps[c].values(), default=0) for c in cat_cols],
                                  p["deep_hidden"], p["cross_layers"], p["dropout"], out_dim)
            else:
                raise ValueError(f"Unknown model: {name}")

            model = model.to(device)
            model = torch_fit(model, "cls", tr_cls, va_cls, device, int(cfg["training"]["max_epochs"]), int(cfg["training"]["patience"]), precision)
            y_true, y_prob = torch_predict(model, te_cls, device, "cls")
            met = classification_metrics(y_true, y_prob, threshold=0.5)
            train_sec = now() - t0
            save_torch(model, f"artifacts/models_dl/{name}_cls.pt")

        rows.append({"model": name, "task": "cls", "split": "test", **met, "train_sec": train_sec})
        mark_done(task_i, train_sec); task_i += 1

    # save metrics
    ensure_dir("results")
    dfm = pd.DataFrame(rows)
    dfm.to_csv("results/metrics_dl.csv", index=False)
    dfm.to_markdown("results/metrics_dl.md", index=False)
    try:
        dfm.to_latex("results/metrics_dl.tex", index=False, float_format="%.6f")
    except Exception:
        # if latex not available, still okay
        pass

    print("[train_dl] metrics -> results/metrics_dl.csv")
    print("[train_dl] models  -> artifacts/models_dl/")

if __name__ == "__main__":
    main()
