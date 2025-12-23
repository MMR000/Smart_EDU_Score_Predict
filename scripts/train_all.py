import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib

from utils import load_yaml, ensure_dir, set_seed
from modeling import build_feature_columns, fit_encoders, transform, save_encoders
from metrics import regression_metrics, classification_metrics, recall_at_topk
from progress import init_progress, mark_start, mark_done, build_planned_tasks_from_config

def _maybe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

def _load_splits():
    tr = pd.read_parquet("data_prepared/train.parquet")
    va = pd.read_parquet("data_prepared/val.parquet")
    te = pd.read_parquet("data_prepared/test.parquet")
    return tr, va, te

def _frame(df: pd.DataFrame, numeric_cols, cat_cols):
    out = df[numeric_cols + cat_cols].copy()
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in cat_cols:
        out[c] = out[c].astype(str).fillna("UNK")
    return out

def _append(rows, model, task, split, met, extra=None):
    r = {"model": model, "task": task, "split": split}
    r.update(met or {})
    if extra:
        r.update(extra)
    rows.append(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--progress", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))

    ensure_dir("artifacts/models")
    ensure_dir("results")

    # init progress
    init_progress(args.progress, build_planned_tasks_from_config(cfg))

    tr, va, te = _load_splits()
    numeric_cols, cat_cols = build_feature_columns(tr)

    enc = fit_encoders(tr, numeric_cols, cat_cols)
    save_encoders(enc, "artifacts/encoders.json")

    Xn_tr_std, Xc_tr = transform(tr, enc, standardize=True)
    Xn_va_std, Xc_va = transform(va, enc, standardize=True)
    Xn_te_std, Xc_te = transform(te, enc, standardize=True)

    Xn_tr_raw, _ = transform(tr, enc, standardize=False)
    Xn_va_raw, _ = transform(va, enc, standardize=False)
    Xn_te_raw, _ = transform(te, enc, standardize=False)

    F_tr = _frame(tr, numeric_cols, cat_cols)
    F_va = _frame(va, numeric_cols, cat_cols)
    F_te = _frame(te, numeric_cols, cat_cols)

    yreg_tr = tr["y_reg"].astype(float).values
    yreg_va = va["y_reg"].astype(float).values
    yreg_te = te["y_reg"].astype(float).values

    yclf_tr = tr["y_clf"].astype(int).values
    yclf_va = va["y_clf"].astype(int).values
    yclf_te = te["y_clf"].astype(int).values

    rows = []
    topk = float(cfg["reporting"]["topk_percent"])

    # ---------- Ridge (reg) ----------
    if cfg["models"].get("ridge_reg", False):
        mark_start(args.progress, "ridge_reg")
        t0 = time.time()
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=1.0, random_state=int(cfg["project"]["seed"]))
        m.fit(Xn_tr_std, yreg_tr)
        pred = m.predict(Xn_te_std)
        dt = time.time() - t0
        joblib.dump(m, "artifacts/models/ridge_reg.joblib")
        _append(rows, "ridge_reg", "reg", "test", regression_metrics(yreg_te, pred), {"train_sec": dt})
        mark_done(args.progress, "ridge_reg", dt)

    # ---------- Logistic Regression (clf) ----------
    if cfg["models"].get("logreg_clf", False):
        mark_start(args.progress, "logreg_clf")
        t0 = time.time()
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, n_jobs=int(cfg["training"]["n_jobs"]), class_weight="balanced")
        m.fit(Xn_tr_std, yclf_tr)
        prob = m.predict_proba(Xn_te_std)[:, 1]
        dt = time.time() - t0
        joblib.dump(m, "artifacts/models/logreg_clf.joblib")
        met = classification_metrics(yclf_te, prob, threshold=0.5)
        met.update(recall_at_topk(yclf_te, prob, topk))
        _append(rows, "logreg_clf", "clf", "test", met, {"train_sec": dt})
        mark_done(args.progress, "logreg_clf", dt)

    # ---------- CatBoost ----------
    if cfg["models"].get("catboost_reg", False) or cfg["models"].get("catboost_clf", False):
        cb = _maybe_import("catboost")
        if cb is None:
            print("[SKIP] catboost not installed")
        else:
            from catboost import CatBoostRegressor, CatBoostClassifier, Pool
            cat_idx = [F_tr.columns.get_loc(c) for c in cat_cols]
            use_gpu = bool(cfg["training"]["catboost"]["use_gpu"])
            task_type = "GPU" if use_gpu else "CPU"

            if cfg["models"].get("catboost_reg", False):
                mark_start(args.progress, "catboost_reg")
                t0 = time.time()
                m = CatBoostRegressor(
                    iterations=int(cfg["training"]["catboost"]["iterations"]),
                    learning_rate=float(cfg["training"]["catboost"]["learning_rate"]),
                    depth=int(cfg["training"]["catboost"]["depth"]),
                    loss_function="RMSE",
                    random_seed=int(cfg["project"]["seed"]),
                    task_type=task_type,
                    od_type="Iter",
                    od_wait=int(cfg["training"]["catboost"]["early_stopping_rounds"]),
                    verbose=False,
                )
                train_pool = Pool(F_tr, yreg_tr, cat_features=cat_idx)
                val_pool = Pool(F_va, yreg_va, cat_features=cat_idx) if len(F_va) else None
                m.fit(train_pool, eval_set=val_pool, use_best_model=True)
                pred = m.predict(F_te)
                dt = time.time() - t0
                m.save_model("artifacts/models/catboost_reg.cbm")
                _append(rows, "catboost_reg", "reg", "test", regression_metrics(yreg_te, pred), {"train_sec": dt})
                mark_done(args.progress, "catboost_reg", dt)

            if cfg["models"].get("catboost_clf", False):
                mark_start(args.progress, "catboost_clf")
                t0 = time.time()
                m = CatBoostClassifier(
                    iterations=int(cfg["training"]["catboost"]["iterations"]),
                    learning_rate=float(cfg["training"]["catboost"]["learning_rate"]),
                    depth=int(cfg["training"]["catboost"]["depth"]),
                    loss_function="Logloss",
                    random_seed=int(cfg["project"]["seed"]),
                    task_type=task_type,
                    od_type="Iter",
                    od_wait=int(cfg["training"]["catboost"]["early_stopping_rounds"]),
                    verbose=False,
                )
                train_pool = Pool(F_tr, yclf_tr, cat_features=cat_idx)
                val_pool = Pool(F_va, yclf_va, cat_features=cat_idx) if len(F_va) else None
                m.fit(train_pool, eval_set=val_pool, use_best_model=True)
                prob = m.predict_proba(F_te)[:, 1]
                dt = time.time() - t0
                m.save_model("artifacts/models/catboost_clf.cbm")
                met = classification_metrics(yclf_te, prob, threshold=0.5)
                met.update(recall_at_topk(yclf_te, prob, topk))
                _append(rows, "catboost_clf", "clf", "test", met, {"train_sec": dt})
                mark_done(args.progress, "catboost_clf", dt)

    # ---------- LightGBM ----------
    if cfg["models"].get("lightgbm_reg", False) or cfg["models"].get("lightgbm_clf", False):
        lgbm = _maybe_import("lightgbm")
        if lgbm is None:
            print("[SKIP] lightgbm not installed")
        else:
            import lightgbm as lgb
            device = "gpu" if bool(cfg["training"]["lightgbm"]["use_gpu"]) else "cpu"
            Ft_tr = F_tr.copy()
            Ft_va = F_va.copy()
            Ft_te = F_te.copy()
            for c in cat_cols:
                Ft_tr[c] = Ft_tr[c].astype("category")
                Ft_va[c] = Ft_va[c].astype("category")
                Ft_te[c] = Ft_te[c].astype("category")

            if cfg["models"].get("lightgbm_reg", False):
                mark_start(args.progress, "lightgbm_reg")
                t0 = time.time()
                m = lgb.LGBMRegressor(
                    n_estimators=int(cfg["training"]["lightgbm"]["n_estimators"]),
                    learning_rate=float(cfg["training"]["lightgbm"]["learning_rate"]),
                    num_leaves=int(cfg["training"]["lightgbm"]["num_leaves"]),
                    subsample=float(cfg["training"]["lightgbm"]["subsample"]),
                    colsample_bytree=float(cfg["training"]["lightgbm"]["colsample_bytree"]),
                    reg_lambda=float(cfg["training"]["lightgbm"]["reg_lambda"]),
                    random_state=int(cfg["project"]["seed"]),
                    device=device,
                )
                callbacks = [lgb.early_stopping(int(cfg["training"]["lightgbm"]["early_stopping_rounds"]), verbose=False)] if len(Ft_va) else None
                m.fit(Ft_tr, yreg_tr, eval_set=[(Ft_va, yreg_va)] if len(Ft_va) else None, eval_metric="rmse", callbacks=callbacks)
                pred = m.predict(Ft_te)
                dt = time.time() - t0
                joblib.dump(m, "artifacts/models/lightgbm_reg.joblib")
                _append(rows, "lightgbm_reg", "reg", "test", regression_metrics(yreg_te, pred), {"train_sec": dt})
                mark_done(args.progress, "lightgbm_reg", dt)

            if cfg["models"].get("lightgbm_clf", False):
                mark_start(args.progress, "lightgbm_clf")
                t0 = time.time()
                m = lgb.LGBMClassifier(
                    n_estimators=int(cfg["training"]["lightgbm"]["n_estimators"]),
                    learning_rate=float(cfg["training"]["lightgbm"]["learning_rate"]),
                    num_leaves=int(cfg["training"]["lightgbm"]["num_leaves"]),
                    subsample=float(cfg["training"]["lightgbm"]["subsample"]),
                    colsample_bytree=float(cfg["training"]["lightgbm"]["colsample_bytree"]),
                    reg_lambda=float(cfg["training"]["lightgbm"]["reg_lambda"]),
                    random_state=int(cfg["project"]["seed"]),
                    device=device,
                )
                callbacks = [lgb.early_stopping(int(cfg["training"]["lightgbm"]["early_stopping_rounds"]), verbose=False)] if len(Ft_va) else None
                m.fit(Ft_tr, yclf_tr, eval_set=[(Ft_va, yclf_va)] if len(Ft_va) else None, eval_metric="auc", callbacks=callbacks)
                prob = m.predict_proba(Ft_te)[:, 1]
                dt = time.time() - t0
                joblib.dump(m, "artifacts/models/lightgbm_clf.joblib")
                met = classification_metrics(yclf_te, prob, threshold=0.5)
                met.update(recall_at_topk(yclf_te, prob, topk))
                _append(rows, "lightgbm_clf", "clf", "test", met, {"train_sec": dt})
                mark_done(args.progress, "lightgbm_clf", dt)

    # ---------- XGBoost ----------
    if cfg["models"].get("xgboost_reg", False) or cfg["models"].get("xgboost_clf", False):
        xgbm = _maybe_import("xgboost")
        if xgbm is None:
            print("[SKIP] xgboost not installed")
        else:
            import xgboost as xgb
            # One-hot categories for xgboost
            def onehot(df):
                return pd.get_dummies(df, columns=cat_cols, dummy_na=True)
            OH_tr = onehot(F_tr)
            OH_va = onehot(F_va) if len(F_va) else OH_tr.iloc[:0].copy()
            OH_te = onehot(F_te)
            OH_tr, OH_va = OH_tr.align(OH_va, join="left", axis=1, fill_value=0)
            OH_tr, OH_te = OH_tr.align(OH_te, join="left", axis=1, fill_value=0)

            use_gpu = bool(cfg["training"]["xgboost"]["use_gpu"])
            tree_method = "hist"
            predictor = "auto"
            # auto-detect xgboost GPU support; fallback to CPU hist if not available
            if use_gpu:
                try:
                    import xgboost as xgb
                    _tmp = xgb.XGBRegressor(n_estimators=1, tree_method="gpu_hist")
                    tree_method = "gpu_hist"
                    predictor = "gpu_predictor"
                except Exception:
                    tree_method = "hist"
                    predictor = "auto"
            predictor = "gpu_predictor" if use_gpu else "auto"

            if cfg["models"].get("xgboost_reg", False):
                mark_start(args.progress, "xgboost_reg")
                t0 = time.time()
                m = xgb.XGBRegressor(
                    n_estimators=int(cfg["training"]["xgboost"]["n_estimators"]),
                    learning_rate=float(cfg["training"]["xgboost"]["learning_rate"]),
                    max_depth=int(cfg["training"]["xgboost"]["max_depth"]),
                    subsample=float(cfg["training"]["xgboost"]["subsample"]),
                    colsample_bytree=float(cfg["training"]["xgboost"]["colsample_bytree"]),
                    reg_lambda=float(cfg["training"]["xgboost"]["reg_lambda"]),
                    random_state=int(cfg["project"]["seed"]),
                    tree_method=tree_method,
                    predictor=predictor,
                )
                m.fit(OH_tr, yreg_tr, eval_set=[(OH_va, yreg_va)] if len(OH_va) else None, verbose=False)
                pred = m.predict(OH_te)
                dt = time.time() - t0
                m.save_model("artifacts/models/xgboost_reg.json")
                _append(rows, "xgboost_reg", "reg", "test", regression_metrics(yreg_te, pred), {"train_sec": dt})
                mark_done(args.progress, "xgboost_reg", dt)

            if cfg["models"].get("xgboost_clf", False):
                mark_start(args.progress, "xgboost_clf")
                t0 = time.time()
                m = xgb.XGBClassifier(
                    n_estimators=int(cfg["training"]["xgboost"]["n_estimators"]),
                    learning_rate=float(cfg["training"]["xgboost"]["learning_rate"]),
                    max_depth=int(cfg["training"]["xgboost"]["max_depth"]),
                    subsample=float(cfg["training"]["xgboost"]["subsample"]),
                    colsample_bytree=float(cfg["training"]["xgboost"]["colsample_bytree"]),
                    reg_lambda=float(cfg["training"]["xgboost"]["reg_lambda"]),
                    random_state=int(cfg["project"]["seed"]),
                    tree_method=tree_method,
                    predictor=predictor,
                    eval_metric="auc",
                )
                m.fit(OH_tr, yclf_tr, eval_set=[(OH_va, yclf_va)] if len(OH_va) else None, verbose=False)
                prob = m.predict_proba(OH_te)[:, 1]
                dt = time.time() - t0
                m.save_model("artifacts/models/xgboost_clf.json")
                met = classification_metrics(yclf_te, prob, threshold=0.5)
                met.update(recall_at_topk(yclf_te, prob, topk))
                _append(rows, "xgboost_clf", "clf", "test", met, {"train_sec": dt})
                mark_done(args.progress, "xgboost_clf", dt)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv("results/metrics.csv", index=False)
    print("[train] metrics -> results/metrics.csv")
    print("[train] models  -> artifacts/models/")

if __name__ == "__main__":
    main()
