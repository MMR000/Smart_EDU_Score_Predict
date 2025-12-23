import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_yaml, ensure_dir, markdown_table, save_latex_table
from progress import mark_start, mark_done

def _save_md_tex(df: pd.DataFrame):
    ensure_dir("results")
    df.to_csv("results/metrics.csv", index=False)
    with open("results/metrics.md", "w", encoding="utf-8") as f:
        f.write(markdown_table(df))
    save_latex_table(df, "results/metrics.tex")

def _bar(df, y, title, fname, higher_better=False):
    if df.empty or y not in df.columns:
        return
    d = df.dropna(subset=[y]).copy()
    if d.empty:
        return
    d = d.sort_values(y, ascending=not higher_better)
    plt.figure()
    plt.bar(d["model"].astype(str), d[y].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", fname), dpi=240)
    plt.close()

def _heatmap(df: pd.DataFrame, fname: str):
    # quick heatmap of numeric metrics
    if df.empty:
        return
    d = df.copy()
    keep = ["model","task"] + [c for c in d.columns if c not in ["model","task","split"]]
    d = d[keep]
    num_cols = [c for c in d.columns if c not in ["model","task"] and pd.api.types.is_numeric_dtype(d[c])]
    if not num_cols:
        return
    pivot = d.set_index(["task","model"])[num_cols].fillna(np.nan)
    # normalize each metric for visualization
    M = pivot.copy()
    for c in num_cols:
        v = M[c].astype(float).values
        v = (v - np.nanmin(v)) / (np.nanmax(v)-np.nanmin(v)+1e-9)
        M[c] = v
    plt.figure(figsize=(10, max(3, 0.35*len(M))))
    plt.imshow(M.values, aspect="auto")
    plt.yticks(range(len(M.index)), [f"{t}/{m}" for t,m in M.index])
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
    plt.title("Metrics heatmap (min-max normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", fname), dpi=240)
    plt.close()

def _radar(df: pd.DataFrame, fname: str):
    # very “科研花里胡哨” radar for best reg + best clf
    reg = df[(df["task"]=="reg") & (df["split"]=="test")].copy()
    clf = df[(df["task"]=="clf") & (df["split"]=="test")].copy()
    if reg.empty and clf.empty:
        return

    # pick best by MAE (low) and PR_AUC (high)
    best_reg = None
    if not reg.empty and "MAE" in reg.columns:
        r = reg.dropna(subset=["MAE"]).sort_values("MAE")
        if not r.empty:
            best_reg = r.iloc[0]
    best_clf = None
    if not clf.empty:
        if "PR_AUC" in clf.columns and clf["PR_AUC"].notna().any():
            c = clf.dropna(subset=["PR_AUC"]).sort_values("PR_AUC", ascending=False)
            if not c.empty:
                best_clf = c.iloc[0]
        elif "ROC_AUC" in clf.columns and clf["ROC_AUC"].notna().any():
            c = clf.dropna(subset=["ROC_AUC"]).sort_values("ROC_AUC", ascending=False)
            if not c.empty:
                best_clf = c.iloc[0]

    # build a radar with 6 axes (normalize)
    axes = []
    values = []
    labels = []
    if best_reg is not None:
        labels.append(f"reg:{best_reg['model']}")
        axes = ["R2", "MAE", "RMSE"]
        v = []
        for a in axes:
            v.append(float(best_reg.get(a, np.nan)))
        values.append(v)
    if best_clf is not None:
        labels.append(f"clf:{best_clf['model']}")
        axes2 = ["ROC_AUC", "PR_AUC", "Recall@TopK"]
        v = []
        for a in axes2:
            v.append(float(best_clf.get(a, np.nan)))
        values.append(v)

    if not values:
        return

    # normalize each set internally (make higher better)
    def norm_vec(v, higher_flags):
        out=[]
        for x, hi in zip(v, higher_flags):
            if np.isnan(x):
                out.append(0.0); continue
            out.append(x)
        # min-max
        mn=min(out); mx=max(out)
        if mx-mn<1e-9:
            out=[0.5]*len(out)
        else:
            out=[(x-mn)/(mx-mn) for x in out]
        # if lower better, flip
        out=[o if hi else 1.0-o for o,hi in zip(out,higher_flags)]
        return out

    # Create two radars: one for reg, one for clf if exist
    ensure_dir("plots")

    if best_reg is not None:
        labs = ["R2↑","MAE↓","RMSE↓"]
        hv = [True, False, False]
        vec = norm_vec([float(best_reg.get("R2",0)), float(best_reg.get("MAE",0)), float(best_reg.get("RMSE",0))], hv)
        _radar_plot(vec, labs, f"Radar (best regression: {best_reg['model']})", "radar_best_reg.png")

    if best_clf is not None:
        labs = ["ROC_AUC↑","PR_AUC↑","Recall@TopK↑"]
        hv = [True, True, True]
        vec = norm_vec([float(best_clf.get("ROC_AUC",0) or 0), float(best_clf.get("PR_AUC",0) or 0), float(best_clf.get("Recall@TopK",0) or 0)], hv)
        _radar_plot(vec, labs, f"Radar (best classification: {best_clf['model']})", "radar_best_clf.png")

def _radar_plot(vals, labs, title, fname):
    # vals: list of floats [0..1]
    import math
    N = len(vals)
    angles = [2*math.pi*i/N for i in range(N)] + [0]
    data = vals + [vals[0]]
    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, data)
    ax.fill(angles, data, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labs)
    ax.set_yticklabels([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", fname), dpi=240)
    plt.close()

def _html_report(df: pd.DataFrame):
    # no extra deps: simple html pointing to images
    ensure_dir("reports")
    imgs = sorted([p for p in os.listdir("plots") if p.lower().endswith(".png")])
    table_html = df.to_html(index=False)
    parts = [
        "<html><head><meta charset='utf-8'><title>SmartEDU Benchmark Report</title></head><body>",
        "<h1>SmartEDU Benchmark Report</h1>",
        "<h2>Metrics Table</h2>",
        table_html,
        "<h2>Plots</h2>"
    ]
    for im in imgs:
        parts.append(f"<div style='margin:18px 0'><h3>{im}</h3><img src='../plots/{im}' style='max-width:1100px;width:100%'/></div>")
    parts.append("</body></html>")
    with open("reports/report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--task", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ensure_dir("plots")
    ensure_dir("results")

    mark_start(args.progress, args.task)

    df = pd.read_csv("results/metrics.csv")
    # ensure canonical columns exist
    _save_md_tex(df)

    reg = df[(df["task"]=="reg") & (df["split"]=="test")].copy()
    clf = df[(df["task"]=="clf") & (df["split"]=="test")].copy()

    # comparisons
    _bar(reg, "MAE", "Regression MAE (lower better)", "compare_reg_mae.png", higher_better=False)
    _bar(reg, "RMSE", "Regression RMSE (lower better)", "compare_reg_rmse.png", higher_better=False)
    _bar(reg, "R2", "Regression R2 (higher better)", "compare_reg_r2.png", higher_better=True)

    _bar(clf, "ROC_AUC", "Classification ROC-AUC (higher better)", "compare_clf_rocauc.png", higher_better=True)
    _bar(clf, "PR_AUC", "Classification PR-AUC (higher better)", "compare_clf_prauc.png", higher_better=True)
    _bar(clf, "Recall@TopK", "Risk Recall@TopK (higher better)", "compare_clf_recall_topk.png", higher_better=True)

    # training time fireworks
    if "train_sec" in df.columns:
        _bar(df[df["split"]=="test"], "train_sec", "Training time per model (sec)", "compare_train_time.png", higher_better=True)

    _heatmap(df[df["split"]=="test"].copy(), "metrics_heatmap.png")
    _radar(df[df["split"]=="test"].copy(), "radar.png")

    _html_report(df[df["split"]=="test"].copy())

    mark_done(args.progress, args.task)

    print("[report] ready:")
    print("  - results/metrics.csv / .md / .tex")
    print("  - plots/*.png")
    print("  - reports/report.html")

if __name__ == "__main__":
    main()
