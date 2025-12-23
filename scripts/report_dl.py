#!/usr/bin/env python3
import argparse, os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Template

def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def barplot(df, x, y, title, out):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(df[x].astype(str).tolist(), df[y].astype(float).tolist())
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def scatter(df, x, y, title, out):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df[x].astype(float), df[y].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics_dl.csv")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    out_dir = args.out_dir
    plots_dir = os.path.join(out_dir, "plots")
    reports_dir = os.path.join(out_dir, "reports")
    ensure_dir(plots_dir)
    ensure_dir(reports_dir)
    ensure_dir(os.path.join(out_dir, "results"))

    df = pd.read_csv(args.metrics)

    reg = df[df["task"] == "reg"].copy()
    cls = df[df["task"] == "cls"].copy()

    if len(reg):
        reg = reg.sort_values("RMSE", ascending=True)
        barplot(reg, "model", "RMSE", "Regression RMSE (lower=better)", os.path.join(plots_dir, "dl_reg_rmse.png"))
        barplot(reg, "model", "train_sec", "Train time (sec) - Regression", os.path.join(plots_dir, "dl_reg_time.png"))
        scatter(reg, "train_sec", "RMSE", "RMSE vs Train time (reg)", os.path.join(plots_dir, "dl_reg_time_vs_rmse.png"))

    if len(cls):
        if "PR_AUC" in cls.columns:
            cls_pr = cls.sort_values("PR_AUC", ascending=False)
            barplot(cls_pr, "model", "PR_AUC", "Classification PR_AUC (higher=better)", os.path.join(plots_dir, "dl_cls_prauc.png"))
            scatter(cls_pr, "train_sec", "PR_AUC", "PR_AUC vs Train time (cls)", os.path.join(plots_dir, "dl_cls_time_vs_prauc.png"))
        if "ROC_AUC" in cls.columns:
            cls_roc = cls.sort_values("ROC_AUC", ascending=False)
            barplot(cls_roc, "model", "ROC_AUC", "Classification ROC_AUC (higher=better)", os.path.join(plots_dir, "dl_cls_rocauc.png"))
        barplot(cls, "model", "train_sec", "Train time (sec) - Classification", os.path.join(plots_dir, "dl_cls_time.png"))

    # also export markdown/latex
    df.to_markdown(os.path.join(out_dir, "results", "metrics_dl.md"), index=False)
    try:
        df.to_latex(os.path.join(out_dir, "results", "metrics_dl.tex"), index=False, float_format="%.6f")
    except Exception:
        pass

    tpl = Template("""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>SmartEDU DL Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    h1 { margin-bottom: 4px; }
    .muted { color: #666; margin-top: 0px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    img { width: 100%; border: 1px solid #ddd; border-radius: 10px; }
    .card { border: 1px solid #eee; border-radius: 14px; padding: 14px; box-shadow: 0 1px 6px rgba(0,0,0,.05); }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; }
    th { background: #f7f7f7; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }
  </style>
</head>
<body>
  <h1>SmartEDU Deep Learning Benchmark</h1>
  <p class="muted">Artifacts: <span class="mono">results/metrics_dl.csv</span>, <span class="mono">artifacts/models_dl/</span>, <span class="mono">plots/*dl*.png</span></p>

  <div class="card">
    <h2>Metrics table</h2>
    {{ table | safe }}
  </div>

  <h2>Plots</h2>
  <div class="grid">
    {% for p in plots %}
    <div class="card">
      <div class="mono">{{ p }}</div>
      <img src="../{{ p }}" />
    </div>
    {% endfor %}
  </div>
</body>
</html>""")

    plot_files = [
        "plots/dl_reg_rmse.png",
        "plots/dl_reg_time.png",
        "plots/dl_reg_time_vs_rmse.png",
        "plots/dl_cls_prauc.png",
        "plots/dl_cls_rocauc.png",
        "plots/dl_cls_time.png",
        "plots/dl_cls_time_vs_prauc.png",
        "plots/eta_progress_dl.png",
    ]
    plot_files = [p for p in plot_files if os.path.exists(os.path.join(out_dir, p))]

    html = tpl.render(table=df.to_html(index=False, escape=False), plots=plot_files)
    out_html = os.path.join(reports_dir, "report_dl.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("[report_dl] ready:")
    print("  - results/metrics_dl.csv / .md / .tex")
    print("  - plots/*dl*.png")
    print("  - reports/report_dl.html")

if __name__ == "__main__":
    main()
