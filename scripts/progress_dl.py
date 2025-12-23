#!/usr/bin/env python3
import argparse, json, os, time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def load_progress(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def compute_eta(p):
    total = p.get("total_tasks", 1)
    done = p.get("done_tasks", 0)
    tasks = p.get("tasks", [])
    durs = [t.get("duration_sec", 0) for t in tasks if t.get("status") == "done" and t.get("duration_sec")]
    if done == 0 or not durs:
        return None
    avg = sum(durs) / len(durs)
    remain = (total - done) * avg
    return max(0, remain)

def plot(progress, out_path):
    ensure_dir(os.path.dirname(out_path))
    tasks = progress.get("tasks", [])
    labels = [t.get("name","") for t in tasks]
    done = [1 if t.get("status") == "done" else 0 for t in tasks]
    dur = [t.get("duration_sec", 0) for t in tasks]

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.barh(labels, done)
    ax1.set_title("Completed (1=done)")
    ax1.set_xlim(0, 1.0)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.barh(labels, dur)
    ax2.set_title("Duration (sec)")

    eta = compute_eta(progress)
    title = "ETA: calculating..."
    if eta is not None:
        m = int(eta // 60)
        s = int(eta % 60)
        title = f"ETA ~ {m:02d}:{s:02d} (mm:ss)"
    fig.suptitle(f"DL Progress | {title} | updated {datetime.now().strftime('%H:%M:%S')}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dl.yaml")
    ap.add_argument("--progress", default="logs/progress_dl.json")
    ap.add_argument("--out", default="plots/eta_progress_dl.png")
    ap.add_argument("--watch", type=float, default=0.0, help="seconds. if >0, watch mode")
    args = ap.parse_args()

    if args.watch and args.watch > 0:
        while True:
            p = load_progress(args.progress)
            if p:
                plot(p, args.out)
            time.sleep(args.watch)
    else:
        p = load_progress(args.progress)
        if p:
            plot(p, args.out)

if __name__ == "__main__":
    main()
