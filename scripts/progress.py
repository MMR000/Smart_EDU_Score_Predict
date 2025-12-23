import os
import json
import time
import argparse
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

def _now() -> int:
    return int(time.time())

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save(path: str, obj: dict):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_planned_tasks_from_config(cfg: dict) -> List[str]:
    tasks = ["prepare_data"]
    models = cfg.get("models", {})
    for k, v in models.items():
        if v:
            tasks.append(k)
    tasks.append("make_reports")
    return tasks

def init_progress(progress_path: str, planned_tasks: List[str]):
    p = _load(progress_path)
    if not p:
        p = {
            "start_ts": _now(),
            "planned_tasks": planned_tasks,
            "tasks": {},
            "eta_history": [],
            "last_update_ts": _now(),
        }
    else:
        p.setdefault("start_ts", _now())
        if not p.get("planned_tasks"):
            p["planned_tasks"] = planned_tasks
        p.setdefault("tasks", {})
        p.setdefault("eta_history", [])
        p["last_update_ts"] = _now()
    _save(progress_path, p)
    _touch_eta_point(progress_path)

def mark_start(progress_path: str, name: str):
    p = _load(progress_path)
    p.setdefault("tasks", {})
    t = p["tasks"].setdefault(name, {})
    if t.get("status") != "done":
        t["status"] = "running"
        t["start_ts"] = _now()
        p["last_update_ts"] = _now()
        _save(progress_path, p)

def _median_done_durations(tasks: Dict[str, dict]) -> float:
    durs = [t.get("elapsed_sec") for t in tasks.values()
            if t.get("status") == "done" and isinstance(t.get("elapsed_sec"), (int, float))]
    return float(np.median(durs)) if durs else 60.0

def _compute_eta(progress: dict) -> float:
    tasks = progress.get("tasks", {})
    planned = progress.get("planned_tasks", [])
    done = [t for t in planned if tasks.get(t, {}).get("status") == "done"]
    todo = [t for t in planned if tasks.get(t, {}).get("status") != "done"]
    elapsed_done = sum([tasks[t].get("elapsed_sec", 0.0) for t in done if t in tasks])
    med = _median_done_durations(tasks)
    total_est = elapsed_done + med * len(todo)
    return float(max(0.0, total_est - elapsed_done))

def _update_history(progress: dict, eta_sec: float):
    start_ts = progress.get("start_ts", _now())
    t_elapsed = _now() - start_ts
    hist = progress.setdefault("eta_history", [])
    hist.append({"t_elapsed_sec": int(t_elapsed), "eta_remaining_sec": float(eta_sec), "ts": _now()})
    if len(hist) > 500:
        del hist[:-500]

def _plot(progress: dict, out_path: str):
    hist = progress.get("eta_history", [])
    if len(hist) < 2:
        return
    x = [h["t_elapsed_sec"] / 60.0 for h in hist]
    y = [h["eta_remaining_sec"] / 60.0 for h in hist]
    _ensure_dir(os.path.dirname(out_path))
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Elapsed time (min)")
    plt.ylabel("Estimated remaining (min)")
    plt.title("Live ETA (auto-updated after each model/stage)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def _touch_eta_point(progress_path: str):
    p = _load(progress_path)
    eta = _compute_eta(p)
    p["eta_remaining_sec"] = float(eta)
    _update_history(p, eta)
    p["last_update_ts"] = _now()
    _save(progress_path, p)
    _plot(p, os.path.join("plots", "eta_progress.png"))

def mark_done(progress_path: str, name: str, elapsed_sec: Optional[float] = None):
    p = _load(progress_path)
    p.setdefault("tasks", {})
    t = p["tasks"].setdefault(name, {})
    if elapsed_sec is None:
        st = t.get("start_ts")
        elapsed_sec = float(_now() - st) if st else None
    t["status"] = "done"
    t["end_ts"] = _now()
    if elapsed_sec is not None:
        t["elapsed_sec"] = float(elapsed_sec)

    eta = _compute_eta(p)
    p["eta_remaining_sec"] = float(eta)
    _update_history(p, eta)
    p["last_update_ts"] = _now()
    _save(progress_path, p)
    _plot(p, os.path.join("plots", "eta_progress.png"))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--config", required=True)
    p_init.add_argument("--progress", required=True)

    p_start = sub.add_parser("start")
    p_start.add_argument("--progress", required=True)
    p_start.add_argument("--task", required=True)

    p_done = sub.add_parser("done")
    p_done.add_argument("--progress", required=True)
    p_done.add_argument("--task", required=True)
    p_done.add_argument("--elapsed", type=float, default=None)

    args = ap.parse_args()
    if args.cmd == "init":
        import yaml
        cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
        planned = build_planned_tasks_from_config(cfg)
        init_progress(args.progress, planned)
        print(f"[progress] init: {len(planned)} tasks -> plots/eta_progress.png")

    elif args.cmd == "start":
        mark_start(args.progress, args.task)
        print(f"[progress] start: {args.task}")

    elif args.cmd == "done":
        mark_done(args.progress, args.task, elapsed_sec=args.elapsed)
        p = _load(args.progress)
        print(f"[progress] done: {args.task} | ETA ~ {p.get('eta_remaining_sec', 0)/60.0:.1f} min")

if __name__ == "__main__":
    main()
