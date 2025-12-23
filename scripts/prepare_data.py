import os
import argparse
import numpy as np
import pandas as pd

from utils import (
    load_yaml, ensure_dir, set_seed, choose_sheet_by_columns, read_excel_auto,
    detect_columns, safe_float, parse_start_year
)
from progress import init_progress, mark_start, mark_done, build_planned_tasks_from_config

def _xlsx(pdir: str, name: str) -> str:
    return os.path.join(pdir, name)

def _load_grade_table(cfg: dict) -> pd.DataFrame:
    pdir = cfg["paths"]["xlsx_dir"]
    journal_path = _xlsx(pdir, cfg["paths"]["journal_xlsx"])
    if not os.path.exists(journal_path):
        raise FileNotFoundError(f"Missing grade XLSX: {journal_path}")

    sheet_hint = cfg["schema"].get("journal_sheet")
    # we don't know exact headers; use fuzzy needed list
    needed = ["StudentID", "Subject", "Mark", "Term", "Academic Year"]
    sheet = choose_sheet_by_columns(journal_path, needed=needed, sheet_hint=sheet_hint)
    df = read_excel_auto(journal_path, sheet=sheet)

    # columns: pin or auto
    pinned = {k: cfg["schema"].get(k) for k in ["student_id", "subject", "mark", "term", "academic_year"]}
    if cfg["schema"].get("auto_detect", True):
        detected = detect_columns(df)
    else:
        detected = {}

    cols = {}
    for k in pinned:
        cols[k] = pinned[k] or detected.get(k)

    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise ValueError(
            f"Cannot detect columns {missing} in grade sheet '{sheet}'.\n"
            f"Found headers: {list(df.columns)}\n"
            f"Fix: open configs/default.yaml and set schema.{missing[0]} to the exact header text."
        )
    df = df.rename(columns={
        cols["student_id"]: "student_id",
        cols["subject"]: "subject",
        cols["mark"]: "mark",
        cols["term"]: "term",
        cols["academic_year"]: "academic_year",
    })
    df["term"] = safe_float(df["term"]).astype("Int64")
    df["mark"] = safe_float(df["mark"])
    df = df.dropna(subset=["student_id", "academic_year", "term", "mark"])
    allowed = set(cfg["dataset"]["allowed_terms"])
    df = df[df["term"].isin(list(allowed))].copy()
    return df

def _build_student_semester(grades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gcols = ["student_id", "academic_year", "term"]
    agg = grades.groupby(gcols).agg(
        avg_mark=("mark", "mean"),
        std_mark=("mark", "std"),
        min_mark=("mark", "min"),
        max_mark=("mark", "max"),
        n_subjects=("subject", pd.Series.nunique),
        n_marks=("mark", "size"),
    ).reset_index()
    agg["std_mark"] = agg["std_mark"].fillna(0.0)
    agg["start_year"] = agg["academic_year"].astype(str).apply(parse_start_year).astype(int)
    agg = agg.sort_values(["student_id", "start_year", "term"]).reset_index(drop=True)

    # lag + deltas
    agg["avg_mark_lag1"] = agg.groupby("student_id")["avg_mark"].shift(1)
    agg["delta_avg"] = agg["avg_mark"] - agg["avg_mark_lag1"]
    agg["rolling_avg_2"] = agg.groupby("student_id")["avg_mark"].rolling(2).mean().reset_index(level=0, drop=True)
    agg["rolling_std_2"] = agg.groupby("student_id")["avg_mark"].rolling(2).std().reset_index(level=0, drop=True).fillna(0.0)

    # targets: next semester average
    agg["target_next_avg"] = agg.groupby("student_id")["avg_mark"].shift(-1)
    agg = agg.dropna(subset=["target_next_avg"]).copy()
    agg["y_reg"] = agg["target_next_avg"].astype(float)
    thr = float(cfg["dataset"]["risk_threshold"])
    agg["y_clf"] = (agg["target_next_avg"] < thr).astype(int)
    return agg

def _split(df: pd.DataFrame, cfg: dict):
    tr = df[df["start_year"] <= int(cfg["split"]["train_max_year"])].copy()
    va = df[df["start_year"] == int(cfg["split"]["val_year"])].copy()
    te = df[df["start_year"] == int(cfg["split"]["test_year"])].copy()
    return tr, va, te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--task", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))
    ensure_dir("data_prepared")

    # init progress
    init_progress(args.progress, build_planned_tasks_from_config(cfg))
    mark_start(args.progress, args.task)

    grades = _load_grade_table(cfg)
    df = _build_student_semester(grades, cfg)
    tr, va, te = _split(df, cfg)

    df.to_parquet("data_prepared/student_semester.parquet", index=False)
    df.to_csv("data_prepared/student_semester.csv", index=False)
    tr.to_parquet("data_prepared/train.parquet", index=False)
    va.to_parquet("data_prepared/val.parquet", index=False)
    te.to_parquet("data_prepared/test.parquet", index=False)

    print("[prepare] student_semester:", df.shape, "train/val/test:", tr.shape, va.shape, te.shape)
    print("[prepare] saved -> data_prepared/")

    mark_done(args.progress, args.task)

if __name__ == "__main__":
    main()
