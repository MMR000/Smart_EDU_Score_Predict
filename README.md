# SmartEDU Score Predict Lab 🚀📚✨  
### End-to-End Student Performance Prediction & Risk Early‑Warning  
**Classic Machine Learning + Deep Learning for Tabular Data**  
**Auto Dataset Build → Multi‑Architecture Training → Fancy Plots → HTML Reports → Live ETA Tracking**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OS](https://img.shields.io/badge/OS-Ubuntu%2024.04-orange)
![GPU](https://img.shields.io/badge/GPU-RTX%204090-00c853)
![CUDA](https://img.shields.io/badge/CUDA-12.8-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1%2Bcu128-red)
![Tabular](https://img.shields.io/badge/Tabular-Learning-purple)
![Outputs](https://img.shields.io/badge/Outputs-CSV%20%7C%20MD%20%7C%20TEX%20%7C%20HTML%20%7C%20PNG-yellow)

---

## 🌟 TL;DR

This repository is a **one‑click, assignment‑ready** experimental suite for:

- **Regression:** predict numeric grades (score/mark)  
- **Classification:** risk / fail early‑warning (highly imbalanced supported)  
- **Models:** Classic ML + Deep tabular architectures (multi‑model, multi‑backbone)  
- **Outputs:** reproducible metrics tables (CSV/MD/TEX), **HTML reports**, **research‑style plots**, and **live ETA/progress curves**  

> ✅ If you need a project that looks like a mini research system (pipeline + tables + figures + report pages), this repo is designed exactly for that.

---

## 📌 Table of Contents

- [1. What’s Inside](#1-whats-inside)
- [2. Tasks & Metrics](#2-tasks--metrics)
- [3. Dataset Build (Student‑Semester)](#3-dataset-build-studentsemester)
- [4. Quick Start](#4-quick-start)
- [5. Results (Embedded Tables)](#5-results-embedded-tables)
- [6. 🏆 Leaderboard](#6--leaderboard)
- [7. ⏱️ Time‑Performance Trade‑off](#7-️-timeperformance-trade-off)
- [8. Visual Gallery](#8-visual-gallery)
- [9. Outputs Checklist](#9-outputs-checklist)
- [10. Configuration](#10-configuration)
- [11. Reproducibility](#11-reproducibility)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Assignment Report Template](#13-assignment-report-template)
- [14. License](#14-license)

---

## 1. What’s Inside

### ✅ Pipeline (end‑to‑end)

1) Inspect raw `.xlsx` tables (headers + previews exported to `reports/`)  
2) Merge multiple sources by **StudentID** and generate a **student‑semester** dataset  
3) Split into **train/val/test**  
4) Train **Classic ML** models  
5) Train **Deep tabular** models  
6) Evaluate, rank, export tables  
7) Generate plots + HTML reports  
8) Track progress → auto‑refresh **ETA curves** during training

### 🧠 Model Zoo (Classic + Deep)

**Classic ML**
- Regression: Ridge, LightGBM (GPU), CatBoost, XGBoost  
- Classification: Logistic Regression, LightGBM (GPU), CatBoost, XGBoost  

**Deep Learning for Tabular**
- MLP + Embeddings  
- FT‑Transformer  
- AutoInt  
- Deep & Cross Network (DCN)  
- TabNet  

---

## 2. Tasks & Metrics

### 2.1 Regression — Score/Mark Prediction
Predict a numeric grade for each student‑semester record.

- **RMSE** (↓)  
- **MAE** (↓)  
- **R²** (↑)

### 2.2 Classification — Risk Early‑Warning
Predict whether a record belongs to a “risk” category (highly imbalanced supported).

- **PR‑AUC** (↑) — more informative for imbalance  
- **ROC‑AUC** (↑)  
- **Recall@TopK** (↑) — capture positives in the top K% highest‑risk predictions  
- **F1** (↑) — shown for classic run where it is meaningful

> **Important:** Classic and DL classification may use different label definitions / positive rates depending on config.  
> Compare models **within the same run** unless label construction is identical.

---

## 3. Dataset Build (Student‑Semester)

A unified dataset is generated into `data_prepared/`:

- `student_semester.csv` / `student_semester.parquet`  
- `train.parquet`, `val.parquet`, `test.parquet`  

This makes the project **portable**: the training scripts operate on the prepared dataset rather than raw Excel files.

---

## 4. Quick Start

> Recommended: use a fresh environment (conda/venv).  
> The scripts are designed to work on Ubuntu 24.04 and can leverage GPU when available.

### 4.1 Create environment
```bash
conda create -n smartedu python=3.11 -y
conda activate smartedu
```

### 4.2 Install dependencies
```bash
cd smartedu_lab
pip install -r requirements.txt
pip install -r requirements_dl.txt
```

If you see missing optional packages:
```bash
pip install -U matplotlib pyyaml pyarrow
export MPLBACKEND=Agg
```

### 4.3 Run Classic ML (one command)
```bash
export MPLBACKEND=Agg
bash run_all.sh
```

### 4.4 Run Deep Learning (one command)
```bash
export MPLBACKEND=Agg
bash run_all_dl.sh
```

### 4.5 Run EVERYTHING (recommended for submission)
```bash
export MPLBACKEND=Agg
bash run_all.sh && bash run_all_dl.sh
```

---

## 5. Results (Embedded Tables)

The following tables are **test‑set** results produced by this repo run.  
The same content is exported to:

- Classic: `results/metrics.csv`, `results/metrics.md`, `results/metrics.tex`  
- DL: `results/metrics_dl.csv`, `results/metrics_dl.md`, `results/metrics_dl.tex`

### 5.1 Regression — Classic ML (Test Set)

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Train Time (s) ↓ |
|---|---:|---:|---:|---:|
| **ridge_reg** | **5.4263** | 4.0010 | **0.5519** | **0.014** |
| lightgbm_reg (GPU) | 5.4431 | **3.9859** | 0.5491 | 0.809 |
| catboost_reg | 5.4849 | 4.0631 | 0.5422 | 1.369 |
| xgboost_reg | 5.8416 | 4.3474 | 0.4807 | 6.363 |

**Takeaway:** Ridge is extremely strong and *ultra fast*; boosted trees are stable but do not beat Ridge here.

---

### 5.2 Regression — Deep Learning (Test Set)

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Train Time (s) ↓ |
|---|---:|---:|---:|---:|
| **tabnet_reg** | **1.8126** | **1.2756** | **0.9476** | 33.458 |
| mlp_emb_reg | 9.6234 | 7.9301 | -0.4757 | 3.893 |
| ft_transformer_reg | 15.4948 | 14.0685 | -2.8258 | 19.180 |
| deep_cross_reg | 19.3322 | 18.0096 | -4.9554 | 2.985 |
| autoint_reg | 30.2307 | 29.2018 | -13.5629 | 16.397 |

**Takeaway:** TabNet dominates regression in this configuration (very low RMSE, high R²).  
Other DL architectures likely need stronger tuning/regularization or different feature handling.

---

### 5.3 Classification — Classic ML (Test Set)

> Classic classification positive rate: **PosRate ≈ 0.0796** (depends on config).  
> PR‑AUC is a recommended metric under imbalance.

| Model | PR‑AUC ↑ | ROC‑AUC ↑ | Recall@Top5% ↑ | F1 ↑ | Train Time (s) ↓ |
|---|---:|---:|---:|---:|---:|
| **logreg_clf** | **0.4317** | **0.9050** | **0.3310** | **0.3776** | 1.114 |
| catboost_clf | 0.3994 | 0.8968 | 0.2977 | 0.0619 | 2.532 |
| lightgbm_clf (GPU) | 0.3594 | 0.8878 | 0.2837 | 0.0107 | **0.787** |
| xgboost_clf | 0.3083 | 0.8583 | 0.2457 | 0.2743 | 5.602 |

**Takeaway:** Logistic Regression leads PR‑AUC/ROC‑AUC/TopK recall in this run.

---

### 5.4 Classification — Deep Learning (Test Set)

> DL classification positive rate in this run: **PosRate ≈ 0.000432** (extremely rare positives).  
> In such cases, **Recall@TopK** is often the most practical decision metric.

| Model | PR‑AUC ↑ | ROC‑AUC ↑ | Recall@Top5% ↑ | Recall@Top10% ↑ | Train Time (s) ↓ |
|---|---:|---:|---:|---:|---:|
| **autoint_cls** | **0.1377** | **0.9949** | **1.0000** | **1.0000** | 3.483 |
| ft_transformer_cls | 0.0061 | 0.8295 | 0.5714 | 0.5714 | 4.092 |
| tabnet_cls | 0.00036 | 0.2485 | 0.0000 | 0.0000 | 7.869 |
| mlp_emb_cls | 0.00025 | 0.00059 | 0.0000 | 0.0000 | 1.789 |
| deep_cross_cls | 0.00025 | 0.00024 | 0.0000 | 0.0000 | **1.013** |

**Takeaway:** AutoInt is the best ranking model under extreme imbalance for this label definition.

---

## 6. 🏆 Leaderboard

### 🥇 Best Regression (Score Prediction)
- 🥇 **TabNet (DL)** — best accuracy (RMSE 1.8126, R² 0.9476)  
- 🥈 **Ridge (Classic)** — best classic baseline and *fastest overall*  
- 🥉 **LightGBM (Classic, GPU)** — strong, stable boosted baseline

### 🥇 Best Classification (Risk Early‑Warning)
- 🥇 **AutoInt (DL)** — strongest ranking under extreme imbalance (TopK recall = 1.0)  
- 🥈 **Logistic Regression (Classic)** — best classic PR‑AUC/ROC‑AUC  
- 🥉 **CatBoost / LightGBM (Classic)** — competitive alternatives

> If your course focuses on interpretability and deployment simplicity: **Ridge + Logistic Regression** are excellent baselines.  
> If your course focuses on predictive performance: **TabNet (regression) + AutoInt (classification)** are the winners under current setup.

---

## 7. ⏱️ Time‑Performance Trade‑off

This project explicitly measures training time and generates time-vs-metric plots (see `plots/`).

### 7.1 Regression
- **Ridge (Classic)**: near‑instant training (≈ 0.014s) with surprisingly strong RMSE.  
  ✅ Best for fast baselines and quick iteration.
- **LightGBM/CatBoost (Classic)**: moderate training time, stable performance, better nonlinearity handling.  
  ✅ Best if you want tree-based models with reasonable training cost.
- **TabNet (DL)**: higher training time (≈ 33s) but dramatically better accuracy in this run.  
  ✅ Best when accuracy is priority and GPU is available.

### 7.2 Classification
- Under high imbalance, **ranking quality** is key.  
- **AutoInt (DL)** achieves top recall at low training time (~3.5s), meaning it is both strong and efficient.  
- Classic models can be more stable and easier to explain; Logistic Regression performed best among classic models.

### 7.3 Practical “assignment-friendly” conclusion
If you want a clean story for an assignment report:

- **Baseline**: Ridge + Logistic Regression  
- **Best-performance**: TabNet (reg) + AutoInt (clf)  
- **Trade-off**: TabNet costs more time but wins on regression accuracy; AutoInt is a great balance for classification.

---

## 8. Visual Gallery

All figures are auto-generated into `plots/` and are ready to screenshot into your report.

### 8.1 Classic ML figures
| Type | File |
|---|---|
| Regression RMSE | `plots/compare_reg_rmse.png` |
| Regression R² | `plots/compare_reg_r2.png` |
| Regression MAE | `plots/compare_reg_mae.png` |
| Classification PR‑AUC | `plots/compare_clf_prauc.png` |
| Classification ROC‑AUC | `plots/compare_clf_rocauc.png` |
| Classification Recall@TopK | `plots/compare_clf_recall_topk.png` |
| Train time comparison | `plots/compare_train_time.png` |
| Heatmap | `plots/metrics_heatmap.png` |
| Radar (best reg) | `plots/radar_best_reg.png` |
| Radar (best clf) | `plots/radar_best_clf.png` |
| Live ETA | `plots/eta_progress.png` |

### 8.2 Deep Learning figures
| Type | File |
|---|---|
| DL regression RMSE | `plots/dl_reg_rmse.png` |
| DL regression time | `plots/dl_reg_time.png` |
| DL reg time vs RMSE | `plots/dl_reg_time_vs_rmse.png` |
| DL classification PR‑AUC | `plots/dl_cls_prauc.png` |
| DL classification ROC‑AUC | `plots/dl_cls_rocauc.png` |
| DL classification time | `plots/dl_cls_time.png` |
| DL cls time vs PR‑AUC | `plots/dl_cls_time_vs_prauc.png` |
| Live ETA (DL) | `plots/eta_progress_dl.png` |

### 8.3 Inline preview (renders on GitHub)
If these files exist in the repo, GitHub will render them directly:

**Live ETA (Classic)**  
![ETA Classic](plots/eta_progress.png)

**Live ETA (DL)**  
![ETA DL](plots/eta_progress_dl.png)

**Metrics Heatmap**  
![Heatmap](plots/metrics_heatmap.png)

**Radar Charts**  
![Radar Regression](plots/radar_best_reg.png)  
![Radar Classification](plots/radar_best_clf.png)

---

## 9. Outputs Checklist

After running, you should have:

✅ **Prepared dataset**  
- `data_prepared/student_semester.csv`  
- `data_prepared/train/val/test.parquet`

✅ **Models**  
- Classic: `artifacts/models/*`  
- DL: `artifacts/models_dl/*`

✅ **Metrics tables**  
- Classic: `results/metrics.csv` + `.md` + `.tex`  
- DL: `results/metrics_dl.csv` + `.md` + `.tex`

✅ **HTML reports**  
- `reports/report.html`  
- `reports/report_dl.html`

✅ **Plots**  
- `plots/*.png` (including ETA charts)

---

## 10. Configuration

Key configs:

- `configs/default.yaml` — classic training & evaluation  
- `configs/dl.yaml` — DL models, epochs, batch size, etc.

Typical knobs:
- epochs, batch size, lr  
- which models to enable/disable  
- risk label threshold / positive definition  
- topK ratios for Recall@TopK  

---

## 11. Reproducibility

- Splits are consistent within runs (train/val/test persisted).  
- Deterministic settings are used where applicable.  
- GPU acceleration is enabled when supported (e.g., LightGBM GPU trainer).

---

## 12. Troubleshooting

### Missing modules: matplotlib / yaml / pyarrow
```bash
pip install -U matplotlib pyyaml pyarrow
export MPLBACKEND=Agg
```

### XGBoost GPU error: `gpu_hist` not supported
Disable XGBoost GPU in config:
```yaml
training:
  xgboost:
    use_gpu: false
```

---

## 13. Assignment Report Template (Copy‑Paste Friendly)

**1) Data & Schema**  
Explain what Excel tables are used and how they are merged by StudentID.

**2) Tasks**  
Regression (score prediction) + classification (risk early warning).

**3) Preprocessing**  
Encoding strategy, missing values, normalization, splitting.

**4) Models**  
Classic baselines + deep tabular architectures.

**5) Metrics**  
RMSE/MAE/R²; PR‑AUC/ROC‑AUC/Recall@TopK.

**6) Results**  
Paste the tables from this README and include plots from `plots/`.

**7) Discussion**  
Leaderboards + time-performance trade‑off + why winners win.

**8) Reproducibility**  
Include the one‑click commands and output paths.

---

## 14. License

See `LICENSE`.

---

## ✅ One‑Command Regeneration

```bash
export MPLBACKEND=Agg
bash run_all.sh && bash run_all_dl.sh
```

---

### ⭐ If this repo helps your assignment
A star makes it look even more “real” 😄
