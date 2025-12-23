# SmartEDU 深度学习对比实验补丁 (DL Patch)

你已经跑通了传统机器学习版本（run_all.sh）。这个补丁会在同一套 `data_prepared/` 数据上，自动跑 **多个深度学习架构** 的对比实验，并生成：

- `results/metrics_dl.csv`（以及 `.md` / `.tex`）
- `plots/*dl*.png`（含 ETA 进度图：`plots/eta_progress_dl.png`）
- `reports/report_dl.html`（可视化网页）
- `artifacts/models_dl/`（模型文件）
- `artifacts/encoders_dl/encoders_dl.json`（数值标准化 + 类别编码器）

## 覆盖的模型（回归 + 分类两套任务）

- MLP + Categorical Embeddings (`mlp_emb`)
- FT-Transformer (`ft_transformer`)
- AutoInt (`autoint`)
- Deep & Cross Network (`deep_cross`)
- TabNet (`tabnet`，来自 pytorch-tabnet)

> 分类任务默认是 `Mark < pass_mark`（默认 50）作为 `is_fail`，如果你的 prepared 数据里已经有 `is_fail` 列则直接使用。

## 一键运行

把补丁解压到你的 `smartedu_lab/` 根目录，然后执行：

```bash
unzip smartedu_lab_dl_patch.zip -d ~/Documents/smartedu/smartedu_lab
cd ~/Documents/smartedu/smartedu_lab
bash run_all_dl.sh
```

如果你想强制安装特定 CUDA 的 torch wheel，可以在运行前指定：

```bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
bash run_all_dl.sh
```

## 调参入口

`configs/dl.yaml`

- `training.max_epochs / batch_size / patience`
- `models.enabled`：开关模型
- 每个模型的超参块（如 `ft_transformer.d_token` 等）
