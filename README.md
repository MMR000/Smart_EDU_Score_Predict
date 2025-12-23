# SmartEDU Lab — Multi-Model Benchmark (Grades / Risk)

This folder is designed to live at:

- `~/Documents/smartedu/smartedu_lab`
- Your XLSX stay in the parent folder: `~/Documents/smartedu/*.xlsx`

Outputs:

- `artifacts/models/` trained models
- `results/metrics.csv` + `.md` + `.tex`
- `plots/*.png` including **live ETA plot** updated during training

Quick run:

```bash
cd ~/Documents/smartedu/smartedu_lab
bash run_all.sh
```

If auto-detection fails (different headers), run:

```bash
python scripts/inspect_xlsx.py --xlsx_dir ..
```

Then edit `configs/default.yaml` to pin the correct sheet/column names.
