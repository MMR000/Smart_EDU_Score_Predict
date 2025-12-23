import os
import argparse
import re
import pandas as pd

def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))
    return s[:120]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx_dir", required=True)
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--nrows", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    xlsx_dir = args.xlsx_dir
    if not os.path.isabs(xlsx_dir):
        xlsx_dir = os.path.abspath(os.path.join(os.getcwd(), xlsx_dir))

    files = [f for f in os.listdir(xlsx_dir) if f.lower().endswith(".xlsx")]
    if not files:
        print(f"[inspect] No .xlsx found in: {xlsx_dir}")
        return

    for fn in sorted(files):
        path = os.path.join(xlsx_dir, fn)
        try:
            xls = pd.ExcelFile(path, engine="openpyxl")
            print(f"\n=== {fn} ===")
            for sheet in xls.sheet_names:
                try:
                    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
                except Exception as e:
                    print(f"  - sheet {sheet}: read failed: {e}")
                    continue
                cols = list(df.columns)
                head = df.head(args.nrows)
                print(f"  - sheet: {sheet} | rows={len(df)} cols={len(cols)}")
                print("    columns:", cols[:30], ("..." if len(cols)>30 else ""))
                out_csv = os.path.join(args.out_dir, f"preview__{safe_name(fn)}__{safe_name(sheet)}.csv")
                head.to_csv(out_csv, index=False)
        except Exception as e:
            print(f"[inspect] Failed on {fn}: {e}")

    print(f"\n[inspect] Previews exported to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
