#!/usr/bin/env python3
import argparse, os, subprocess, sys, importlib

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def has_mod(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", default="requirements_dl.txt")
    args = ap.parse_args()

    # 1) ensure pip itself is recent
    try:
        run([sys.executable, "-m", "pip", "install", "-U", "pip"])
    except Exception:
        pass

    # 2) torch install: prefer CUDA wheel if torch missing
    if not has_mod("torch"):
        idx = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121")
        print(f"[deps] torch not found -> installing CUDA wheel from: {idx}", flush=True)
        run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", idx])

    # 3) install remaining requirements
    run([sys.executable, "-m", "pip", "install", "-r", args.requirements])

    # 4) quick sanity checks
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[deps] torch={torch.__version__} | device={dev}", flush=True)
    if dev == "cuda":
        print(f"[deps] cuda={torch.version.cuda} | gpu={torch.cuda.get_device_name(0)}", flush=True)

    # pyyaml name is 'yaml'
    import yaml  # noqa

    print("[deps] OK", flush=True)

if __name__ == "__main__":
    main()
