#!/usr/bin/env python3
# visualize_plan_noargs.py
# ---------------------------------------------------------------
# If no .pkl path is given, defaults to the one hard‑coded below.
# Otherwise behaves exactly like the earlier script.

import argparse
import pickle
from pathlib import Path
import yaml  # pip install pyyaml   (only needed for --dump-yaml)

# ---------- default file ----------------------------------------------------
DEFAULT_PKL = Path(
    "/storage1/sibai/Active/ihab/research_new/dino_wm/plan_outputs/20250603164812_pusht_gH5/plan_targets.pkl"
)

# ---------- helper functions (same as before) -------------------------------
def short_type(x):
    return type(x).__name__

def tensor_like(x):
    try:
        import torch, numpy as np
        if isinstance(x, torch.Tensor):
            return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
        if isinstance(x, np.ndarray):
            return f"ndarray(shape={x.shape}, dtype={x.dtype})"
    except ModuleNotFoundError:
        pass
    return None

def summarize(obj, indent=0, max_depth=3):
    pad = "    " * indent
    if max_depth < 0:
        print(pad + "...")
        return
    tl = tensor_like(obj)
    if tl:
        print(pad + tl)
    elif isinstance(obj, dict):
        print(pad + f"dict (len={len(obj)})")
        for k, v in obj.items():
            print(pad + f"  └─ {k}: ", end="")
            summarize(v, indent + 2, max_depth - 1)
    elif isinstance(obj, (list, tuple)):
        print(pad + f"{short_type(obj)} (len={len(obj)})")
        for i, v in enumerate(obj[:10]):          # show first 10 items
            print(pad + f"  └─ [{i}]: ", end="")
            summarize(v, indent + 2, max_depth - 1)
        if len(obj) > 10:
            print(pad + "  └─ [...]")
    else:
        print(pad + repr(obj))

# ---------- main ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inspect plan_targets.pkl")
    parser.add_argument("pkl", nargs="?", type=Path,
                        default=DEFAULT_PKL,
                        help=f"path to .pkl (default: {DEFAULT_PKL})")
    parser.add_argument("--dump-yaml", metavar="FILE",
                        help="dump full object to YAML file")
    args = parser.parse_args()

    print(f"\nLoading: {args.pkl}\n")
    with args.pkl.open("rb") as f:
        data = pickle.load(f)

    summarize(data)

    if args.dump_yaml:
        with open(args.dump_yaml, "w") as fy:
            yaml.safe_dump(data, fy, default_flow_style=False)
        print(f"\n→ Full YAML written to {args.dump_yaml}")

if __name__ == "__main__":
    main()
