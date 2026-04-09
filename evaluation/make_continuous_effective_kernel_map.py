from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import yaml


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _make_odd(k: int) -> int:
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--freq_tags", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {}) or {}
    dataset_cfg = cfg.get("dataset", {}) or {}

    conv_kernel_size = int(model_cfg.get("conv_kernel_size"))
    window_size = int(dataset_cfg.get("window_size"))

    kernel_support_s = model_cfg.get("kernel_support_s", None)
    if kernel_support_s is None:
        kernel_support_s = float(conv_kernel_size) / float(window_size)
    else:
        kernel_support_s = float(kernel_support_s)

    effective_kernel_by_freq: Dict[str, int] = {}
    rows = []

    for freq_tag in _parse_csv_list(args.freq_tags):
        rate = int(str(freq_tag).lower().replace("hz", "").strip())
        k = int(round(kernel_support_s * float(rate)))
        k = max(1, k)
        k = _make_odd(k)

        effective_kernel_by_freq[freq_tag] = int(k)
        rows.append(
            {
                "freq_tag": freq_tag,
                "sample_rate_hz": rate,
                "effective_kernel": int(k),
                "kernel_support_s": float(kernel_support_s),
                "base_conv_kernel_size": conv_kernel_size,
                "window_size": window_size,
            }
        )

    payload = {
        "config": args.config,
        "base_conv_kernel_size": conv_kernel_size,
        "window_size": window_size,
        "kernel_support_s": float(kernel_support_s),
        "effective_kernel_by_freq": effective_kernel_by_freq,
    }

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("[OK] wrote", args.out_json)

    if args.out_csv.strip():
        import pandas as pd

        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print("[OK] wrote", args.out_csv)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
