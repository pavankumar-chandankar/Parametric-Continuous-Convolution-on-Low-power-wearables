from __future__ import annotations

import argparse
import glob
import os
from typing import List

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect kernel sweep results into a single summary CSV.")
    ap.add_argument("--exp_tag", type=str, required=True, help="Same exp_tag passed to sweep scripts")
    ap.add_argument("--log_root", type=str, default=os.path.join("logs", "deepconvlstm"))
    args = ap.parse_args()

    root = os.path.join(args.log_root, args.exp_tag)
    pattern = os.path.join(root, "**", "best_macro_metrics_*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"[WARN] No files found: {pattern}")
        return

    dfs: List[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue
        if df.empty:
            continue
        df["source_path"] = p
        dfs.append(df)

    if not dfs:
        print("[WARN] No readable CSVs.")
        return

    all_df = pd.concat(dfs, ignore_index=True)

    needed = ["freq_tag", "conv_kernel_size", "conv_type", "f1_mean", "acc_mean", "prec_mean", "rec_mean", "split"]
    missing = [c for c in needed if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing columns in collected best_macro_metrics: {missing}")

    group_cols = ["freq_tag", "conv_type", "conv_kernel_size", "gamma_quant", "quant_bits", "standard_padding"]
    for c in group_cols:
        if c not in all_df.columns:
            all_df[c] = ""

    agg = (
        all_df.groupby(group_cols)[["f1_mean", "acc_mean", "prec_mean", "rec_mean"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    agg.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col for col in agg.columns
    ]

    out_csv = os.path.join(root, "kernel_sweep_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    agg.to_csv(out_csv, index=False)

    best = (
        agg.sort_values("f1_mean_mean", ascending=False)
        .groupby(["freq_tag", "conv_type"], as_index=False)
        .first()
        .sort_values(["conv_type", "freq_tag"])
    )
    best_csv = os.path.join(root, "kernel_sweep_best_per_freq.csv")
    best.to_csv(best_csv, index=False)

    print(f"[OK] Wrote summary: {out_csv}")
    print(f"[OK] Wrote best ks:  {best_csv}")
    print(best[["conv_type", "freq_tag", "conv_kernel_size", "f1_mean_mean", "f1_mean_std", "f1_mean_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
