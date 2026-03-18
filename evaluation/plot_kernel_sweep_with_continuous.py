from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KS_RE = re.compile(r"/ks(\d+)(/|$)")


def _infer_ks(path: str) -> int:
    m = KS_RE.search(path.replace("\\", "/"))
    if not m:
        raise ValueError(f"Cannot infer kernel size from path: {path}")
    return int(m.group(1))


def _read_best_macro(files):
    rows = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if "freq_tag" not in df.columns and "eval_tag" in df.columns:
            df = df.rename(columns={"eval_tag": "freq_tag"})
        if "freq_tag" not in df.columns or "f1_mean" not in df.columns:
            continue
        df = df.copy()
        df["source_path"] = p
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Ensure F1 is in percent (your logs are already in %)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std_root", type=str, required=True)
    ap.add_argument("--cont_root", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="wisdm_watch_accel")
    ap.add_argument("--out_png", type=str, required=True)
    args = ap.parse_args()

    target_freqs = ["20hz", "10hz", "5hz"]

    # ---------- standard sweep ----------
    std_files = sorted(glob.glob(os.path.join(args.std_root, "**", "best_macro_metrics_*.csv"), recursive=True))
    std_df = _read_best_macro(std_files)
    if std_df.empty:
        raise SystemExit(f"No usable standard best_macro_metrics under {args.std_root}")

    if "dataset_name" in std_df.columns:
        std_df = std_df[std_df["dataset_name"].astype(str) == args.dataset_name]

    std_df = std_df[std_df["freq_tag"].astype(str).isin(target_freqs)].copy()
    std_df["ks"] = std_df["source_path"].apply(_infer_ks)

    ks_unique = sorted(std_df["ks"].unique().tolist())

    # aggregate standard: mean±std across splits for each (freq, ks)
    std_agg = (
        std_df.groupby(["freq_tag", "ks"], as_index=False)["f1_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "f1_mean", "std": "f1_std", "count": "n"})
    )

    # ---------- continuous baseline ----------
    cont_files = sorted(glob.glob(os.path.join(args.cont_root, "**", "best_macro_metrics_*.csv"), recursive=True))
    cont_df = _read_best_macro(cont_files)
    if cont_df.empty:
        raise SystemExit(f"No usable continuous best_macro_metrics under {args.cont_root}")

    if "dataset_name" in cont_df.columns:
        cont_df = cont_df[cont_df["dataset_name"].astype(str) == args.dataset_name]
    if "conv_type" in cont_df.columns:
        cont_df = cont_df[cont_df["conv_type"].astype(str) == "continuous"]

    cont_df = cont_df[cont_df["freq_tag"].astype(str).isin(target_freqs)].copy()

    cont_agg = (
        cont_df.groupby(["freq_tag"], as_index=False)["f1_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "f1_mean", "std": "f1_std", "count": "n"})
    )

    # ---------- plot ----------
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.figure(figsize=(10, 6))

    if len(ks_unique) >= 2:
        # True kernel sweep: lines for standard, horizontal bands for continuous
        for freq in target_freqs:
            sub = std_agg[std_agg["freq_tag"] == freq].sort_values("ks")
            if sub.empty:
                continue
            plt.errorbar(
                sub["ks"], sub["f1_mean"], yerr=sub["f1_std"],
                marker="o", linestyle="-", capsize=3,
                label=f"Standard conv @ {freq} (n={int(sub['n'].max())})",
            )

        # continuous: one mean±std per freq as band
        xmin, xmax = min(ks_unique), max(ks_unique)
        for freq in target_freqs:
            sub = cont_agg[cont_agg["freq_tag"] == freq]
            if sub.empty:
                continue
            m = float(sub["f1_mean"].iloc[0])
            s = float(sub["f1_std"].iloc[0]) if np.isfinite(sub["f1_std"].iloc[0]) else 0.0
            n = int(sub["n"].iloc[0])
            plt.axhline(m, linestyle="--", linewidth=1.5, label=f"Continuous (MLP) @ {freq} (n={n})")
            if s > 0:
                plt.fill_between([xmin, xmax], [m - s, m - s], [m + s, m + s], alpha=0.10)

        plt.xlabel("conv\_kernel\_size (samples) — standard baseline")
        plt.ylabel("Macro-F1 (%)")
        plt.title(f"Kernel-size sensitivity (standard) vs continuous MLP kernels ({args.dataset_name})")
        plt.grid(True, alpha=0.3)
        plt.legend()

    else:
        # Only one kernel size -> bar chart is the correct visualization
        ks = ks_unique[0]
        std_bar = std_agg[std_agg["ks"] == ks].set_index("freq_tag")
        cont_bar = cont_agg.set_index("freq_tag")

        xs = np.arange(len(target_freqs))
        width = 0.38

        std_means = [float(std_bar.loc[f, "f1_mean"]) if f in std_bar.index else np.nan for f in target_freqs]
        std_stds  = [float(std_bar.loc[f, "f1_std"])  if f in std_bar.index else 0.0  for f in target_freqs]
        cont_means = [float(cont_bar.loc[f, "f1_mean"]) if f in cont_bar.index else np.nan for f in target_freqs]
        cont_stds  = [float(cont_bar.loc[f, "f1_std"])  if f in cont_bar.index else 0.0  for f in target_freqs]

        plt.bar(xs - width/2, std_means, width=width, yerr=std_stds, capsize=4, label=f"Standard conv (ks={ks})")
        plt.bar(xs + width/2, cont_means, width=width, yerr=cont_stds, capsize=4, label="Continuous (MLP kernel)")

        plt.xticks(xs, target_freqs)
        plt.xlabel("Effective sampling rate")
        plt.ylabel("Macro-F1 (%)")
        plt.title(f"Standard (single ks) vs continuous MLP kernels ({args.dataset_name})")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"[OK] wrote {args.out_png}")


if __name__ == "__main__":
    main()
