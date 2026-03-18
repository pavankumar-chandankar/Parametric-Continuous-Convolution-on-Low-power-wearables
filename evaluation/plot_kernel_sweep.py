from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot macro-F1 vs kernel size from kernel_sweep_summary.csv")
    ap.add_argument("--summary", type=str, required=True, help="Path to kernel_sweep_summary.csv")
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default next to summary).")
    ap.add_argument("--conv_type", type=str, default="standard", help="Which conv_type to plot (default: standard).")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    if df.empty:
        raise ValueError("Summary CSV is empty.")

    for col in ["freq_tag", "conv_type", "conv_kernel_size", "f1_mean_mean"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in summary: {list(df.columns)}")

    df = df[df["conv_type"] == args.conv_type].copy()
    if df.empty:
        raise ValueError(f"No rows for conv_type={args.conv_type}")

    df["conv_kernel_size"] = df["conv_kernel_size"].astype(int)

    fig = plt.figure()
    ax = plt.gca()

    freq_tags = sorted(df["freq_tag"].unique(), key=lambda s: (len(s), s))
    for tag in freq_tags:
        d = df[df["freq_tag"] == tag].sort_values("conv_kernel_size")
        ax.plot(d["conv_kernel_size"], d["f1_mean_mean"], marker="o", label=tag)

    ax.set_xlabel("Kernel size")
    ax.set_ylabel("Macro F1 (mean across splits)")
    ax.set_title(f"Kernel sweep ({args.conv_type})")
    ax.legend()

    out = args.out.strip()
    if not out:
        out = os.path.join(os.path.dirname(args.summary), f"kernel_sweep_macro_f1_{args.conv_type}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved plot: {out}")


if __name__ == "__main__":
    main()
