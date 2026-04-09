from __future__ import annotations
import argparse
import glob
import json
import os
import re
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

KS_RE = re.compile(r"/ks(\d+)(/|$)")


def infer_ks(path: str) -> int:
    m = KS_RE.search(path.replace("\\", "/"))
    if not m:
        raise ValueError(f"Cannot infer kernel size from path: {path}")
    return int(m.group(1))


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def read_best_macro(files: List[str]) -> pd.DataFrame:
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if "freq_tag" not in df.columns and "eval_tag" in df.columns:
            df = df.rename(columns={"eval_tag": "freq_tag"})
        if "freq_tag" not in df.columns or "f1_mean" not in df.columns:
            continue
        df = df.copy()
        df["source_path"] = f
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_cont_kernel_map(path: str, target_freqs: List[str]) -> Dict[str, int]:
    with open(path, "r") as f:
        obj = json.load(f)
    if "effective_kernel_by_freq" in obj:
        obj = obj["effective_kernel_by_freq"]
    out = {}
    for freq in target_freqs:
        out[freq] = int(obj[freq])
    return out


def load_manual_cont_stats(path: str, target_freqs: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"freq_tag", "mean", "std"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[df["freq_tag"].astype(str).isin(target_freqs)].copy()
    return df.rename(columns={"mean": "cont_f1_mean", "std": "cont_f1_std"})[["freq_tag", "cont_f1_mean", "cont_f1_std"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--std_root", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--freq_tags", required=True)
    ap.add_argument("--kernel_values", required=True)
    ap.add_argument("--cont_kernel_json", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--cont_root", default="")
    ap.add_argument("--cont_stats_csv", default="")
    ap.add_argument("--cont_ylim", default="0.4,0.9")
    args = ap.parse_args()

    target_freqs = parse_csv_list(args.freq_tags)
    keep_kernels = sorted(set(parse_int_list(args.kernel_values)))
    cont_kernel_map = load_cont_kernel_map(args.cont_kernel_json, target_freqs)
    cont_ylim = [float(x.strip()) for x in args.cont_ylim.split(",")]

    std_files = sorted(glob.glob(os.path.join(args.std_root, "**", "best_macro_metrics_*.csv"), recursive=True))
    std_df = read_best_macro(std_files)
    if std_df.empty:
        raise SystemExit(f"No usable standard CSVs under {args.std_root}")

    if "dataset_name" in std_df.columns:
        std_df = std_df[std_df["dataset_name"].astype(str) == args.dataset_name]
    if "conv_type" in std_df.columns:
        std_df = std_df[std_df["conv_type"].astype(str).isin(["standard", "standard_multibranch"])]

    std_df = std_df[std_df["freq_tag"].astype(str).isin(target_freqs)].copy()
    std_df["ks"] = std_df["source_path"].apply(infer_ks)
    std_df = std_df[std_df["ks"].isin(keep_kernels)].copy()

    if std_df.empty:
        raise SystemExit("No standard rows left after filtering.")

    std_agg = (
        std_df.groupby(["freq_tag", "ks"], as_index=False)["f1_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "std_f1_mean", "std": "std_f1_std", "count": "n"})
    )
    std_agg["std_f1_mean"] = std_agg["std_f1_mean"] / 100.0
    std_agg["std_f1_std"] = std_agg["std_f1_std"] / 100.0

    if args.cont_stats_csv.strip():
        cont_agg = load_manual_cont_stats(args.cont_stats_csv, target_freqs)
    else:
        if not args.cont_root.strip():
            raise SystemExit("Provide either --cont_root or --cont_stats_csv")
        cont_files = sorted(glob.glob(os.path.join(args.cont_root, "**", "best_macro_metrics_*.csv"), recursive=True))
        cont_df = read_best_macro(cont_files)
        if cont_df.empty:
            raise SystemExit(f"No usable continuous CSVs under {args.cont_root}")

        if "dataset_name" in cont_df.columns:
            cont_df = cont_df[cont_df["dataset_name"].astype(str) == args.dataset_name]
        if "conv_type" in cont_df.columns:
            cont_df = cont_df[cont_df["conv_type"].astype(str) == "continuous"]

        cont_df = cont_df[cont_df["freq_tag"].astype(str).isin(target_freqs)].copy()
        if cont_df.empty:
            raise SystemExit("No continuous rows left after filtering.")

        cont_agg = (
            cont_df.groupby(["freq_tag"], as_index=False)["f1_mean"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "cont_f1_mean", "std": "cont_f1_std", "count": "n"})
        )
        cont_agg["cont_f1_mean"] = cont_agg["cont_f1_mean"] / 100.0
        cont_agg["cont_f1_std"] = cont_agg["cont_f1_std"] / 100.0

    cont_agg["effective_kernel"] = cont_agg["freq_tag"].map(cont_kernel_map)

    summary = pd.merge(
        cont_agg,
        std_agg.rename(columns={"ks": "effective_kernel", "std_f1_mean": "std_same_kernel_mean", "std_f1_std": "std_same_kernel_std"}),
        on=["freq_tag", "effective_kernel"],
        how="left",
    )
    summary["delta_cont_minus_std_at_same_kernel"] = summary["cont_f1_mean"] - summary["std_same_kernel_mean"]

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    summary.sort_values(["freq_tag"]).to_csv(args.out_csv, index=False)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 12.5,
        "axes.labelsize": 13.5,
        "axes.titlesize": 13.5,
        "legend.fontsize": 11.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # compact x positions instead of true kernel values
    x_pos = {k: i for i, k in enumerate(keep_kernels)}
    x_ticks = list(range(len(keep_kernels)))
    x_ticklabels = [str(k) for k in keep_kernels]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(5.8, 6.0), gridspec_kw={"height_ratios": [1.15, 2.55]}, sharex=True
    )

    # explicit stable colors
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {freq: cycle[i % len(cycle)] for i, freq in enumerate(target_freqs)}
    legend_handles = []

    # Top panel: continuous on shared compact x axis
    for freq in target_freqs:
        row = cont_agg[cont_agg["freq_tag"] == freq]
        if row.empty:
            continue

        k = int(row["effective_kernel"].iloc[0])
        y = float(row["cont_f1_mean"].iloc[0])
        s_raw = row["cont_f1_std"].iloc[0]
        s = float(s_raw) if pd.notna(s_raw) else 0.0
        color = color_map[freq]

        ax_top.errorbar(
            [x_pos[k]],
            [y],
            yerr=[s],
            fmt="o",
            markersize=8.5,
            capsize=2.8,
            linestyle="none",
            color=color,
            markeredgecolor="black",
            markeredgewidth=1.0,
            zorder=5,
        )

        legend_handles.append(
            Line2D([0], [0], color=color, marker="o", linewidth=1.8, markersize=4.8, label=freq)
        )

    ax_top.set_title("Continuous conv", pad=4)
    ax_top.set_ylabel("Macro-F1")
    ax_top.set_ylim(cont_ylim[0], cont_ylim[1])
    ax_top.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax_top.grid(axis="x", visible=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Bottom panel: standard sweep on compact x axis
    for freq in target_freqs:
        sub = std_agg[std_agg["freq_tag"] == freq].sort_values("ks")
        if sub.empty:
            continue

        color = color_map[freq]
        xs = [x_pos[int(k)] for k in sub["ks"].tolist()]

        ax_bottom.plot(
            xs,
            sub["std_f1_mean"],
            marker="o",
            linewidth=1.8,
            markersize=4.6,
            color=color,
            label=freq,
        )

        ax_bottom.errorbar(
            xs,
            sub["std_f1_mean"],
            yerr=sub["std_f1_std"].fillna(0.0),
            fmt="none",
            capsize=2.3,
            elinewidth=0.82,
            alpha=0.55,
            color=color,
        )

    ax_bottom.set_title("Standard DeepConvLSTM", pad=4)
    ax_bottom.set_ylabel("Macro-F1")
    ax_bottom.set_xlabel("Temporal kernel size")
    ax_bottom.set_xticks(x_ticks)
    ax_bottom.set_xticklabels(x_ticklabels)
    ax_bottom.set_ylim(0.0, 1.0)
    ax_bottom.set_yticks([i / 10.0 for i in range(0, 11)])
    ax_bottom.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax_bottom.grid(axis="x", visible=False)
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)

    fig.suptitle(args.title, y=0.987, fontsize=16)

    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=len(target_freqs),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        columnspacing=1.0,
        handletextpad=0.45,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(args.out_png, dpi=400, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")

    print("[OK] wrote", args.out_png)
    print("[OK] wrote", args.out_pdf)
    print("[OK] wrote", args.out_csv)
    print(summary.sort_values(["freq_tag"]).to_string(index=False))


if __name__ == "__main__":
    main()
