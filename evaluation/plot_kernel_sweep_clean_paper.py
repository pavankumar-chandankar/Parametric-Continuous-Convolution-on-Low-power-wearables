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
    args = ap.parse_args()

    target_freqs = parse_csv_list(args.freq_tags)
    keep_kernels = set(parse_int_list(args.kernel_values))
    cont_kernel_map = load_cont_kernel_map(args.cont_kernel_json, target_freqs)

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

    same_kernel_std = std_agg.rename(columns={"ks": "effective_kernel"})
    summary = pd.merge(cont_agg, same_kernel_std, on=["freq_tag", "effective_kernel"], how="left")
    summary["delta_cont_minus_std_at_same_kernel"] = summary["cont_f1_mean"] - summary["std_f1_mean"]

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    summary.sort_values(["freq_tag"]).to_csv(args.out_csv, index=False)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(7.4, 8.0))
    ks_unique = sorted(std_agg["ks"].unique().tolist())
    legend_handles = []

    for freq in target_freqs:
        sub = std_agg[std_agg["freq_tag"] == freq].sort_values("ks")
        if sub.empty:
            continue

        line, = ax.plot(
            sub["ks"],
            sub["std_f1_mean"],
            marker="o",
            linewidth=1.9,
            markersize=4.8,
            label=freq,
        )
        color = line.get_color()

        ax.errorbar(
            sub["ks"],
            sub["std_f1_mean"],
            yerr=sub["std_f1_std"].fillna(0.0),
            fmt="none",
            capsize=3.5,
            elinewidth=0.9,
            alpha=0.6,
            color=color,
        )

        crow = cont_agg[cont_agg["freq_tag"] == freq]
        if not crow.empty:
            x = float(crow["effective_kernel"].iloc[0])
            y = float(crow["cont_f1_mean"].iloc[0])
            s_raw = crow["cont_f1_std"].iloc[0]
            s = float(s_raw) if pd.notna(s_raw) else 0.0

            ax.errorbar(
                [x],
                [y],
                yerr=[s],
                fmt="o",
                markersize=9,
                capsize=3,
                linestyle="none",
                color=color,
                markeredgecolor="black",
                markeredgewidth=1.0,
                zorder=6,
            )

        legend_handles.append(
            Line2D([0], [0], color=color, marker="o", linewidth=1.9, markersize=5, label=freq)
        )

    legend_handles.append(
        Line2D([0], [0], linestyle="none", marker="o", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=1.0, markersize=8,
               label="Continuous conv")
    )

    ax.set_xlabel("Temporal kernel size")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(ks_unique)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([i / 10.0 for i in range(0, 11)])
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(args.title, y=0.985, fontsize=14)

    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=len(target_freqs) + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        columnspacing=1.2,
        handletextpad=0.5,
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
