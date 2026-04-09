from __future__ import annotations
import glob
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

KS_RE = re.compile(r"/ks(\d+)(/|$)")


def infer_ks(path: str) -> int:
    m = KS_RE.search(path.replace("\\", "/"))
    if not m:
        raise ValueError(f"Cannot infer kernel size from path: {path}")
    return int(m.group(1))


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
    return {freq: int(obj[freq]) for freq in target_freqs}


def load_manual_cont_stats(path: str, target_freqs: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["freq_tag"].astype(str).isin(target_freqs)].copy()
    return df.rename(columns={"mean": "cont_f1_mean", "std": "cont_f1_std"})[
        ["freq_tag", "cont_f1_mean", "cont_f1_std"]
    ]


def prepare_standard(std_root: str, dataset_name: str, target_freqs: List[str], keep_kernels: List[int]) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(std_root, "**", "best_macro_metrics_*.csv"), recursive=True))
    df = read_best_macro(files)
    if df.empty:
        raise SystemExit(f"No usable standard CSVs under {std_root}")
    if "dataset_name" in df.columns:
        df = df[df["dataset_name"].astype(str) == dataset_name]
    if "conv_type" in df.columns:
        df = df[df["conv_type"].astype(str).isin(["standard", "standard_multibranch"])]
    df = df[df["freq_tag"].astype(str).isin(target_freqs)].copy()
    df["ks"] = df["source_path"].apply(infer_ks)
    df = df[df["ks"].isin(keep_kernels)].copy()
    if df.empty:
        raise SystemExit(f"No standard rows left after filtering for {dataset_name}")
    agg = (
        df.groupby(["freq_tag", "ks"], as_index=False)["f1_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "std_f1_mean", "std": "std_f1_std", "count": "n"})
    )
    agg["std_f1_mean"] = agg["std_f1_mean"] / 100.0
    agg["std_f1_std"] = agg["std_f1_std"] / 100.0
    return agg


def prepare_continuous(
    target_freqs: List[str],
    cont_kernel_json: str,
    cont_stats_csv: Optional[str] = None,
    cont_root: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    kernel_map = load_cont_kernel_map(cont_kernel_json, target_freqs)
    if cont_stats_csv:
        agg = load_manual_cont_stats(cont_stats_csv, target_freqs)
    else:
        files = sorted(glob.glob(os.path.join(cont_root, "**", "best_macro_metrics_*.csv"), recursive=True))
        df = read_best_macro(files)
        if df.empty:
            raise SystemExit(f"No usable continuous CSVs under {cont_root}")
        if "dataset_name" in df.columns and dataset_name is not None:
            df = df[df["dataset_name"].astype(str) == dataset_name]
        if "conv_type" in df.columns:
            df = df[df["conv_type"].astype(str) == "continuous"]
        df = df[df["freq_tag"].astype(str).isin(target_freqs)].copy()
        if df.empty:
            raise SystemExit(f"No continuous rows left after filtering for {dataset_name}")
        agg = (
            df.groupby(["freq_tag"], as_index=False)["f1_mean"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "cont_f1_mean", "std": "cont_f1_std", "count": "n"})
        )
        agg["cont_f1_mean"] = agg["cont_f1_mean"] / 100.0
        agg["cont_f1_std"] = agg["cont_f1_std"] / 100.0

    agg["effective_kernel"] = agg["freq_tag"].map(kernel_map)
    return agg


def plot_dataset(
    ax_top,
    ax_bottom,
    title: str,
    std_agg: pd.DataFrame,
    cont_agg: pd.DataFrame,
    target_freqs: List[str],
    keep_kernels: List[int],
    color_map: Dict[str, str],
    cont_row_ylim: tuple[float, float],
):
    x_pos = {k: i for i, k in enumerate(keep_kernels)}
    x_ticks = list(range(len(keep_kernels)))
    x_ticklabels = [str(k) for k in keep_kernels]

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
            [x_pos[k]], [y], yerr=[s],
            fmt="o",
            markersize=8.8,
            capsize=4.0,
            linestyle="none",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            markeredgecolor="black",
            markeredgewidth=1.0,
            zorder=5,
        )

    for freq in target_freqs:
        sub = std_agg[std_agg["freq_tag"] == freq].sort_values("ks")
        if sub.empty:
            continue
        xs = [x_pos[int(k)] for k in sub["ks"].tolist()]
        color = color_map[freq]
        ax_bottom.plot(
            xs, sub["std_f1_mean"],
            marker="o", linewidth=1.9, markersize=4.6, color=color
        )
        ax_bottom.errorbar(
            xs, sub["std_f1_mean"], yerr=sub["std_f1_std"].fillna(0.0),
            fmt="none", capsize=2.5, elinewidth=1.0, alpha=0.45, color=color
        )

    ax_top.set_title(title, pad=14)
    ax_top.set_ylim(cont_row_ylim[0], cont_row_ylim[1])
    ax_top.set_yticks([0.1, 0.4, 0.7, 1.0])
    ax_top.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax_top.grid(axis="x", visible=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax_bottom.set_xticks(x_ticks)
    ax_bottom.set_xticklabels(x_ticklabels)
    ax_bottom.set_ylim(0, 1.0)
    ax_bottom.set_yticks([i / 10.0 for i in range(0, 11)])
    ax_bottom.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax_bottom.grid(axis="x", visible=False)
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)
    ax_bottom.set_xlabel("Temporal kernel size")


def main():
    keep_kernels = [1, 3, 5, 7, 9, 11]
    cont_row_ylim = (0.10, 1.00)

    wear_freqs = ["50hz", "25hz", "12hz", "6hz"]
    rw_freqs = ["50hz", "25hz", "12hz", "6hz"]
    wisdm_freqs = ["20hz", "10hz", "5hz"]

    wear_std = prepare_standard(
        "/home/g263412/GAMMA/logs/deepconvlstm/experiments/kernel_sweep_standard/wear_nogamma",
        "wear", wear_freqs, keep_kernels
    )
    wear_cont = prepare_continuous(
        wear_freqs,
        "/home/g263412/GAMMA/visualization/wear_continuous_effective_kernel_map.json",
        cont_stats_csv="/home/g263412/GAMMA/visualization/wear_continuous_stats_manual.csv",
        dataset_name="wear",
    )

    rw_std = prepare_standard(
        "/home/g263412/GAMMA/logs/deepconvlstm/experiments/kernel_sweep_standard/realworld_waist_accel_nogamma",
        "realworld_waist_accel", rw_freqs, keep_kernels
    )
    rw_cont = prepare_continuous(
        rw_freqs,
        "/home/g263412/GAMMA/visualization/realworld_continuous_effective_kernel_map.json",
        cont_root="/home/g263412/GAMMA/logs/deepconvlstm/experiments/continuous_nogamma/realworld_waist_accel/50hz",
        dataset_name="realworld_waist_accel",
    )

    wisdm_std = prepare_standard(
        "/home/g263412/GAMMA/logs/deepconvlstm/experiments/kernel_sweep_standard/wisdm_watch_accel_nogamma",
        "wisdm_watch_accel", wisdm_freqs, keep_kernels
    )
    wisdm_cont = prepare_continuous(
        wisdm_freqs,
        "/home/g263412/GAMMA/visualization/wisdm_watch_continuous_effective_kernel_map.json",
        cont_root="/home/g263412/GAMMA/logs/deepconvlstm/20hz",
        dataset_name="wisdm_watch_accel",
    )

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 12.0,
        "axes.labelsize": 12.5,
        "axes.titlesize": 13.0,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
    })

    fig, axes = plt.subplots(
        2, 3, figsize=(12.8, 5.8),
        gridspec_kw={"height_ratios": [1.0, 2.2]},
        sharey='row'
    )

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    wear_colors = {freq: cycle[i % len(cycle)] for i, freq in enumerate(wear_freqs)}
    rw_colors = {freq: cycle[i % len(cycle)] for i, freq in enumerate(rw_freqs)}
    wisdm_colors = {freq: cycle[i % len(cycle)] for i, freq in enumerate(wisdm_freqs)}

    plot_dataset(axes[0, 0], axes[1, 0], "WEAR", wear_std, wear_cont, wear_freqs, keep_kernels, wear_colors, cont_row_ylim)
    plot_dataset(axes[0, 1], axes[1, 1], "RealWorld HAR", rw_std, rw_cont, rw_freqs, keep_kernels, rw_colors, cont_row_ylim)
    plot_dataset(axes[0, 2], axes[1, 2], "WISDM-watch", wisdm_std, wisdm_cont, wisdm_freqs, keep_kernels, wisdm_colors, cont_row_ylim)

    for ax in axes.ravel():
        ax.set_ylabel("")

    fig.text(0.028, 0.52, "Macro-F1", rotation=90, va="center", ha="center", fontsize=13)

    # move these farther left so they no longer collide with the plots
    fig.text(0.002, 0.79, "Continuous conv", rotation=90, va="center", ha="left", fontsize=13)
    fig.text(0.002, 0.34, "Standard DeepConvLSTM", rotation=90, va="center", ha="left", fontsize=13)

    # keep legends as they are now
    wear_handles = [Line2D([0], [0], color=wear_colors[f], marker="o", linewidth=1.9, markersize=4.6, label=f) for f in wear_freqs]
    wisdm_handles = [Line2D([0], [0], color=wisdm_colors[f], marker="o", linewidth=1.9, markersize=4.6, label=f) for f in wisdm_freqs]

    fig.legend(
        handles=wear_handles,
        loc="lower center",
        bbox_to_anchor=(0.37, 0.00),
        ncol=4,
        frameon=False,
        title="WEAR / RealWorld"
    )
    fig.legend(
        handles=wisdm_handles,
        loc="lower center",
        bbox_to_anchor=(0.79, 0.00),
        ncol=3,
        frameon=False,
        title="WISDM-watch"
    )

    out_png = "/home/g263412/GAMMA/visualization/kernel_sweep_combined_3dataset_paper_v4.png"
    out_pdf = "/home/g263412/GAMMA/visualization/kernel_sweep_combined_3dataset_paper_v4.pdf"

    fig.tight_layout(rect=[0.06, 0.10, 0.99, 0.98])
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print("[OK] wrote", out_png)
    print("[OK] wrote", out_pdf)


if __name__ == "__main__":
    main()
