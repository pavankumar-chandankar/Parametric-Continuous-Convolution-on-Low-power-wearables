from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

KS_RE = re.compile(r"/ks(\d+)(/|$)")


def _infer_ks(path: str) -> int:
    m = KS_RE.search(path.replace("\\", "/"))
    if not m:
        raise ValueError(f"Cannot infer kernel size from path: {path}")
    return int(m.group(1))


def _read_best_macro(files: List[str]) -> pd.DataFrame:
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
    return pd.concat(rows, ignore_index=True)


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or str(s).strip() == "":
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _load_cont_kernel_map(path: str, target_freqs: List[str]) -> Dict[str, int]:
    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "effective_kernel_by_freq" in obj:
        obj = obj["effective_kernel_by_freq"]

    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {path}")

    out = {}
    for freq in target_freqs:
        if freq not in obj:
            raise KeyError(f"Missing frequency {freq} in {path}")
        out[freq] = int(obj[freq])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--std_root", type=str, required=True)
    ap.add_argument("--cont_root", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--freq_tags", type=str, required=True)
    ap.add_argument("--kernel_values", type=str, default="")
    ap.add_argument("--cont_kernel_json", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--title", type=str, default="")
    args = ap.parse_args()

    target_freqs = _parse_csv_list(args.freq_tags)
    keep_kernels = _parse_int_list(args.kernel_values)
    cont_kernel_map = _load_cont_kernel_map(args.cont_kernel_json, target_freqs)

    std_files = sorted(glob.glob(os.path.join(args.std_root, "**", "best_macro_metrics_*.csv"), recursive=True))
    std_df = _read_best_macro(std_files)
    if std_df.empty:
        raise SystemExit(f"No usable standard best_macro_metrics under {args.std_root}")

    if "dataset_name" in std_df.columns:
        std_df = std_df[std_df["dataset_name"].astype(str) == args.dataset_name]
    if "conv_type" in std_df.columns:
        std_df = std_df[std_df["conv_type"].astype(str).isin(["standard", "standard_multibranch"])]

    std_df = std_df[std_df["freq_tag"].astype(str).isin(target_freqs)].copy()
    std_df["ks"] = std_df["source_path"].apply(_infer_ks)

    if keep_kernels is not None:
        std_df = std_df[std_df["ks"].isin(keep_kernels)].copy()

    if std_df.empty:
        raise SystemExit("No standard rows left after filtering.")

    std_agg = (
        std_df.groupby(["freq_tag", "ks"], as_index=False)["f1_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "std_f1_mean", "std": "std_f1_std", "count": "n"})
    )

    cont_files = sorted(glob.glob(os.path.join(args.cont_root, "**", "best_macro_metrics_*.csv"), recursive=True))
    cont_df = _read_best_macro(cont_files)
    if cont_df.empty:
        raise SystemExit(f"No usable continuous best_macro_metrics under {args.cont_root}")

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
    cont_agg["effective_kernel"] = cont_agg["freq_tag"].map(cont_kernel_map)

    same_kernel_std = std_agg.rename(columns={"ks": "effective_kernel"})
    summary = pd.merge(cont_agg, same_kernel_std, on=["freq_tag", "effective_kernel"], how="left")
    summary["delta_cont_minus_std_at_same_kernel"] = summary["cont_f1_mean"] - summary["std_f1_mean"]

    if args.out_csv.strip():
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary.sort_values(["freq_tag"]).to_csv(args.out_csv, index=False)

    out_dir = os.path.dirname(args.out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    ks_unique = sorted(std_agg["ks"].unique().tolist())

    for freq in target_freqs:
        sub = std_agg[std_agg["freq_tag"] == freq].sort_values("ks")
        if sub.empty:
            continue

        line = ax.errorbar(
            sub["ks"],
            sub["std_f1_mean"],
            yerr=sub["std_f1_std"].fillna(0.0),
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=5.5,
            capsize=3,
            label=f"Standard @ {freq}",
        )
        color = line.lines[0].get_color()

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
                markersize=8.5,
                capsize=3,
                linestyle="none",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=f"Continuous @ {freq}",
                zorder=6,
            )

    ax.set_xlabel("Temporal kernel size")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(ks_unique)
    ax.grid(True, alpha=0.25)
    ax.set_title(args.title.strip() or f"Kernel sweep vs continuous kernels ({args.dataset_name})")
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")

    print("[OK] wrote", args.out_png)
    print(summary.sort_values(["freq_tag"]).to_string(index=False))


if __name__ == "__main__":
    main()
