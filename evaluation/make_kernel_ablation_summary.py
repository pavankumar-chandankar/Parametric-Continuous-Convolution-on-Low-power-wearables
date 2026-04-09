from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--std_root", type=str, required=True)
    ap.add_argument("--cont_root", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--freq_tags", type=str, required=True)
    ap.add_argument("--cont_kernel_json", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    target_freqs = _parse_csv_list(args.freq_tags)

    with open(args.cont_kernel_json, "r") as f:
        obj = json.load(f)
    cont_kernel_map: Dict[str, int] = obj["effective_kernel_by_freq"]

    std_files = sorted(glob.glob(os.path.join(args.std_root, "**", "best_macro_metrics_*.csv"), recursive=True))
    cont_files = sorted(glob.glob(os.path.join(args.cont_root, "**", "best_macro_metrics_*.csv"), recursive=True))

    std_df = _read_best_macro(std_files)
    cont_df = _read_best_macro(cont_files)

    if std_df.empty:
        raise SystemExit("No standard CSVs found")
    if cont_df.empty:
        raise SystemExit("No continuous CSVs found")

    if "dataset_name" in std_df.columns:
        std_df = std_df[std_df["dataset_name"].astype(str) == args.dataset_name]
    if "dataset_name" in cont_df.columns:
        cont_df = cont_df[cont_df["dataset_name"].astype(str) == args.dataset_name]

    if "conv_type" in std_df.columns:
        std_df = std_df[std_df["conv_type"].astype(str).isin(["standard", "standard_multibranch"])]
    if "conv_type" in cont_df.columns:
        cont_df = cont_df[cont_df["conv_type"].astype(str) == "continuous"]

    std_df = std_df[std_df["freq_tag"].astype(str).isin(target_freqs)].copy()
    cont_df = cont_df[cont_df["freq_tag"].astype(str).isin(target_freqs)].copy()

    std_df["ks"] = std_df["source_path"].apply(_infer_ks)

    std_agg = (
        std_df.groupby(["freq_tag", "ks"], as_index=False)["f1_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "std_f1_mean", "std": "std_f1_std", "count": "n"})
    )

    best_std = (
        std_agg.sort_values(["freq_tag", "std_f1_mean", "ks"], ascending=[True, False, True])
        .groupby("freq_tag", as_index=False)
        .first()
        .rename(columns={"ks": "best_std_kernel", "std_f1_mean": "best_std_f1", "std_f1_std": "best_std_f1_std"})
    )

    cont_agg = (
        cont_df.groupby(["freq_tag"], as_index=False)["f1_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "cont_f1", "std": "cont_f1_std", "count": "cont_n"})
    )
    cont_agg["cont_effective_kernel"] = cont_agg["freq_tag"].map(cont_kernel_map)

    std_same = std_agg.rename(columns={"ks": "cont_effective_kernel", "std_f1_mean": "std_same_kernel_f1", "std_f1_std": "std_same_kernel_f1_std"})

    out = pd.merge(cont_agg, best_std, on="freq_tag", how="left")
    out = pd.merge(out, std_same[["freq_tag", "cont_effective_kernel", "std_same_kernel_f1", "std_same_kernel_f1_std"]], on=["freq_tag", "cont_effective_kernel"], how="left")

    out["delta_cont_minus_best_std"] = out["cont_f1"] - out["best_std_f1"]
    out["delta_cont_minus_same_kernel_std"] = out["cont_f1"] - out["std_same_kernel_f1"]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("[OK] wrote", args.out_csv)
    print(out.sort_values("freq_tag").to_string(index=False))


if __name__ == "__main__":
    main()
