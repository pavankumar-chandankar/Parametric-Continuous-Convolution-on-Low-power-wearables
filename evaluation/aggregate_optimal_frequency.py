from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import pandas as pd


def _find_csvs(log_root: str, pattern: str) -> List[str]:
    path = os.path.join(log_root, pattern)
    return sorted(glob.glob(path, recursive=True))


def _normalize_freq_tag(df: pd.DataFrame) -> pd.DataFrame:
    if "freq_tag" not in df.columns and "eval_tag" in df.columns:
        df = df.rename(columns={"eval_tag": "freq_tag"})
    if "freq_tag" not in df.columns:
        raise ValueError(f"Missing freq_tag/eval_tag in CSV columns: {list(df.columns)}")
    df["freq_tag"] = df["freq_tag"].astype(str)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-class metrics and select best frequency per activity.")
    ap.add_argument("--log_root", type=str, default=os.path.join("logs", "deepconvlstm"))
    ap.add_argument(
        "--pattern",
        type=str,
        default="**/per_class_metrics_*.csv",
        help="Glob pattern relative to log_root (recursive).",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "acc", "prec", "rec"],
        help="Metric to maximize per class.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join("logs", "deepconvlstm", "best_frequency_per_activity.csv"),
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("logs", "deepconvlstm", "best_frequency_per_activity.json"),
        help="JSON mapping: list where index=class_id, value=freq_tag.",
    )
    ap.add_argument(
        "--filter_dataset",
        type=str,
        default="",
        help="Optional. Keep only rows with dataset_name == this value.",
    )
    args = ap.parse_args()

    csv_files = _find_csvs(args.log_root, args.pattern)
    if not csv_files:
        print(f"[WARN] No CSV files found under {args.log_root} with pattern {args.pattern}")
        return

    all_dfs = []
    for p in csv_files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue
        if df.empty:
            continue
        try:
            df = _normalize_freq_tag(df)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue
        all_dfs.append(df)

    if not all_dfs:
        print("[WARN] No usable per-class metrics CSVs found.")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)

    required_cols = {"class_id", "class_name", "freq_tag", "acc", "prec", "rec", "f1"}
    missing = required_cols.difference(all_df.columns)
    if missing:
        raise ValueError(f"Missing columns in aggregated DataFrame: {missing}")

    all_df["class_id"] = all_df["class_id"].astype(int)
    all_df["class_name"] = all_df["class_name"].astype(str)

    if args.filter_dataset.strip():
        if "dataset_name" not in all_df.columns:
            raise ValueError("--filter_dataset was provided, but CSVs have no dataset_name column.")
        all_df = all_df[all_df["dataset_name"].astype(str) == args.filter_dataset.strip()].copy()
        if all_df.empty:
            print(f"[WARN] No rows remain after dataset filter: {args.filter_dataset}")
            return

    metric_cols = ["acc", "prec", "rec", "f1"]
    summary = (
        all_df.groupby(["class_id", "class_name", "freq_tag"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )

    idx_best = summary.groupby("class_id")[args.metric].idxmax()
    best_per_activity = summary.loc[idx_best].sort_values("class_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    best_per_activity.to_csv(args.out_csv, index=False)

    max_id = int(best_per_activity["class_id"].max())
    mapping = ["25hz"] * (max_id + 1)
    for _, r in best_per_activity.iterrows():
        mapping[int(r["class_id"])] = str(r["freq_tag"])

    with open(args.out_json, "w") as f:
        json.dump(mapping, f, indent=2)

    print("Best frequency per activity (by mean metric across splits):")
    print(best_per_activity.to_string(index=False))
    print(f"Saved CSV to:  {args.out_csv}")
    print(f"Saved JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
