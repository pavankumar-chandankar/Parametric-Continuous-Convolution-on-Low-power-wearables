#!/usr/bin/env python3
"""
Create a per-activity F1 summary table:
Activity | 5 Hz (avg across splits) | 10 Hz | 20 Hz

Reads:
  per_class_metrics_loso_sbj_<k>.csv        -> 20hz
  per_class_metrics_loso_sbj_<k>_10hz.csv   -> 10hz
  per_class_metrics_loso_sbj_<k>_5hz.csv    -> 5hz
"""

import argparse
import glob
import os
import re
import pandas as pd


WISDM_CODE_TO_NAME = {
    "A": "Walking",
    "B": "Jogging",
    "C": "Stairs",
    "D": "Sitting",
    "E": "Standing",
    "F": "Typing",
    "G": "Brushing Teeth",
    "H": "Eating Soup",
    "I": "Eating Chips",
    "J": "Eating Pasta",
    "K": "Drinking from Cup",
    "L": "Eating Sandwich",
    "M": "Kicking (Soccer Ball)",
    "O": "Playing Catch w/Tennis Ball",
    "P": "Dribbling (Basketball)",
    "Q": "Writing",
    "R": "Clapping",
    "S": "Folding Clothes",
}


def infer_freq_from_filename(path: str) -> str:
    b = os.path.basename(path).lower()
    if b.endswith("_5hz.csv"):
        return "5hz"
    if b.endswith("_10hz.csv"):
        return "10hz"
    # default: base evaluation at window_size=20 -> treat as 20hz
    return "20hz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="logs/deepconvlstm/experiments/continuous_nogamma/wisdm_watch_accel",
        help="Root folder containing 20hz/loso_sbj_*/per_class_metrics_*.csv",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Output CSV path (default: <root>/per_activity_f1_summary.csv)",
    )
    ap.add_argument(
        "--as_percent",
        action="store_true",
        help="If set, convert F1 from [0,1] to percentage [0,100].",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    pattern = os.path.join(root, "20hz", "loso_sbj_*", "per_class_metrics_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No per_class_metrics_*.csv found with pattern: {pattern}")

    rows = []
    for p in files:
        df = pd.read_csv(p)
        if df.empty:
            continue

        if "class_id" not in df.columns or "class_name" not in df.columns or "f1" not in df.columns:
            continue

        freq = infer_freq_from_filename(p)

        d = df[["class_id", "class_name", "f1"]].copy()
        d["freq"] = freq
        d["source_path"] = p
        rows.append(d)

    if not rows:
        raise SystemExit("Found files, but none had the required columns: class_id, class_name, f1.")

    all_df = pd.concat(rows, ignore_index=True)

    # Mean F1 across splits for each class and frequency
    agg = (
        all_df.groupby(["class_id", "class_name", "freq"], as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "f1_mean"})
    )

    if args.as_percent:
        agg["f1_mean"] = agg["f1_mean"] * 100.0

    # Add human-readable activity name
    agg["Activity"] = agg["class_name"].astype(str).map(WISDM_CODE_TO_NAME).fillna(agg["class_name"].astype(str))

    # Pivot to Activity × {5hz,10hz,20hz}
    pivot = agg.pivot_table(index=["class_id", "class_name", "Activity"], columns="freq", values="f1_mean")
    pivot = pivot.reset_index()

    # Ensure all columns exist
    for c in ["5hz", "10hz", "20hz"]:
        if c not in pivot.columns:
            pivot[c] = pd.NA

    # Order + rename columns
    pivot = pivot[["Activity", "5hz", "10hz", "20hz"]].copy()
    pivot = pivot.rename(
        columns={
            "5hz": "5 Hz (avg across splits)",
            "10hz": "10 Hz (avg across splits)",
            "20hz": "20 Hz (avg across splits)",
        }
    )

    out_csv = args.out_csv.strip() or os.path.join(root, "per_activity_f1_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pivot.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
