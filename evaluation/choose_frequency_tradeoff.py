from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
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


def _hz_from_tag(tag: str) -> int:
    m = re.match(r"^\s*(\d+)\s*hz\s*$", str(tag).lower())
    if not m:
        raise ValueError(f"Bad freq_tag '{tag}', expected like '20hz'")
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_root", type=str, required=True, help="Root to search per_class_metrics_*.csv")
    ap.add_argument("--pattern", type=str, default="**/per_class_metrics_*.csv")
    ap.add_argument("--dataset_name", type=str, default="wisdm_watch_accel")
    ap.add_argument("--tolerance_abs", type=float, default=0.05,
                    help="Absolute F1 tolerance (0..1). Example: 0.05 means accept -5 percentage points.")
    ap.add_argument("--prefer", type=str, default="lowest", choices=["lowest", "highest"],
                    help="Choose lowest or highest frequency among those within tolerance.")
    ap.add_argument("--freq_allow", type=str, default="20hz,10hz,5hz",
                    help="Comma-separated allowed freqs to consider, e.g. '20hz,10hz,5hz'")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    allow = [x.strip() for x in args.freq_allow.split(",") if x.strip()]
    allow_set = set(allow)

    files = sorted(glob.glob(os.path.join(args.log_root, args.pattern), recursive=True))
    if not files:
        raise SystemExit(f"No files found under {args.log_root} with pattern {args.pattern}")

    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue

        # filter dataset if column exists
        if "dataset_name" in df.columns:
            df = df[df["dataset_name"].astype(str) == str(args.dataset_name)]

        # pick freq tag column
        if "freq_tag" in df.columns:
            tag_col = "freq_tag"
        elif "eval_tag" in df.columns:
            tag_col = "eval_tag"
            df = df.rename(columns={"eval_tag": "freq_tag"})
            tag_col = "freq_tag"
        else:
            continue

        df = df[df[tag_col].astype(str).isin(allow_set)]
        if df.empty:
            continue

        required = {"class_id", "class_name", "f1", "freq_tag"}
        if not required.issubset(set(df.columns)):
            continue

        df["source_path"] = p
        dfs.append(df[["class_id", "class_name", "freq_tag", "f1", "prec", "rec", "acc", "source_path"]]
                   if set(["prec", "rec", "acc"]).issubset(df.columns)
                   else df[["class_id", "class_name", "freq_tag", "f1", "source_path"]])

    if not dfs:
        raise SystemExit("No usable per_class_metrics found after filtering. Check log_root/pattern/dataset_name.")

    all_df = pd.concat(dfs, ignore_index=True)

    # Aggregate mean f1 per class per freq across splits/runs
    grp = all_df.groupby(["class_id", "class_name", "freq_tag"], as_index=False)["f1"].mean()

    # Best f1 per class
    best = grp.groupby(["class_id", "class_name"], as_index=False)["f1"].max().rename(columns={"f1": "best_f1"})
    merged = grp.merge(best, on=["class_id", "class_name"], how="left")
    merged["gap"] = merged["best_f1"] - merged["f1"]

    tol = float(args.tolerance_abs)
    eligible = merged[merged["f1"] >= (merged["best_f1"] - tol)].copy()

    # Choose lowest/highest Hz among eligible
    eligible["hz"] = eligible["freq_tag"].apply(_hz_from_tag)

    if args.prefer == "lowest":
        pick = eligible.sort_values(["class_id", "hz"], ascending=[True, True]).groupby("class_id").head(1)
    else:
        pick = eligible.sort_values(["class_id", "hz"], ascending=[True, False]).groupby("class_id").head(1)

    pick = pick.sort_values("class_id").reset_index(drop=True)
    pick["activity"] = pick["class_name"].astype(str).map(WISDM_CODE_TO_NAME).fillna(pick["class_name"].astype(str))

    # Save CSV
    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pick_out = pick[["class_id", "class_name", "activity", "freq_tag", "f1", "best_f1", "gap"]]
    pick_out.to_csv(out_csv, index=False)

    # Save JSON mapping list (index = class_id)
    max_id = int(pick_out["class_id"].max())
    mapping = [""] * (max_id + 1)
    for _, r in pick_out.iterrows():
        mapping[int(r["class_id"])] = str(r["freq_tag"])

    out_json = args.out_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    import json
    with open(out_json, "w") as f:
        json.dump(mapping, f, indent=2)

    print(pick_out.to_string(index=False))
    print(f"\nSaved CSV:  {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
