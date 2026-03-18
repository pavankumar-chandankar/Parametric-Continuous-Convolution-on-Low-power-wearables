from __future__ import annotations

import argparse
import glob
import os
import pandas as pd


# Must match your annotation generator label order (ACTIVITIES list)
ACTIVITIES = ["walking", "running", "sitting", "standing", "lying", "jumping", "climbingup", "climbingdown"]
ACT_PRETTY = {
    "walking": "Walking",
    "running": "Running",
    "sitting": "Sitting",
    "standing": "Standing",
    "lying": "Lying",
    "jumping": "Jumping",
    "climbingup": "Climbing Up",
    "climbingdown": "Climbing Down",
}


def infer_freq_tag_from_filename(path: str) -> str:
    b = os.path.basename(path).lower()
    if b.endswith("_25hz.csv"):
        return "25hz"
    if b.endswith("_12hz.csv"):
        return "12hz"
    if b.endswith("_6hz.csv"):
        return "6hz"
    # base evaluation (no suffix) corresponds to 50hz for this dataset
    return "50hz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Experiment root containing 50hz/loso_sbj_*/per_class_metrics_*.csv",
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
        help="If set, multiply F1 by 100.",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    pattern = os.path.join(root, "50hz", "loso_sbj_*", "per_class_metrics_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No per_class_metrics_*.csv found: {pattern}")

    rows = []
    for p in files:
        df = pd.read_csv(p)
        if df.empty:
            continue

        # required columns
        if not {"class_id", "f1"}.issubset(df.columns):
            continue

        # figure out activity name
        if "class_name" in df.columns:
            cname = df["class_name"].astype(str)
        else:
            # fallback: map class_id to ACTIVITY list
            cname = df["class_id"].apply(lambda i: ACTIVITIES[int(i)] if int(i) < len(ACTIVITIES) else f"class_{int(i)}").astype(str)

        freq_tag = infer_freq_tag_from_filename(p)

        tmp = pd.DataFrame({
            "class_id": df["class_id"].astype(int),
            "class_name": cname,
            "freq_tag": freq_tag,
            "f1": df["f1"].astype(float),
        })
        rows.append(tmp)

    if not rows:
        raise SystemExit("No usable per_class_metrics found (missing class_id/f1 columns).")

    all_df = pd.concat(rows, ignore_index=True)

    # average across splits for each class and frequency
    agg = (
        all_df.groupby(["class_id", "class_name", "freq_tag"], as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "f1_mean"})
    )

    if args.as_percent:
        agg["f1_mean"] = agg["f1_mean"] * 100.0

    # pretty activity names
    agg["Activity"] = agg["class_name"].map(ACT_PRETTY).fillna(agg["class_name"])

    pivot = agg.pivot_table(index=["class_id", "Activity"], columns="freq_tag", values="f1_mean").reset_index()

    # ensure all columns exist
    for c in ["50hz", "25hz", "12hz", "6hz"]:
        if c not in pivot.columns:
            pivot[c] = pd.NA

    pivot = pivot.sort_values("class_id").reset_index(drop=True)

    out = pivot[["Activity", "50hz", "25hz", "12hz", "6hz"]].rename(
        columns={
            "50hz": "50 Hz (avg across splits)",
            "25hz": "25 Hz (avg across splits)",
            "12hz": "12 Hz (avg across splits)",
            "6hz": "6 Hz (avg across splits)",
        }
    )

    out_csv = args.out_csv.strip() or os.path.join(root, "per_activity_f1_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
