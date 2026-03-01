import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import pandas as pd


REQ_COLS = [
    "subject_id",
    "activity",
    "timestamp",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
]


def _read_subject_id_fast(csv_path: str) -> int:
    df0 = pd.read_csv(csv_path, nrows=1)
    if "subject_id" not in df0.columns:
        raise RuntimeError(f"'subject_id' column not found in {csv_path}. Columns={list(df0.columns)}")
    return int(df0["subject_id"].iloc[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="Folder with raw WISDM watch CSVs like phone_subject_1600.csv ...",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output folder for WEAR-format CSVs (sbj_0.csv ...).",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern inside --in-dir (default: *.csv).",
    )
    args = p.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    if not files:
        raise RuntimeError(f"No CSV files found in: {in_dir} (pattern={args.pattern})")

    # Build stable mapping subject_id -> sbj_index (0..N-1)
    subj_pairs: List[Tuple[int, str]] = []
    for f in files:
        sid = _read_subject_id_fast(f)
        subj_pairs.append((sid, f))

    unique_sids = sorted({sid for sid, _ in subj_pairs})
    sid_to_idx: Dict[int, int] = {sid: i for i, sid in enumerate(unique_sids)}

    manifest = {
        "in_dir": in_dir,
        "out_dir": out_dir,
        "num_subjects": len(unique_sids),
        "subject_ids": unique_sids,
        "mapping": {str(sid): f"sbj_{sid_to_idx[sid]}" for sid in unique_sids},
        "files": [],
    }

    print(f"Found {len(unique_sids)} unique subjects: {unique_sids[0]} .. {unique_sids[-1]}")
    print(f"Writing WEAR-format CSVs to: {out_dir}")

    for sid, f in sorted(subj_pairs, key=lambda x: (x[0], x[1])):
        sbj_idx = sid_to_idx[sid]
        out_path = os.path.join(out_dir, f"sbj_{sbj_idx}.csv")

        df = pd.read_csv(f, low_memory=False)

        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing columns {missing} in {f}. Columns={list(df.columns)}")

        # Sort by timestamp to preserve order, then drop timestamp from training features
        df = df.sort_values("timestamp", kind="mergesort")

        out_df = pd.DataFrame(
            {
                "subject": df["subject_id"].astype(int),
                "acc_x": df["acc_x"].astype(float),
                "acc_y": df["acc_y"].astype(float),
                "acc_z": df["acc_z"].astype(float),
                "gyr_x": df["gyr_x"].astype(float),
                "gyr_y": df["gyr_y"].astype(float),
                "gyr_z": df["gyr_z"].astype(float),
                "label": df["activity"].astype(str),
            }
        )

        out_df.to_csv(out_path, index=False)

        manifest["files"].append(
            {
                "raw_file": os.path.basename(f),
                "subject_id": int(sid),
                "sbj": f"sbj_{sbj_idx}",
                "out_file": os.path.basename(out_path),
                "rows": int(len(out_df)),
            }
        )

        print(f"Wrote {out_path} (subject_id={sid}, rows={len(out_df)})")

    mani_path = os.path.join(out_dir, "manifest_wisdm_phone_wearformat.json")
    with open(mani_path, "w", encoding="utf-8") as g:
        json.dump(manifest, g, indent=2)
    print(f"Saved manifest: {mani_path}")


if __name__ == "__main__":
    main()

