from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

ACTIVITIES = ["walking", "running", "sitting", "standing", "lying", "jumping", "climbingup", "climbingdown"]
PAT = re.compile(r"^proband(\d+)_([a-z]+)\.csv$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_dir", type=str, required=True, help="Processed folder with probandX_activity.csv")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for loso_sbj_*.json")
    ap.add_argument("--sampling_rate", type=float, default=50.0)
    args = ap.parse_args()

    sens_dir = os.path.abspath(args.sens_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(sens_dir) if f.endswith(".csv")])
    items: List[Tuple[int, str, str]] = []  # (pid, act, stem)
    for fn in files:
        m = PAT.match(fn)
        if not m:
            continue
        pid = int(m.group(1))
        act = m.group(2)
        stem = os.path.splitext(fn)[0]
        items.append((pid, act, stem))

    if not items:
        raise SystemExit(f"No proband CSVs found in {sens_dir}")

    acts_found = sorted(set(a for _, a, _ in items))
    missing = [a for a in ACTIVITIES if a not in acts_found]
    if missing:
        print(f"[WARN] Missing activities in folder: {missing}")
    extra = [a for a in acts_found if a not in ACTIVITIES]
    if extra:
        print(f"[WARN] Extra activities in folder: {extra}")

    probands = sorted(set(pid for pid, _, _ in items))
    if len(probands) != 15:
        print(f"[WARN] Expected 15 probands, found {len(probands)}: {probands}")

    # Stable label mapping
    labels = [a for a in ACTIVITIES if a in acts_found] + [a for a in acts_found if a not in ACTIVITIES]
    label_dict = {lab: i for i, lab in enumerate(labels)}

    # Duration per file (seconds): rows / fs
    durations: Dict[str, float] = {}
    for pid, act, stem in items:
        path = os.path.join(sens_dir, f"{stem}.csv")
        with open(path, "r") as f:
            n_lines = sum(1 for _ in f)
        n_rows = max(0, n_lines - 1)
        durations[stem] = float(n_rows) / float(args.sampling_rate) if args.sampling_rate > 0 else 0.0

    # LOSO over probands: split_idx aligns with proband order
    for split_idx, holdout_pid in enumerate(probands):
        db = {}
        for pid, act, stem in items:
            subset = "Validation" if pid == holdout_pid else "Training"
            db[stem] = {
                "subset": subset,
                "annotations": [
                    {
                        "segment": [0.0, durations.get(stem, 0.0)],
                        "label": act,
                        "label_id": int(label_dict[act]),
                    }
                ],
            }

        out = {
            "version": "1.0",
            "dataset": "realworld_waist_accel",
            "sampling_rate_hz": float(args.sampling_rate),
            "label_dict": label_dict,
            "labels": labels,
            "database": db,
        }

        out_path = os.path.join(out_dir, f"loso_sbj_{split_idx}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] wrote {out_path} (holdout proband{holdout_pid})")

    print("\nDone.")


if __name__ == "__main__":
    main()
