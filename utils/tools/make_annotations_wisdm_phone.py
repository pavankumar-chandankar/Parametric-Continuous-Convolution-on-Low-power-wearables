import argparse
import json
import os
from typing import List, Dict

import pandas as pd


def collect_all_labels(csv_paths: List[str], label_col: str) -> List[str]:
    labels = set()
    for p in csv_paths:
        df = pd.read_csv(p, usecols=[label_col])
        labels.update(str(x) for x in df[label_col].dropna().unique())
    return sorted(labels)


def build_annotations_for_subject(csv_path: str, label_col: str, label_to_id: Dict[str, int], sampling_rate: float):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise RuntimeError(f"'{label_col}' column not found in {csv_path}. Columns: {list(df.columns)}")

    labels = df[label_col].tolist()
    annotations = []
    if not labels:
        return annotations

    current_label = labels[0]
    start_idx = 0

    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != current_label:
            end_idx = idx - 1
            if pd.isna(current_label):
                pass
            else:
                lab = str(current_label)
                if lab in label_to_id:
                    t_start = start_idx / sampling_rate
                    t_end = (end_idx + 1) / sampling_rate
                    annotations.append(
                        {
                            "segment": [float(t_start), float(t_end)],
                            "label_id": int(label_to_id[lab]),
                        }
                    )
            if idx < len(labels):
                current_label = labels[idx]
                start_idx = idx

    return annotations


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sens-folder", type=str, required=True, help="Folder with sbj_*.csv (WEAR-format).")
    p.add_argument("--anno-folder", type=str, required=True, help="Output folder for loso_sbj_*.json")
    p.add_argument("--sampling-rate", type=float, default=20.0, help="Sampling rate used for time conversion.")
    p.add_argument("--label-col", type=str, default="label", help="Label column name (default: label)")
    args = p.parse_args()

    sens_folder = args.sens_folder
    anno_folder = args.anno_folder
    sampling_rate = float(args.sampling_rate)
    label_col = args.label_col

    os.makedirs(anno_folder, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(sens_folder) if f.endswith(".csv") and f.startswith("sbj_"))
    if not csv_files:
        raise RuntimeError(f"No sbj_*.csv files found in {sens_folder}")

    stems = [os.path.splitext(f)[0] for f in csv_files]
    main_subjects = sorted(
        s for s in stems if s.startswith("sbj_") and s.count("_") == 1 and s[4:].isdigit()
    )

    if not main_subjects:
        raise RuntimeError(f"No LOSO subjects detected in {sens_folder}. Found stems: {stems[:10]}...")

    csv_paths = [os.path.join(sens_folder, s + ".csv") for s in main_subjects]
    label_dict = collect_all_labels(csv_paths, label_col=label_col)
    label_to_id = {lab: i for i, lab in enumerate(label_dict)}

    print(f"Detected {len(main_subjects)} subjects: {main_subjects[0]} .. {main_subjects[-1]}")
    print(f"Detected labels ({len(label_dict)}): {label_dict}")
    print(f"Writing annotations to: {anno_folder}")

    for val_main in main_subjects:
        database = {}
        for s in main_subjects:
            subset = "Validation" if s == val_main else "Training"
            database[s] = {
                "subset": subset,
                "annotations": build_annotations_for_subject(
                    csv_path=os.path.join(sens_folder, s + ".csv"),
                    label_col=label_col,
                    label_to_id=label_to_id,
                    sampling_rate=sampling_rate,
                ),
            }

        anno_obj = {"database": database, "label_dict": label_dict}

        out_path = os.path.join(anno_folder, f"loso_{val_main}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(anno_obj, f)
        print(f"Wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

