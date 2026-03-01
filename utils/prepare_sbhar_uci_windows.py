import argparse
import json
from pathlib import Path

import numpy as np

SIGS = [
    ("body_acc_x", "body_acc_x_{}.txt"),
    ("body_acc_y", "body_acc_y_{}.txt"),
    ("body_acc_z", "body_acc_z_{}.txt"),
    ("body_gyro_x", "body_gyro_x_{}.txt"),
    ("body_gyro_y", "body_gyro_y_{}.txt"),
    ("body_gyro_z", "body_gyro_z_{}.txt"),
    ("total_acc_x", "total_acc_x_{}.txt"),
    ("total_acc_y", "total_acc_y_{}.txt"),
    ("total_acc_z", "total_acc_z_{}.txt"),
]

def load_vec(p: Path) -> np.ndarray:
    # one int per line
    return np.loadtxt(p, dtype=np.int32)

def load_signal(p: Path) -> np.ndarray:
    # rows=windows, cols=128 samples
    return np.loadtxt(p, dtype=np.float32)

def load_split(root: Path, split: str):
    split_dir = root / split
    inertial = split_dir / "Inertial Signals"
    subj = load_vec(split_dir / f"subject_{split}.txt")
    y = load_vec(split_dir / f"y_{split}.txt") - 1  # to 0..5

    sig_arrays = []
    for _, tpl in SIGS:
        sig_arrays.append(load_signal(inertial / tpl.format(split)))  # (N,128)

    # stack into (N,128,9)
    X = np.stack(sig_arrays, axis=-1)  # (N,128,9)
    return X, y.astype(np.int64), subj.astype(np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="data/sbhar/UCI HAR Dataset")
    ap.add_argument("--out", type=str, default="data/sbhar/processed/windows_50hz")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # labels
    lab_txt = (src / "activity_labels.txt").read_text(encoding="utf-8").strip().splitlines()
    label_map = {int(line.split()[0]) - 1: line.split()[1] for line in lab_txt}  # 0..5
    (out / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    Xtr, ytr, str_ = load_split(src, "train")
    Xte, yte, ste_ = load_split(src, "test")

    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)
    s = np.concatenate([str_, ste_], axis=0)

    # write per-subject
    subjects = np.unique(s)
    rows = []
    for sid in subjects:
        idx = np.where(s == sid)[0]
        Xi = X[idx]
        yi = y[idx]
        np.savez_compressed(out / f"sbj_{sid}.npz", x=Xi, y=yi)
        rows.append((int(sid), int(len(idx))))

    # simple summary
    rows.sort(key=lambda t: t[0])
    summary = "subject,windows\n" + "\n".join([f"{sid},{n}" for sid, n in rows]) + "\n"
    (out / "subject_window_counts.csv").write_text(summary, encoding="utf-8")

    print(f"[OK] wrote {len(subjects)} subject files to: {out}")
    print(f"[OK] channels=9, window_len=128, classes=6, fs=50Hz")
    print(f"[OK] label map: {out/'label_map.json'}")
    print(f"[OK] subject counts: {out/'subject_window_counts.csv'}")

if __name__ == "__main__":
    main()
