from __future__ import annotations

import argparse
import os
import re
import zipfile
from typing import Dict, List

import pandas as pd


ACTIVITIES = [
    "walking", "running", "sitting", "standing", "lying",
    "jumping", "climbingup", "climbingdown",
]

ACC_FILE_RE_TPL = r"^acc_{act}(?:_(\d+))?_waist\.csv$"
ACC_SEGDIR_RE_TPL = r"acc_{act}_(\d+)_csv"
ACC_ZIP_RE_TPL = r"^acc_{act}(?:_(\d+))?_csv\.zip$"


def _seg_from_path(activity: str, path: str) -> str:
    base = os.path.basename(path)

    m = re.match(ACC_FILE_RE_TPL.format(act=re.escape(activity)), base, re.IGNORECASE)
    if m and m.group(1):
        return m.group(1)

    m = re.match(ACC_ZIP_RE_TPL.format(act=re.escape(activity)), base, re.IGNORECASE)
    if m and m.group(1):
        return m.group(1)

    parts = path.replace("\\", "/").split("/")
    for p in reversed(parts):
        m = re.match(ACC_SEGDIR_RE_TPL.format(act=re.escape(activity)), p, re.IGNORECASE)
        if m:
            return m.group(1)

    return ""


def _standardize_acc(df: pd.DataFrame) -> pd.DataFrame:
    # expected columns: id, attr_time, attr_x, attr_y, attr_z
    return df.rename(columns={"attr_time": "t", "attr_x": "ax", "attr_y": "ay", "attr_z": "az"})[["t", "ax", "ay", "az"]]


def _read_acc_waist_all(data_dir: str, activity: str) -> Dict[str, pd.DataFrame]:
    """
    Recursively search acc_<activity>_csv/** for accel waist CSV files,
    and also look into any acc_...zip files there.

    Returns dict: seg_token -> concatenated df
    """
    root = os.path.join(data_dir, f"acc_{activity}_csv")
    out: Dict[str, List[pd.DataFrame]] = {}

    if not os.path.isdir(root):
        # fallback: sometimes only zip exists directly under data_dir
        z0 = os.path.join(data_dir, f"acc_{activity}_csv.zip")
        if os.path.isfile(z0):
            root = data_dir
        else:
            raise FileNotFoundError(f"Missing accel folder acc_{activity}_csv and no acc zip in {data_dir}")

    acc_file_re = re.compile(ACC_FILE_RE_TPL.format(act=re.escape(activity)), re.IGNORECASE)
    acc_zip_re = re.compile(ACC_ZIP_RE_TPL.format(act=re.escape(activity)), re.IGNORECASE)

    # 1) raw CSVs (recursive)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if acc_file_re.match(fn):
                p = os.path.join(dirpath, fn)
                tok = _seg_from_path(activity, p)
                df = pd.read_csv(p)
                out.setdefault(tok, []).append(df)

    # 2) accel zips inside the tree
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if acc_zip_re.match(fn):
                zpath = os.path.join(dirpath, fn)
                tok_zip = _seg_from_path(activity, zpath)
                with zipfile.ZipFile(zpath, "r") as zf:
                    for m in zf.namelist():
                        base = os.path.basename(m)
                        if acc_file_re.match(base):
                            tok = _seg_from_path(activity, base) or tok_zip
                            with zf.open(m, "r") as f:
                                df = pd.read_csv(f)
                            out.setdefault(tok, []).append(df)

    if not out:
        raise FileNotFoundError(f"No waist accel CSVs found under {root} (recursive scan)")

    out2: Dict[str, pd.DataFrame] = {}
    for tok, lst in out.items():
        out2[tok] = pd.concat(lst, ignore_index=True) if len(lst) > 1 else lst[0]
    return out2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="folder containing proband1..proband15")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--only_missing", action="store_true")
    args = ap.parse_args()

    raw_root = os.path.abspath(args.raw_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    proband_dirs = sorted([d for d in os.listdir(raw_root) if re.match(r"^proband\d+$", d)])
    if not proband_dirs:
        raise SystemExit(f"No proband folders found under: {raw_root}")

    for prob in proband_dirs:
        pid = int(prob.replace("proband", ""))
        data_dir = os.path.join(raw_root, prob, "data")
        if not os.path.isdir(data_dir):
            print(f"[SKIP] {prob}: missing data/ folder")
            continue

        for act in ACTIVITIES:
            out_path = os.path.join(out_dir, f"proband{pid}_{act}.csv")
            if args.only_missing and os.path.isfile(out_path):
                continue

            try:
                acc_map = _read_acc_waist_all(data_dir, act)
            except FileNotFoundError as e:
                print(f"[SKIP] {prob} {act}: {e}")
                continue

            # concat all segments (preserve time order within each segment, then concat)
            parts = []
            for tok in sorted(acc_map.keys(), key=lambda x: (x == "", x)):
                df = _standardize_acc(acc_map[tok]).sort_values("t")
                parts.append(df)

            acc = pd.concat(parts, ignore_index=True)

            out = pd.DataFrame({
                "id": pid,
                "acc_x": acc["ax"].astype("float32"),
                "acc_y": acc["ay"].astype("float32"),
                "acc_z": acc["az"].astype("float32"),
                "label": act,
            })

            out.to_csv(out_path, index=False)
            print(f"[OK] wrote {out_path}  rows={len(out)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
