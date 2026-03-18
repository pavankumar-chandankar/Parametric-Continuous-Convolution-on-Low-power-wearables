from __future__ import annotations

import argparse
import os
import re
import zipfile
from typing import Dict, List, Tuple

import pandas as pd


ACTIVITIES = [
    "walking", "running", "sitting", "standing", "lying",
    "jumping", "climbingup", "climbingdown",
]


ACC_FILE_RE_TPL = r"^acc_{act}(?:_(\d+))?_waist\.csv$"
GYR_FILE_RE_TPL = r"^Gyroscope_{act}(?:_(\d+))?_waist\.csv$"

ACC_SEGDIR_RE_TPL = r"acc_{act}_(\d+)_csv"
GYR_ZIP_RE_TPL = r"^gyr_{act}(?:_(\d+))?_csv\.zip$"
ACC_ZIP_RE_TPL = r"^acc_{act}(?:_(\d+))?_csv\.zip$"


def _seg_from_path(activity: str, path: str) -> str:
    """
    Try to infer a segment token from:
      - filename: acc_<act>_<k>_waist.csv
      - parent dirs: acc_<act>_<k>_csv/
      - zip filename: acc_<act>_<k>_csv.zip or gyr_<act>_<k>_csv.zip
    """
    base = os.path.basename(path)

    # filename based
    m = re.match(ACC_FILE_RE_TPL.format(act=re.escape(activity)), base, re.IGNORECASE)
    if m and m.group(1):
        return m.group(1)

    m = re.match(GYR_FILE_RE_TPL.format(act=re.escape(activity)), base, re.IGNORECASE)
    if m and m.group(1):
        return m.group(1)

    # zip filename based
    m = re.match(ACC_ZIP_RE_TPL.format(act=re.escape(activity)), base, re.IGNORECASE)
    if m and m.group(1):
        return m.group(1)

    m = re.match(GYR_ZIP_RE_TPL.format(act=re.escape(activity)), base, re.IGNORECASE)
    if m and m.group(1):
        return m.group(1)

    # directory based
    parts = path.replace("\\", "/").split("/")
    for p in reversed(parts):
        m = re.match(ACC_SEGDIR_RE_TPL.format(act=re.escape(activity)), p, re.IGNORECASE)
        if m:
            return m.group(1)

    return ""  # default segment


def _standardize_acc(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"attr_time": "t", "attr_x": "ax", "attr_y": "ay", "attr_z": "az"})[["t", "ax", "ay", "az"]]


def _standardize_gyr(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"attr_time": "t", "attr_x": "gx", "attr_y": "gy", "attr_z": "gz"})[["t", "gx", "gy", "gz"]]


def _merge_asof(acc: pd.DataFrame, gyr: pd.DataFrame, tol_ms: int) -> pd.DataFrame:
    acc = acc.sort_values("t")
    gyr = gyr.sort_values("t")
    merged = pd.merge_asof(acc, gyr, on="t", direction="nearest", tolerance=int(tol_ms)).dropna()
    return merged


def _read_acc_waist_all(data_dir: str, activity: str) -> Dict[str, pd.DataFrame]:
    """
    Recursively search acc_<activity>_csv/** for accel waist CSV files, and also look into any acc_...zip files there.
    Returns dict: seg_token -> concatenated raw df(s)
    """
    root = os.path.join(data_dir, f"acc_{activity}_csv")
    out: Dict[str, List[pd.DataFrame]] = {}

    if not os.path.isdir(root):
        # fallback: some datasets keep only the zip at data_dir level
        z0 = os.path.join(data_dir, f"acc_{activity}_csv.zip")
        if os.path.isfile(z0):
            root = data_dir
        else:
            raise FileNotFoundError(f"Missing accel folder {os.path.join(data_dir, f'acc_{activity}_csv')} and no acc zip in {data_dir}")

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

    # concatenate per token
    out2: Dict[str, pd.DataFrame] = {}
    for tok, lst in out.items():
        out2[tok] = pd.concat(lst, ignore_index=True) if len(lst) > 1 else lst[0]
    return out2


def _read_gyr_waist_all(data_dir: str, activity: str) -> Dict[str, pd.DataFrame]:
    """
    Gyro is usually stored in zip(s). Handle both:
      gyr_<activity>_csv.zip
      gyr_<activity>_<k>_csv.zip
    Returns dict: seg_token -> concatenated raw df(s)
    """
    out: Dict[str, List[pd.DataFrame]] = {}

    gyr_file_re = re.compile(GYR_FILE_RE_TPL.format(act=re.escape(activity)), re.IGNORECASE)
    gyr_zip_re = re.compile(GYR_ZIP_RE_TPL.format(act=re.escape(activity)), re.IGNORECASE)

    # find all candidate gyro zips in data_dir (and in activity folder if any)
    candidates: List[str] = []

    # common: directly in data_dir
    for fn in os.listdir(data_dir):
        if gyr_zip_re.match(fn):
            candidates.append(os.path.join(data_dir, fn))

    # also scan inside acc_<activity>_csv tree (some datasets colocate zips)
    acc_root = os.path.join(data_dir, f"acc_{activity}_csv")
    if os.path.isdir(acc_root):
        for dirpath, _, filenames in os.walk(acc_root):
            for fn in filenames:
                if gyr_zip_re.match(fn):
                    candidates.append(os.path.join(dirpath, fn))

    if not candidates:
        raise FileNotFoundError(f"No gyro zip matched gyr_{activity}_*_csv.zip under {data_dir}")

    for zpath in sorted(set(candidates)):
        tok_zip = _seg_from_path(activity, zpath)
        with zipfile.ZipFile(zpath, "r") as zf:
            found_any = False
            for m in zf.namelist():
                base = os.path.basename(m)
                if gyr_file_re.match(base):
                    found_any = True
                    tok = _seg_from_path(activity, base) or tok_zip
                    with zf.open(m, "r") as f:
                        df = pd.read_csv(f)
                    out.setdefault(tok, []).append(df)
            if not found_any:
                # not fatal; some zips may have different naming, but we keep going
                pass

    if not out:
        raise FileNotFoundError(f"No waist gyro CSVs matched inside gyro zip(s) for {activity}")

    out2: Dict[str, pd.DataFrame] = {}
    for tok, lst in out.items():
        out2[tok] = pd.concat(lst, ignore_index=True) if len(lst) > 1 else lst[0]
    return out2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="folder containing proband1..proband15")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tol_ms", type=int, default=10)
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
        pdir = os.path.join(raw_root, prob)
        data_dir = os.path.join(pdir, "data")
        if not os.path.isdir(data_dir):
            print(f"[SKIP] {prob}: missing data/ folder")
            continue

        for act in ACTIVITIES:
            out_path = os.path.join(out_dir, f"proband{pid}_{act}.csv")
            if args.only_missing and os.path.isfile(out_path):
                continue

            try:
                acc_map = _read_acc_waist_all(data_dir, act)
                gyr_map = _read_gyr_waist_all(data_dir, act)
            except FileNotFoundError as e:
                print(f"[SKIP] {prob} {act}: {e}")
                continue

            merged_parts: List[pd.DataFrame] = []
            tokens = sorted(set(acc_map.keys()) | set(gyr_map.keys()), key=lambda x: (x == "", x))

            for tok in tokens:
                if tok not in acc_map:
                    continue

                acc_df = _standardize_acc(acc_map[tok])

                if tok in gyr_map:
                    gyr_df = _standardize_gyr(gyr_map[tok])
                else:
                    gyr_df = pd.concat([_standardize_gyr(g) for g in gyr_map.values()], ignore_index=True)

                m = _merge_asof(acc_df, gyr_df, tol_ms=args.tol_ms)
                if not m.empty:
                    merged_parts.append(m)

            if not merged_parts:
                print(f"[SKIP] {prob} {act}: merged acc+gyr is empty (check tol_ms={args.tol_ms})")
                continue

            merged = pd.concat(merged_parts, ignore_index=True)

            out = pd.DataFrame({
                "id": pid,
                "acc_x": merged["ax"].astype("float32"),
                "acc_y": merged["ay"].astype("float32"),
                "acc_z": merged["az"].astype("float32"),
                "gyr_x": merged["gx"].astype("float32"),
                "gyr_y": merged["gy"].astype("float32"),
                "gyr_z": merged["gz"].astype("float32"),
                "label": act,
            })

            out.to_csv(out_path, index=False)
            print(f"[OK] wrote {out_path}  rows={len(out)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
