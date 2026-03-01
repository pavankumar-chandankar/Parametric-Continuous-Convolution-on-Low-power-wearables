import argparse
import io
import os
import re
import zipfile
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _iter_metric_csvs_from_zip(zip_path: str) -> Iterable[Tuple[str, bytes]]:
    zf = zipfile.ZipFile(zip_path, "r")
    for name in zf.namelist():
        if re.search(r"per_class_metrics_loso_sbj_\d+(_(10hz|5hz))?\.csv$", name):
            yield name, zf.read(name)


def _iter_metric_csvs_from_root(root_dir: str) -> Iterable[Tuple[str, bytes]]:
    # Walk root_dir and load matching CSVs
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if re.search(r"per_class_metrics_loso_sbj_\d+(_(10hz|5hz))?\.csv$", fn):
                p = os.path.join(dirpath, fn)
                with open(p, "rb") as f:
                    yield p, f.read()


def _rate_from_filename(path: str) -> str:
    # per_class_metrics_loso_sbj_<k>.csv -> 20hz
    # per_class_metrics_loso_sbj_<k>_10hz.csv -> 10hz, etc.
    m = re.search(r"_((10hz|5hz))\.csv$", path)
    return m.group(1) if m else "20hz"


def _subject_from_filename(path: str) -> int:
    m = re.search(r"loso_sbj_(\d+)", path)
    if not m:
        raise ValueError(f"Could not parse subject from filename: {path}")
    return int(m.group(1))


def _resolve_class_names(df: pd.DataFrame) -> Dict[int, str]:
    # Prefer single-letter labels (A,B,...) when present; else fallback to class_<id>
    id_to_name: Dict[int, str] = {}
    for cid in sorted(df["class_id"].unique()):
        names = df.loc[df["class_id"] == cid, "class_name"].astype(str).tolist()
        letters = [n for n in names if len(n) == 1 and n.isalpha()]
        if letters:
            id_to_name[cid] = Counter(letters).most_common(1)[0][0]
        else:
            id_to_name[cid] = f"class_{cid}"
    return id_to_name


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\\\pm$ {std:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=str, help="Path to a zip containing per_class_metrics_*.csv files.")
    src.add_argument("--root-dir", type=str, help="Root directory containing per_class_metrics_*.csv files.")
    ap.add_argument("--caption", type=str, required=True, help="LaTeX caption text.")
    ap.add_argument("--label", type=str, required=True, help="LaTeX label, e.g., tab:wisdm_watch_per_activity_f1")
    ap.add_argument("--rates", type=str, default="20hz,10hz,5hz", help="Comma-separated rates (default: 20hz,10hz,5hz)")
    args = ap.parse_args()

    rates = [r.strip() for r in args.rates.split(",") if r.strip()]
    if not rates:
        raise SystemExit("No rates provided.")

    # Load all metric CSVs
    recs = []
    it = _iter_metric_csvs_from_zip(args.zip) if args.zip else _iter_metric_csvs_from_root(args.root_dir)

    for path, raw in it:
        rate = _rate_from_filename(path)
        sbj = _subject_from_filename(path)
        df = pd.read_csv(io.BytesIO(raw))

        # expected columns: class_id, class_name, f1
        if "class_id" not in df.columns or "f1" not in df.columns:
            raise SystemExit(f"Missing required columns in {path}. Columns={list(df.columns)}")

        # normalize f1 to percent
        f1 = df["f1"].astype(float).to_numpy()
        if np.nanmax(f1) <= 1.0:
            f1 = f1 * 100.0

        names = df["class_name"].astype(str) if "class_name" in df.columns else pd.Series([""] * len(df))
        for cid, cname, f1v in zip(df["class_id"].astype(int), names, f1):
            recs.append({"sbj": sbj, "rate": rate, "class_id": int(cid), "class_name": str(cname), "f1": float(f1v)})

    all_df = pd.DataFrame(recs)
    if all_df.empty:
        raise SystemExit("No per_class_metrics_*.csv files found.")

    # resolve activity names
    id_to_name = _resolve_class_names(all_df)
    all_df["activity"] = all_df["class_id"].map(id_to_name)

    # aggregate mean/std across subjects per activity per rate
    agg = (
        all_df.groupby(["activity", "rate"])["f1"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # pivot
    pivot_mean = agg.pivot(index="activity", columns="rate", values="mean")
    pivot_std = agg.pivot(index="activity", columns="rate", values="std")

    # ensure desired rate columns exist
    missing = [r for r in rates if r not in pivot_mean.columns]
    if missing:
        raise SystemExit(f"Missing rates in data: {missing}. Available: {list(pivot_mean.columns)}")

    # stable activity order: single-letter first, then others
    activities = sorted(pivot_mean.index.tolist(), key=lambda x: (len(x) != 1, x))

    # emit LaTeX
    print(r"\begin{table}[H]\centering")
    print(rf"\caption{{{args.caption}}}\label{{{args.label}}}")
    print(r"\begin{tabular}{l" + "c" * len(rates) + "}")
    print(r"\toprule")
    print("Activity & " + " & ".join([r.replace("hz", " Hz") for r in rates]) + r" \\")
    print(r"\midrule")

    for act in activities:
        cells = []
        for r in rates:
            cells.append(_fmt(float(pivot_mean.loc[act, r]), float(pivot_std.loc[act, r])))
        print(f"{act} & " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
