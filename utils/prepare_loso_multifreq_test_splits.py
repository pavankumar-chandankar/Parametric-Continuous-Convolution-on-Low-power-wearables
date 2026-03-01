# prepare_loso_multifreq_test_splits.py
import argparse
import copy
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from utils.data_utils import apply_sliding_window


def _freq_tag_from_sampling_rate(sr: float) -> str:
    sr = float(sr)
    if abs(sr - 50.0) < 1e-3:
        return "50hz"
    if abs(sr - 25.0) < 1e-3:
        return "25hz"
    if abs(sr - 12.5) < 1e-3:
        return "12_5hz"
    if abs(sr - 12.0) < 1e-3:
        return "12hz"
    if abs(sr - 6.0) < 1e-3:
        return "6hz"
    return f"{sr:g}hz"


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_anno(anno_path: str) -> Dict:
    with open(anno_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_labels_and_label_dict_from_anno(anno: Dict, has_null: bool) -> Tuple[List[str], Dict[str, int]]:
    """
    Match training behavior:
    - Uses the JSON's `label_dict` entry.
    - If it's a dict: list(raw_ld) yields keys in JSON insertion order (matches how your loso loader behaved).
    - If it's a list: uses list as-is.
    - If has_null: prepend 'null' at index 0.
    """
    if "label_dict" not in anno:
        raise RuntimeError("Annotation JSON has no 'label_dict' key, cannot map CSV labels.")

    raw_ld = anno["label_dict"]

    if isinstance(raw_ld, dict):
        base_labels = list(raw_ld)  # keys in insertion order
    elif isinstance(raw_ld, list):
        base_labels = list(raw_ld)
    else:
        raise RuntimeError(f"Unsupported label_dict type in anno: {type(raw_ld)}")

    labels = (["null"] + base_labels) if has_null else base_labels
    label_dict = {lab: idx for idx, lab in enumerate(labels)}
    return labels, label_dict


def _parse_loso_split(anno_path: str, anno: Dict) -> Tuple[str, List[str], List[str]]:
    """
    Primary: use database[*]['subset'] == Training/Validation
    Fallback: if 'subset' is missing, derive held-out subject from filename (loso_sbj_X -> sbj_X).
    """
    split_name = os.path.splitext(os.path.basename(anno_path))[0]

    db = anno.get("database", {})
    if not isinstance(db, dict) or len(db) == 0:
        raise RuntimeError(f"Annotation JSON {anno_path} has no usable 'database' dict.")

    has_subset = any(isinstance(v, dict) and ("subset" in v) for v in db.values())

    if has_subset:
        train_sbjs = [k for k, v in db.items() if isinstance(v, dict) and v.get("subset") == "Training"]
        val_sbjs = [k for k, v in db.items() if isinstance(v, dict) and v.get("subset") == "Validation"]
        if len(val_sbjs) == 0:
            # Some datasets call it "Testing"
            val_sbjs = [k for k, v in db.items() if isinstance(v, dict) and v.get("subset") in ("Test", "Testing")]
        if len(train_sbjs) == 0 or len(val_sbjs) == 0:
            raise RuntimeError(
                f"Could not get train/val subjects from subsets in {anno_path}. "
                f"Found train={len(train_sbjs)} val={len(val_sbjs)}."
            )
        return split_name, train_sbjs, val_sbjs

    # Fallback: filename-driven LOSO
    # expected: loso_sbj_0 -> heldout sbj_0
    heldout = None
    if "loso_" in split_name:
        heldout = split_name.replace("loso_", "")
    if heldout is None or heldout not in db:
        raise RuntimeError(
            f"No 'subset' fields found and cannot derive held-out subject from filename: {split_name}"
        )

    val_sbjs = [heldout]
    train_sbjs = [k for k in db.keys() if k != heldout]
    return split_name, train_sbjs, val_sbjs


def _read_subject_csv(sens_folder: str, sbj: str, label_dict: Dict[str, int]) -> np.ndarray:
    csv_path = os.path.join(sens_folder, sbj + ".csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path, index_col=False, low_memory=False)

    # Match train.py behavior
    df = df.replace({"label": label_dict}).fillna(0)

    # Enforce numeric labels (if this fails, your label_dict does not match CSV strings)
    if "label" not in df.columns:
        raise RuntimeError(f"'label' column missing in {csv_path}")
    try:
        df["label"] = pd.to_numeric(df["label"], errors="raise")
    except Exception as e:
        ex = df["label"].astype(str).unique()[:12]
        raise ValueError(
            f"Label mapping failed in {csv_path}. Example labels after replace: {ex}. "
            f"Your anno label_dict does not match CSV label strings."
        ) from e

    arr = df.to_numpy()

    # Keep float array behavior like train.py (its train_data starts float)
    return arr.astype(np.float32)


def _compute_train_minmax(train_arrays: List[np.ndarray]) -> Tuple[float, float]:
    # train.py uses min/max over columns 1:-1
    mn = None
    mx = None
    for a in train_arrays:
        feats = a[:, 1:-1]
        local_mn = float(np.min(feats))
        local_mx = float(np.max(feats))
        mn = local_mn if mn is None else min(mn, local_mn)
        mx = local_mx if mx is None else max(mx, local_mx)
    if mn is None or mx is None:
        raise RuntimeError("Could not compute min/max (empty training arrays).")
    return mn, mx


def _normalize_inplace(data: np.ndarray, mn: float, mx: float) -> None:
    denom = (mx - mn) if (mx - mn) != 0 else 1.0
    data[:, 1:-1] = 2.0 * (data[:, 1:-1] - mn) / denom - 1.0


def _make_windows_like_training(cfg: Dict, val_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    window_size = int(cfg["dataset"]["window_size"])
    window_overlap = cfg["dataset"]["window_overlap"]

    # InertialDataset -> apply_sliding_window(data, window_size, window_overlap)
    _, windows, labels = apply_sliding_window(
        val_data,
        window_size=window_size,
        window_overlap=window_overlap,
        no_context_windows=0,
    )

    # Training feeds model features[:, :, 1:]
    x = windows[:, :, 1:].astype(np.float32)
    y = labels.astype(np.uint8)
    return x, y


def build_fold_test_windows(cfg: Dict, anno_path: str) -> Tuple[str, str, np.ndarray, np.ndarray]:
    """
    Returns:
      split_name, freq_tag, x, y
    """
    anno = _load_anno(anno_path)

    split_name, train_sbjs, val_sbjs = _parse_loso_split(anno_path, anno)

    labels, label_dict = _build_labels_and_label_dict_from_anno(anno, bool(cfg.get("has_null", False)))

    # Inject like training expects
    cfg = copy.deepcopy(cfg)
    cfg["labels"] = labels
    cfg["label_dict"] = label_dict
    cfg.setdefault("dataset", {})
    cfg["dataset"]["json_anno"] = anno_path

    sens_folder = cfg["dataset"]["sens_folder"]
    sr = float(cfg["dataset"]["sampling_rate"])
    freq_tag = _freq_tag_from_sampling_rate(sr)

    # Read training subjects (for min/max)
    train_arrays = [_read_subject_csv(sens_folder, s, label_dict) for s in train_sbjs]
    mn, mx = _compute_train_minmax(train_arrays)

    # Read held-out subject(s)
    val_arrays = [_read_subject_csv(sens_folder, s, label_dict) for s in val_sbjs]
    val_data = np.concatenate(val_arrays, axis=0)

    # Normalize val with train min/max (exactly train.py)
    _normalize_inplace(val_data, mn, mx)

    x, y = _make_windows_like_training(cfg, val_data)
    return split_name, freq_tag, x, y


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create persistent LOSO test windows per fold and per frequency (matches train.py normalization + windowing)."
    )

    # accept both styles to avoid your earlier argument mismatch headaches
    p.add_argument("--cfg-50", "--config-50", dest="cfg_50", required=True)
    p.add_argument("--cfg-25", "--config-25", dest="cfg_25", required=True)
    p.add_argument("--cfg-12", "--config-12", dest="cfg_12", required=True)
    p.add_argument("--cfg-6", "--config-6", dest="cfg_6", required=True)

    p.add_argument(
        "--out-root",
        "--outdir",
        dest="out_root",
        default="./data/wear/multifreq_test_loso",
        help="Root folder where fold subfolders will be created.",
    )
    p.add_argument(
        "--only-fold",
        dest="only_fold",
        default="",
        help="Optional: only generate one fold (by split name like 'loso_sbj_3' or by index like '4').",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing npz files. Default: keep existing and skip.",
    )

    args = p.parse_args()

    cfg50 = _load_yaml(args.cfg_50)
    cfg25 = _load_yaml(args.cfg_25)
    cfg12 = _load_yaml(args.cfg_12)
    cfg6 = _load_yaml(args.cfg_6)

    anno_list = cfg25.get("anno_json", [])
    if not anno_list:
        raise RuntimeError("cfg_25 has no anno_json list. This script needs the LOSO anno_json list.")

    # Sanity: all configs should share same number of folds
    for name, cfg in [("50", cfg50), ("25", cfg25), ("12", cfg12), ("6", cfg6)]:
        if len(cfg.get("anno_json", [])) != len(anno_list):
            raise RuntimeError(f"Config {name} does not have the same anno_json length as cfg_25.")

    os.makedirs(args.out_root, exist_ok=True)

    for fold_idx, anno_path in enumerate(anno_list, start=1):
        # filter fold if requested
        split_name = os.path.splitext(os.path.basename(anno_path))[0]
        if args.only_fold:
            if args.only_fold.isdigit():
                if int(args.only_fold) != fold_idx:
                    continue
            else:
                if args.only_fold != split_name:
                    continue

        fold_dir = os.path.join(args.out_root, split_name)
        os.makedirs(fold_dir, exist_ok=True)

        print("=" * 60)
        print(f"[GEN] Fold {fold_idx}/{len(anno_list)}  split={split_name}")
        print(f"Anno: {anno_path}")
        print(f"Out : {fold_dir}")
        print("=" * 60)

        for cfg in [cfg50, cfg25, cfg12, cfg6]:
            split_name2, freq_tag, x, y = build_fold_test_windows(cfg, anno_path)
            assert split_name2 == split_name, "Internal split_name mismatch."

            out_npz = os.path.join(fold_dir, f"test_windows_{freq_tag}.npz")
            if (not args.overwrite) and os.path.exists(out_npz):
                print(f"[SKIP] exists: {out_npz}")
                continue

            np.savez_compressed(out_npz, x=x, y=y)
            print(f"[OK]  {freq_tag}: x={x.shape} y={y.shape} -> {out_npz}")


if __name__ == "__main__":
    main()
