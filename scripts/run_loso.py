from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

# Ensure repo root is on PYTHONPATH when running as `python scripts/...`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.train import run_inertial_network
from utils.os_utils import load_config
from utils.torch_utils_final import fix_random_seed


def _build_labels_and_label_dict(anno: Dict, has_null: bool) -> Tuple[List[str], Dict[str, int]]:
    raw_ld = anno.get("label_dict", None)
    if raw_ld is None:
        raise KeyError("Annotation JSON missing 'label_dict'")

    if isinstance(raw_ld, dict):
        labels = list(raw_ld)
    else:
        labels = list(raw_ld)

    if has_null:
        labels = ["null"] + labels

    label_dict = {lab: idx for idx, lab in enumerate(labels)}
    return labels, label_dict


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LOSO training for a single YAML config (all splits).")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start_split", type=int, default=1, help="1-based index")
    ap.add_argument("--end_split", type=int, default=0, help="0 = until the end")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--ckpt_freq", type=int, default=-1)

    ap.add_argument("--gamma_quant", type=str, default=None)
    ap.add_argument("--quant_bits", type=int, default=4)
    ap.add_argument("--gamma_type", type=str, default="id")
    ap.add_argument("--apply_gamma", type=str, default="global")

    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = fix_random_seed(args.seed, include_cuda=True)

    anno_list = cfg.get("anno_json", [])
    if not anno_list:
        raise ValueError("Config has no 'anno_json' list.")

    s0 = max(1, int(args.start_split))
    s1 = int(args.end_split) if int(args.end_split) > 0 else len(anno_list)

    print(f"[LOSO] config={args.config} splits={s0}..{s1} seed={args.seed}")

    for i, anno_path in enumerate(anno_list, start=1):
        if i < s0 or i > s1:
            continue

        with open(anno_path, "r") as f:
            anno = json.load(f)

        db = anno["database"]
        has_null = bool(cfg.get("has_null", False))

        labels, label_dict = _build_labels_and_label_dict(anno, has_null=has_null)
        cfg["labels"] = labels
        cfg["label_dict"] = label_dict

        train_sbjs = [x for x in db if db[x].get("subset") == "Training"]
        val_sbjs = [x for x in db if db[x].get("subset") == "Validation"]
        if len(val_sbjs) == 0:
            val_sbjs = [x for x in db if db[x].get("subset") in ("Test", "Testing")]

        cfg["dataset"]["json_anno"] = anno_path
        split_name = os.path.splitext(os.path.basename(anno_path))[0]

        print("=" * 70)
        print(f"[SPLIT {i}/{len(anno_list)}] {split_name}")
        print(f"train={len(train_sbjs)} val={len(val_sbjs)}")
        print("=" * 70)

        run_inertial_network(
            train_sbjs=train_sbjs,
            val_sbjs=val_sbjs,
            cfg=cfg,
            ckpt_folder="",
            ckpt_freq=args.ckpt_freq,
            resume=args.resume,
            rng_generator=rng,
            run=None,
            gamma_quant=args.gamma_quant,
            quant_bits=args.quant_bits,
            gamma_type=args.gamma_type,
            apply_gamma=args.apply_gamma,
        )


if __name__ == "__main__":
    main()
