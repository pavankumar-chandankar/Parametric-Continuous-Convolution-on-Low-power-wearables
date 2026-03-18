from __future__ import annotations

import argparse
import copy
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


def _resolve_path(p: str) -> str:
    p = str(p)
    if os.path.exists(p):
        return p

    # Try common case variants for WEAR annotation folders
    variants = [
        p,
        p.replace("/50hz/", "/50Hz/"),
        p.replace("/25hz/", "/25Hz/"),
        p.replace("/12hz/", "/12Hz/"),
        p.replace("/6hz/", "/6Hz/"),
    ]
    for v in variants:
        if os.path.exists(v):
            return v

    raise FileNotFoundError(f"Could not resolve path: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kernel-size sweep for multiple fixed-frequency configs (LOSO).")
    ap.add_argument("--configs", type=str, required=True, help="Comma-separated YAML configs (one per frequency).")
    ap.add_argument("--exp_tag", type=str, required=True)

    ap.add_argument("--kernel_min", type=int, default=3)
    ap.add_argument("--kernel_max", type=int, default=31)
    ap.add_argument("--kernel_step", type=int, default=2)
    ap.add_argument("--odd_only", action="store_true")

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start_split", type=int, default=1)
    ap.add_argument("--end_split", type=int, default=0, help="0 = until end")

    ap.add_argument("--epochs", type=int, default=0, help="If >0, override train_cfg.epochs")

    ap.add_argument("--gamma_quant", type=str, default=None)
    ap.add_argument("--quant_bits", type=int, default=4)
    ap.add_argument("--gamma_type", type=str, default="id")
    ap.add_argument("--apply_gamma", type=str, default="global")

    args = ap.parse_args()

    cfg_paths = [p.strip() for p in args.configs.split(",") if p.strip()]
    if not cfg_paths:
        raise ValueError("--configs produced empty list")

    ks = list(range(int(args.kernel_min), int(args.kernel_max) + 1, int(args.kernel_step)))
    if args.odd_only:
        ks = [k for k in ks if (k % 2 == 1)]
    if not ks:
        raise ValueError("Kernel size list is empty after filtering.")

    rng = fix_random_seed(args.seed, include_cuda=True)

    for cfg_path in cfg_paths:
        base_cfg = load_config(cfg_path)

        anno_list = base_cfg.get("anno_json", [])
        if not anno_list:
            raise ValueError(f"Config has no 'anno_json': {cfg_path}")

        # resolve annotation paths (relative to repo root)
        anno_list = [_resolve_path(p) for p in anno_list]
        base_cfg["anno_json"] = anno_list

        s0 = max(1, int(args.start_split))
        s1 = int(args.end_split) if int(args.end_split) > 0 else len(anno_list)

        for k in ks:
            cfg = copy.deepcopy(base_cfg)

            cfg.setdefault("train_cfg", {})
            cfg["train_cfg"]["log_subdir"] = os.path.join(str(args.exp_tag), f"ks{k:02d}")
            cfg["train_cfg"]["multirate_val_lengths"] = []

            if int(args.epochs) > 0:
                cfg["train_cfg"]["epochs"] = int(args.epochs)

            cfg.setdefault("model", {})
            cfg["model"]["conv_type"] = "standard"
            cfg["model"]["conv_kernel_size"] = int(k)
            cfg["model"]["standard_padding"] = cfg["model"].get("standard_padding", "same")

            for i, anno_path in enumerate(cfg["anno_json"], start=1):
                if i < s0 or i > s1:
                    continue

                with open(anno_path, "r") as f_json:
                    anno = json.load(f_json)

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

                print("=" * 80)
                print(f"[RUN] cfg={os.path.basename(cfg_path)} ks={k} split={i}/{len(cfg['anno_json'])} name={split_name}")
                print("=" * 80)

                run_inertial_network(
                    train_sbjs=train_sbjs,
                    val_sbjs=val_sbjs,
                    cfg=cfg,
                    ckpt_folder="",
                    ckpt_freq=-1,
                    resume="",
                    rng_generator=rng,
                    run=None,
                    gamma_quant=args.gamma_quant,
                    quant_bits=args.quant_bits,
                    gamma_type=args.gamma_type,
                    apply_gamma=args.apply_gamma,
                )


if __name__ == "__main__":
    main()
