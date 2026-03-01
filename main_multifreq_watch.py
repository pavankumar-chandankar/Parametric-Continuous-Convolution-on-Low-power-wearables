import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Set

import torch
import yaml

from models.train_multifreq_watch import (
    run_inertial_expert_702010,
    run_inertial_expert_loso,
    test_multifreq_cascade,
)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def prepare_labels_and_subjects(cfg: Dict) -> List[str]:
    anno_list = cfg.get("anno_json", [])
    if not anno_list:
        raise ValueError("Config file must contain an 'anno_json' entry with at least one annotation file.")

    all_subjects: Set[str] = set()
    labels: List[str] = []

    first_label_dict_done = False

    for anno_path in anno_list:
        if not os.path.isfile(anno_path):
            raise FileNotFoundError(f"Annotation JSON not found: {anno_path}")

        with open(anno_path, "r") as f:
            anno_file = json.load(f)

        db = anno_file.get("database", {})
        if not isinstance(db, dict) or len(db) == 0:
            raise KeyError(f"'database' not found or empty in annotation file {anno_path}")

        all_subjects.update(db.keys())

        if not first_label_dict_done:
            raw_ld = anno_file.get("label_dict", None)
            if raw_ld is None:
                raise KeyError(f"'label_dict' not found in annotation file {anno_path}")

            if isinstance(raw_ld, dict):
                labels = [name for name, _ in sorted(raw_ld.items(), key=lambda kv: kv[1])]
            else:
                labels = list(raw_ld)

            if cfg.get("has_null", False):
                labels = ["null"] + labels

            label_dict = {lab: idx for idx, lab in enumerate(labels)}
            cfg["labels"] = labels
            cfg["label_dict"] = label_dict
            first_label_dict_done = True

    if not first_label_dict_done:
        raise RuntimeError("Could not build label_dict from any annotation file.")

    sbj_list = sorted(list(all_subjects))
    return sbj_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gamma-Quant Inertial – multi-frequency experts + oracle cascade."
    )

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--mode", type=str, default="train_expert", choices=["train_expert", "test_multifreq"])
    parser.add_argument(
        "--protocol",
        type=str,
        default="loso",
        choices=["loso", "702010"],
        help="Training protocol for expert models. Default: loso.",
    )

    parser.add_argument("--freq-tag", type=str, default=None, help="Frequency tag for logging (e.g. 20hz, 15hz, 10hz, 5hz).")
    parser.add_argument(
        "--test-split-dir",
        type=str,
        default="./data/wear/multifreq_test",
        help="Directory where held-out test windows are stored.",
    )
    parser.add_argument("--ckpt-freq", type=int, default=-1, help="(Kept for compatibility, currently unused.)")
    parser.add_argument("--resume", type=str, default="", help="Optional path to resume checkpoint.")

    parser.add_argument(
        "--gamma_quant",
        default=None,
        type=str,
        help="Gamma-Quant mode: 'quantize', 'quantize_and_calibrate', 'quantize_and_gamma', or None.",
    )
    parser.add_argument("--quant_bits", default=4, type=int, help="Number of bits for quantization (default: 4).")
    parser.add_argument("--gamma_type", default="s_shaped", type=str, help="Gamma function type (e.g. 's_shaped', 'id').")
    parser.add_argument("--apply_gamma", default="global", type=str, help="Apply gamma 'global' or 'local' (default: global).")

    parser.add_argument("--ckpt-6hz", type=str, default="", help="Checkpoint for lowest-rate expert (legacy arg name).")
    parser.add_argument("--ckpt-12hz", type=str, default="", help="Checkpoint for mid-rate expert (legacy arg name).")
    parser.add_argument("--ckpt-25hz", type=str, default="", help="Checkpoint for high-rate expert (legacy arg name).")
    parser.add_argument("--ckpt-50hz", type=str, default="", help="Optional checkpoint for highest-rate expert (legacy arg name).")

    args = parser.parse_args()

    cfg = load_config(args.config)
    sbj_list = prepare_labels_and_subjects(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Helpful sanity prints (answers your “WISDM vs WEAR?” confusion)
    ds = cfg.get("dataset", {})
    print("---- Config sanity ----")
    print(f"name: {cfg.get('name', '')}")
    print(f"dataset.sens_folder: {ds.get('sens_folder', '')}")
    print(f"dataset.sampling_rate: {ds.get('sampling_rate', '')}")
    print(f"dataset.window_size: {ds.get('window_size', '')}")
    print(f"anno_json count: {len(cfg.get('anno_json', []) or [])}")
    print("-----------------------")

    cfg_name = cfg.get("name", "deepconvlstm")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "train_expert":
        # DON’T force --freq-tag; infer from cfg.dataset.sampling_rate
        if args.freq_tag is None:
            sr = float(cfg.get("dataset", {}).get("sampling_rate", 0.0))
            if sr <= 0:
                raise ValueError("--freq-tag not provided and cfg.dataset.sampling_rate missing/invalid.")
            args.freq_tag = f"{sr:g}hz"
            print(f"[auto] Using --freq-tag={args.freq_tag} from cfg.dataset.sampling_rate={sr}")

        log_dir = os.path.join("logs", cfg_name, f"{args.freq_tag}_{args.protocol}", ts)
        print(f"Logging to: {log_dir}")

        if args.protocol == "702010":
            best_ckpt = run_inertial_expert_702010(
                cfg=cfg,
                sbj_list=sbj_list,
                log_dir=log_dir,
                freq_tag=args.freq_tag,
                seed=args.seed,
                test_split_dir=args.test_split_dir,
                device=device,
                gamma_quant=args.gamma_quant,
                quant_bits=args.quant_bits,
                gamma_type=args.gamma_type,
                apply_gamma=args.apply_gamma,
            )
            print("\n------------------------------------------------------------")
            print(f"Finished training expert for {args.freq_tag} (702010 window split).")
            print(f"Best checkpoint saved at: {best_ckpt}")
            print("------------------------------------------------------------\n")
        else:
            best_ckpts = run_inertial_expert_loso(
                cfg=cfg,
                sbj_list=sbj_list,
                log_dir=log_dir,
                seed=args.seed,
                test_split_dir=args.test_split_dir,
                device=device,
                gamma_quant=args.gamma_quant,
                quant_bits=args.quant_bits,
                gamma_type=args.gamma_type,
                apply_gamma=args.apply_gamma,
                resume=args.resume,
                ckpt_freq=args.ckpt_freq,
            )

            print("\n------------------------------------------------------------")
            print(f"Finished LOSO training for {args.freq_tag}.")
            print(f"Number of splits trained: {len(best_ckpts)}")
            if best_ckpts:
                print(f"Example best checkpoint: {best_ckpts[0]}")
            print("------------------------------------------------------------\n")

    elif args.mode == "test_multifreq":
        if not (args.ckpt_6hz and args.ckpt_12hz and args.ckpt_25hz):
            raise ValueError("Please provide --ckpt-6hz, --ckpt-12hz and --ckpt-25hz for test_multifreq.")

        print("Running multi-frequency cascade evaluation on held-out windows...")
        test_multifreq_cascade(
            cfg=cfg,
            ckpt_6hz=args.ckpt_6hz,
            ckpt_12hz=args.ckpt_12hz,
            ckpt_25hz=args.ckpt_25hz,
            ckpt_50hz=args.ckpt_50hz,
            test_split_dir=args.test_split_dir,
            device=device,
            gamma_quant=args.gamma_quant,
            quant_bits=args.quant_bits,
            gamma_type=args.gamma_type,
            apply_gamma=args.apply_gamma,
        )


if __name__ == "__main__":
    main()

