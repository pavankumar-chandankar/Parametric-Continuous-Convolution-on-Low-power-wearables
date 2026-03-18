from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from models.DeepConvLSTM_final import DeepConvLSTM
from models.multifreq_deepconvlstm import MultiFreqDeepConvLSTM, load_freq_mapping
from utils.os_utils import load_config
from utils.power_calculations import estimate_gated_energy


def _load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    x = d["x"]
    y = d["y"]
    return x, y


def _load_expert_from_split(
    cfg_path: str,
    ckpt_path: str,
    device: torch.device,
) -> torch.nn.Module:
    cfg = load_config(cfg_path)

    channels = int(cfg["dataset"]["input_dim"])
    classes = int(cfg["dataset"]["num_classes"])
    window_size = int(cfg["dataset"]["window_size"])

    mcfg = cfg["model"]
    net = DeepConvLSTM(
        channels=channels,
        classes=classes,
        window_size=window_size,
        conv_kernels=mcfg["conv_kernels"],
        conv_kernel_size=mcfg["conv_kernel_size"],
        lstm_units=mcfg["lstm_units"],
        lstm_layers=mcfg["lstm_layers"],
        dropout=mcfg["dropout"],
        gamma_quant=None,
        conv_type=mcfg.get("conv_type", "standard"),
        conv_rank=mcfg.get("conv_rank", 8),
        conv_mlp_hidden_dim=mcfg.get("conv_mlp_hidden_dim", 32),
        supported_sample_rates=mcfg.get("supported_sample_rates", (50, 25, 12, 6)),
        multirate_training=False,
        kernel_support_s=mcfg.get("kernel_support_s", None),
        standard_padding=mcfg.get("standard_padding", "valid"),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.to(device)
    return net


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    prec = precision_score(y_true, y_pred, average=None, zero_division=1, labels=list(range(num_classes)))
    rec = recall_score(y_true, y_pred, average=None, zero_division=1, labels=list(range(num_classes)))
    f1 = f1_score(y_true, y_pred, average=None, zero_division=1, labels=list(range(num_classes)))
    return {
        "acc_mean": float(np.nanmean((confusion_matrix(y_true, y_pred, normalize="true").diagonal())) * 100.0),
        "prec_mean": float(np.nanmean(prec) * 100.0),
        "rec_mean": float(np.nanmean(rec) * 100.0),
        "f1_mean": float(np.nanmean(f1) * 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate cascade inference using learned per-class frequency mapping.")
    ap.add_argument("--test_split_dir", type=str, required=True, help="Directory produced by prepare_loso_multifreq_test_splits.py")
    ap.add_argument("--mapping", type=str, required=True, help="best_frequency_per_activity.json or CSV")
    ap.add_argument("--history_len", type=int, default=5)
    ap.add_argument("--stability_threshold", type=float, default=0.8)
    ap.add_argument("--min_confidence", type=float, default=0.6)
    ap.add_argument("--default_freq", type=str, default="25hz")

    ap.add_argument("--cfg_6", type=str, required=True)
    ap.add_argument("--cfg_12", type=str, required=True)
    ap.add_argument("--cfg_25", type=str, required=True)
    ap.add_argument("--cfg_50", type=str, default="")

    ap.add_argument("--expert_log_root", type=str, default=os.path.join("logs", "deepconvlstm"))
    ap.add_argument("--expert_log_subdir", type=str, default="", help="Optional subdir under expert_log_root, e.g. experiments/...")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--energy_mj", type=str, default="", help="Optional energy per window: '25hz:3.2,12hz:1.9,6hz:1.1'")
    ap.add_argument("--baseline_freq", type=str, default="25hz")
    ap.add_argument("--window_rate_hz", type=float, default=None)

    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    freq_map = load_freq_mapping(args.mapping)

    folds = sorted([d for d in os.listdir(args.test_split_dir) if os.path.isdir(os.path.join(args.test_split_dir, d))])
    if not folds:
        raise ValueError(f"No fold directories found in {args.test_split_dir}")

    all_gt = []
    all_pred = []
    usage_counts: Dict[str, int] = {}

    for fold in folds:
        fold_dir = os.path.join(args.test_split_dir, fold)

        x25, y = _load_npz(os.path.join(fold_dir, "test_windows_25hz.npz"))
        x12, _ = _load_npz(os.path.join(fold_dir, "test_windows_12hz.npz"))
        x6, _ = _load_npz(os.path.join(fold_dir, "test_windows_6hz.npz"))

        x50 = None
        have50 = False
        if args.cfg_50.strip():
            p50 = os.path.join(fold_dir, "test_windows_50hz.npz")
            if os.path.isfile(p50):
                x50, _ = _load_npz(p50)
                have50 = True

        root = args.expert_log_root
        if args.expert_log_subdir.strip():
            root = os.path.join(root, args.expert_log_subdir.strip())

        ck6 = os.path.join(root, "6hz", fold, f"best_{fold}.pth.tar")
        ck12 = os.path.join(root, "12hz", fold, f"best_{fold}.pth.tar")
        ck25 = os.path.join(root, "25hz", fold, f"best_{fold}.pth.tar")
        ck50 = os.path.join(root, "50hz", fold, f"best_{fold}.pth.tar") if have50 else ""

        for p in [ck6, ck12, ck25]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {p}")
        if have50 and (not os.path.isfile(ck50)):
            raise FileNotFoundError(f"Missing 50hz checkpoint for fold {fold}: {ck50}")

        m6 = _load_expert_from_split(args.cfg_6, ck6, device=device)
        m12 = _load_expert_from_split(args.cfg_12, ck12, device=device)
        m25 = _load_expert_from_split(args.cfg_25, ck25, device=device)
        m50 = _load_expert_from_split(args.cfg_50, ck50, device=device) if have50 else None

        cascade = MultiFreqDeepConvLSTM(
            model_6hz=m6,
            model_12hz=m12,
            model_25hz=m25,
            model_50hz=m50,
            freq_for_class=freq_map,
            history_len=args.history_len,
            stability_threshold=args.stability_threshold,
            min_confidence=args.min_confidence,
            default_freq=args.default_freq,
        ).to(device)
        cascade.eval()
        cascade.reset_history()

        y_np = y.astype(np.int64)
        preds_fold = []

        with torch.no_grad():
            for i in range(y_np.shape[0]):
                x25_i = torch.from_numpy(x25[i:i+1]).to(device)
                x12_i = torch.from_numpy(x12[i:i+1]).to(device)
                x6_i = torch.from_numpy(x6[i:i+1]).to(device)
                x50_i = torch.from_numpy(x50[i:i+1]).to(device) if have50 else None

                logits, chosen = cascade.step(
                    x_25hz=x25_i,
                    x_12hz=x12_i,
                    x_6hz=x6_i,
                    x_50hz=x50_i,
                )
                pred = int(torch.argmax(logits, dim=1).item())
                preds_fold.append(pred)

                f = chosen[0]
                usage_counts[f] = int(usage_counts.get(f, 0) + 1)

        all_gt.append(y_np)
        all_pred.append(np.asarray(preds_fold, dtype=np.int64))

        num_classes = int(max(y_np.max(), np.asarray(preds_fold).max())) + 1
        met = _macro_metrics(y_np, np.asarray(preds_fold), num_classes=num_classes)
        print(f"[FOLD {fold}] macro-F1={met['f1_mean']:.2f}  usage_counts={usage_counts}")

    y_all = np.concatenate(all_gt, axis=0)
    p_all = np.concatenate(all_pred, axis=0)
    num_classes = int(max(y_all.max(), p_all.max())) + 1
    met_all = _macro_metrics(y_all, p_all, num_classes=num_classes)

    total = float(sum(usage_counts.values()))
    usage_fracs = {k: float(v) / total for k, v in usage_counts.items()} if total > 0 else {}

    print("\n[ALL FOLDS] macro metrics:")
    for k, v in met_all.items():
        print(f"  {k}: {v:.2f}")
    print("\nFrequency usage:")
    for k in sorted(usage_fracs.keys()):
        print(f"  {k}: {usage_fracs[k]*100.0:6.2f}% ({usage_counts[k]})")

    if args.energy_mj.strip():
        energy = {}
        parts = [p.strip() for p in args.energy_mj.split(",") if p.strip()]
        for p in parts:
            kk, vv = p.split(":", 1)
            energy[kk.strip()] = float(vv.strip())

        rep = estimate_gated_energy(
            usage_fracs=usage_fracs,
            energy_mj=energy,
            baseline_freq=args.baseline_freq,
            window_rate_hz=args.window_rate_hz,
        )
        print("\nEnergy estimate:")
        print(f"  Expected (gated): {rep.expected_energy_mj_per_window:.6f} mJ/window")
        print(f"  Baseline ({args.baseline_freq}): {rep.baseline_energy_mj_per_window:.6f} mJ/window")
        print(f"  Relative savings: {rep.relative_savings_vs_baseline*100.0:+.2f}%")

        if rep.expected_power_w is not None and rep.baseline_power_w is not None:
            print("\nAverage power:")
            print(f"  Expected (gated): {rep.expected_power_w:.6f} W")
            print(f"  Baseline:         {rep.baseline_power_w:.6f} W")


if __name__ == "__main__":
    main()
