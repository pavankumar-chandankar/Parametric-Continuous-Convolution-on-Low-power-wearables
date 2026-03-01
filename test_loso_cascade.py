import argparse
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models.DeepConvLSTM_final import DeepConvLSTM
from models.multifreq_deepconvlstm import MultiFreqDeepConvLSTM, FREQ_FOR_CLASS
from models.train_multifreq import InertialMultiFreqDataset
from utils.logging_utils import classification_scores
from utils.torch_utils_final import worker_init_reset_seed


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _freq_tag_from_cfg(cfg: Dict) -> str:
    sr = float(cfg["dataset"]["sampling_rate"])
    if abs(sr - 50.0) < 1e-3:
        return "50hz"
    if abs(sr - 25.0) < 1e-3:
        return "25hz"
    if abs(sr - 12.0) < 1e-3:
        return "12hz"
    if abs(sr - 6.0) < 1e-3:
        return "6hz"
    # fallback
    return f"{sr:g}hz"


def _available_folds(ckpt_root: str) -> List[int]:
    """
    Expects structure:
      <ckpt_root>/loso_sbj_<k>/best_loso_sbj_<k>.pth.tar
    Returns sorted fold indices found.
    """
    pat = os.path.join(ckpt_root, "loso_sbj_*", "best_loso_sbj_*.pth.tar")
    paths = glob.glob(pat)
    folds = []
    for p in paths:
        base = os.path.basename(p)  # best_loso_sbj_k.pth.tar
        # extract last integer in name
        digits = "".join([c if c.isdigit() else " " for c in base]).split()
        if not digits:
            continue
        k = int(digits[-1])
        folds.append(k)
    return sorted(list(set(folds)))


def _load_test_npz(test_split_dir: str, freq_tag: str) -> Tuple[np.ndarray, np.ndarray]:
    p = os.path.join(test_split_dir, f"test_windows_{freq_tag}.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing test split file: {p}\n"
            f"Expected: {test_split_dir}/test_windows_{freq_tag}.npz"
        )
    d = np.load(p)
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.uint8)
    return x, y


def _align_by_class(
    x_6: np.ndarray,
    y_6: np.ndarray,
    x_12: np.ndarray,
    y_12: np.ndarray,
    x_25: np.ndarray,
    y_25: np.ndarray,
    x_50: Optional[np.ndarray],
    y_50: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align windows across frequencies by taking, for each class c,
    the minimum count across freqs and slicing that many from each.
    This matches your existing cascade evaluation logic.
    """
    n_classes = len(FREQ_FOR_CLASS)

    x50_list: List[np.ndarray] = []
    x6_list: List[np.ndarray] = []
    x12_list: List[np.ndarray] = []
    x25_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    need_50 = (x_50 is not None) and (y_50 is not None)

    for c in range(n_classes):
        idx6 = np.where(y_6 == c)[0]
        idx12 = np.where(y_12 == c)[0]
        idx25 = np.where(y_25 == c)[0]

        if need_50:
            idx50 = np.where(y_50 == c)[0]
            m = min(len(idx6), len(idx12), len(idx25), len(idx50))
        else:
            idx50 = None
            m = min(len(idx6), len(idx12), len(idx25))

        if m == 0:
            continue

        x6_list.append(x_6[idx6[:m]])
        x12_list.append(x_12[idx12[:m]])
        x25_list.append(x_25[idx25[:m]])
        if need_50:
            assert x_50 is not None and idx50 is not None
            x50_list.append(x_50[idx50[:m]])
        y_list.append(np.full(m, c, dtype=np.uint8))

    if not x6_list:
        raise RuntimeError(
            "Could not align any windows across frequencies by label. "
            "Check that your test_windows_*.npz files contain consistent class ids 0..18."
        )

    x_6_al = np.concatenate(x6_list, axis=0)
    x_12_al = np.concatenate(x12_list, axis=0)
    x_25_al = np.concatenate(x25_list, axis=0)
    y_al = np.concatenate(y_list, axis=0)
    x_50_al = np.concatenate(x50_list, axis=0) if need_50 else None

    return x_50_al, x_25_al, x_12_al, x_6_al, y_al


def _build_net_from_ckpt(
    ckpt_path: str,
    x_example: np.ndarray,
    cfg_model: Dict,
    device: torch.device,
    gamma_quant: Optional[str],
    quant_bits: int,
    gamma_type: str,
    apply_gamma: str,
    n_classes: int,
) -> DeepConvLSTM:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Determine state_dict
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

    channels = x_example.shape[2]
    window_size = x_example.shape[1]

    net = DeepConvLSTM(
        channels=channels,
        classes=n_classes,
        window_size=window_size,
        conv_kernels=cfg_model["conv_kernels"],
        conv_kernel_size=cfg_model["conv_kernel_size"],
        lstm_units=cfg_model["lstm_units"],
        lstm_layers=cfg_model["lstm_layers"],
        dropout=cfg_model["dropout"],
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
        conv_type=cfg_model.get("conv_type", "standard"),
        conv_rank=cfg_model.get("conv_rank", 2),
        conv_mlp_hidden_dim=cfg_model.get("conv_mlp_hidden_dim", 64),
    )
    net.load_state_dict(sd)
    net.to(device)
    net.eval()
    return net


@torch.no_grad()
def _eval_fold_cascade(
    cfg_base: Dict,
    ckpt_50: str,
    ckpt_25: str,
    ckpt_12: str,
    ckpt_6: str,
    test_split_dir: str,
    device: torch.device,
    gamma_quant: Optional[str],
    quant_bits: int,
    gamma_type: str,
    apply_gamma: str,
    history_len: int,
    stability_frac: float,
    min_conf: float,
    default_freq: str,
    batch_size: Optional[int] = None,
) -> Dict:
    # Load test windows (numeric labels)
    x6, y6 = _load_test_npz(test_split_dir, "6hz")
    x12, y12 = _load_test_npz(test_split_dir, "12hz")
    x25, y25 = _load_test_npz(test_split_dir, "25hz")
    x50, y50 = _load_test_npz(test_split_dir, "50hz")

    # Align by class counts (same logic as your existing cascade eval)
    x50_al, x25_al, x12_al, x6_al, y_al = _align_by_class(x6, y6, x12, y12, x25, y25, x50, y50)

    assert x50_al is not None
    n_classes = len(FREQ_FOR_CLASS)

    test_ds = InertialMultiFreqDataset(
        x_50=x50_al,
        x_25=x25_al,
        x_12=x12_al,
        x_6=x6_al,
        y=y_al,
    )

    bs = int(batch_size) if batch_size is not None else int(cfg_base["loader"]["test_batch_size"])
    nw = int(cfg_base["loader"].get("num_workers", 0))

    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        worker_init_fn=worker_init_reset_seed,
    )

    cfg_model = cfg_base["model"]

    net_6 = _build_net_from_ckpt(ckpt_6, x6_al, cfg_model, device, gamma_quant, quant_bits, gamma_type, apply_gamma, n_classes)
    net_12 = _build_net_from_ckpt(ckpt_12, x12_al, cfg_model, device, gamma_quant, quant_bits, gamma_type, apply_gamma, n_classes)
    net_25 = _build_net_from_ckpt(ckpt_25, x25_al, cfg_model, device, gamma_quant, quant_bits, gamma_type, apply_gamma, n_classes)
    net_50 = _build_net_from_ckpt(ckpt_50, x50_al, cfg_model, device, gamma_quant, quant_bits, gamma_type, apply_gamma, n_classes)

    multi = MultiFreqDeepConvLSTM(
        model_6hz=net_6,
        model_12hz=net_12,
        model_25hz=net_25,
        model_50hz=net_50,
        freq_for_class=FREQ_FOR_CLASS,
        history_len=history_len,
        stability_threshold=stability_frac,
        min_confidence=min_conf,
        default_freq=default_freq,
    ).to(device)
    multi.eval()
    multi.reset_history()

    all_true: List[int] = []
    all_pred: List[int] = []
    all_freq: List[str] = []

    for x50_b, x25_b, x12_b, x6_b, y_b in test_loader:
        x50_b = x50_b.to(device, non_blocking=True)
        x25_b = x25_b.to(device, non_blocking=True)
        x12_b = x12_b.to(device, non_blocking=True)
        x6_b = x6_b.to(device, non_blocking=True)

        logits, chosen_freqs = multi.step(x_25hz=x25_b, x_12hz=x12_b, x_6hz=x6_b, x_50hz=x50_b)
        preds = torch.argmax(logits, dim=1)

        all_true.extend(y_b.numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_freq.extend(chosen_freqs)

    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)

    _, _, _, f1_per_class = classification_scores(y_true, y_pred, no_classes=len(FREQ_FOR_CLASS))
    macro_f1 = float(np.nanmean(f1_per_class) * 100.0)
    acc = float((y_true == y_pred).mean() * 100.0)

    uniq, cnt = np.unique(np.array(all_freq), return_counts=True)
    freq_usage = {str(u): int(c) for u, c in zip(uniq, cnt)}
    total = int(len(all_freq))
    freq_usage_pct = {k: (100.0 * v / total) for k, v in freq_usage.items()}

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "f1_per_class": (f1_per_class * 100.0).tolist(),
        "freq_usage": freq_usage,
        "freq_usage_pct": freq_usage_pct,
        "n_eval_windows": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser("LOSO fold-by-fold cascade evaluation on held-out test windows (.npz)")

    # configs (only used for model hyperparams + loader batch size)
    ap.add_argument("--config-50", "--cfg-50", dest="config_50", required=False, default="")
    ap.add_argument("--config-25", "--cfg-25", dest="config_25", required=True)
    ap.add_argument("--config-12", "--cfg-12", dest="config_12", required=False, default="")
    ap.add_argument("--config-6", "--cfg-6", dest="config_6", required=False, default="")

    # checkpoint roots
    ap.add_argument("--ckpt-root-50", dest="ckpt_root_50", default="logs/deepconvlstm/50hz")
    ap.add_argument("--ckpt-root-25", dest="ckpt_root_25", default="logs/deepconvlstm/25hz")
    ap.add_argument("--ckpt-root-12", dest="ckpt_root_12", default="logs/deepconvlstm/12hz")
    ap.add_argument("--ckpt-root-6", dest="ckpt_root_6", default="logs/deepconvlstm/6hz")

    ap.add_argument("--test-split-dir", dest="test_split_dir", default="data/wear/multifreq_test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=-1)

    # gamma-quant
    ap.add_argument("--gamma-quant", default=None)
    ap.add_argument("--quant-bits", type=int, default=4)
    ap.add_argument("--gamma-type", default="s_shaped")
    ap.add_argument("--apply-gamma", default="global")

    # cascade params
    ap.add_argument("--history-len", type=int, default=5)
    ap.add_argument("--stability-frac", type=float, default=0.8)
    ap.add_argument("--min-confidence", type=float, default=0.6)
    ap.add_argument("--default-freq", type=str, default="50hz")

    # fold control / output
    ap.add_argument("--only-fold", type=int, default=-1, help="If set (>=0), evaluate only that fold index.")
    ap.add_argument("--out-dir", type=str, default="", help="Optional: directory to write summary CSV.")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    cfg25 = _load_yaml(args.config_25)

    # Decide which folds exist across all four checkpoint roots
    folds_50 = set(_available_folds(args.ckpt_root_50))
    folds_25 = set(_available_folds(args.ckpt_root_25))
    folds_12 = set(_available_folds(args.ckpt_root_12))
    folds_6 = set(_available_folds(args.ckpt_root_6))

    folds = sorted(list(folds_50 & folds_25 & folds_12 & folds_6))
    if not folds:
        raise RuntimeError(
            "No common folds found across ckpt roots.\n"
            f"50hz folds: {sorted(list(folds_50))[:10]}...\n"
            f"25hz folds: {sorted(list(folds_25))[:10]}...\n"
            f"12hz folds: {sorted(list(folds_12))[:10]}...\n"
            f"6hz  folds: {sorted(list(folds_6))[:10]}...\n"
        )

    if args.only_fold >= 0:
        if args.only_fold not in folds:
            raise RuntimeError(f"Requested fold {args.only_fold} not found in common folds: {folds[:10]}...")
        folds = [args.only_fold]

    # Evaluate
    results = []
    for idx, k in enumerate(folds, start=1):
        print("============================================================")
        print(f"[LOSO TEST] Fold {idx}/{len(folds)}  held-out=sbj_{k}")
        print("============================================================")

        split = f"loso_sbj_{k}"
        ckpt_50 = os.path.join(args.ckpt_root_50, split, f"best_{split}.pth.tar")
        ckpt_25 = os.path.join(args.ckpt_root_25, split, f"best_{split}.pth.tar")
        ckpt_12 = os.path.join(args.ckpt_root_12, split, f"best_{split}.pth.tar")
        ckpt_6 = os.path.join(args.ckpt_root_6, split, f"best_{split}.pth.tar")

        r = _eval_fold_cascade(
            cfg_base=cfg25,
            ckpt_50=ckpt_50,
            ckpt_25=ckpt_25,
            ckpt_12=ckpt_12,
            ckpt_6=ckpt_6,
            test_split_dir=args.test_split_dir,
            device=device,
            gamma_quant=args.gamma_quant,
            quant_bits=args.quant_bits,
            gamma_type=args.gamma_type,
            apply_gamma=args.apply_gamma,
            history_len=args.history_len,
            stability_frac=args.stability_frac,
            min_conf=args.min_confidence,
            default_freq=args.default_freq,
            batch_size=(args.batch_size if args.batch_size > 0 else None),
        )

        print(f"Fold sbj_{k}:  Acc={r['acc']:.2f}%  MacroF1={r['macro_f1']:.2f}%  N={r['n_eval_windows']}")
        print("Freq usage (%): " + ", ".join([f"{f}={p:.1f}" for f, p in sorted(r["freq_usage_pct"].items())]))

        results.append({"fold": k, **r})

    # Summary
    accs = [x["acc"] for x in results]
    f1s = [x["macro_f1"] for x in results]
    print("----------------------------------------------------------------")
    print(f"Done. Folds evaluated: {len(results)}")
    print(f"Mean Acc:     {float(np.mean(accs)):.2f}%")
    print(f"Mean MacroF1: {float(np.mean(f1s)):.2f}%")
    print("----------------------------------------------------------------")

    # Optional CSV
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_csv = os.path.join(args.out_dir, "loso_cascade_summary.csv")
        # lightweight CSV without pandas dependency
        header = ["fold", "acc", "macro_f1", "n_eval_windows", "freq_usage_pct"]
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for x in results:
                # store dict as a string
                f.write(f"{x['fold']},{x['acc']:.6f},{x['macro_f1']:.6f},{x['n_eval_windows']},\"{x['freq_usage_pct']}\"\n")
        print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()