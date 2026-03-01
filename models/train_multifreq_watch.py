# models/train_multifreq.py
import os
import time
from typing import Dict, List, Tuple, Optional
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, message=r"Downcasting behavior in `replace` is deprecated.*")
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

import torch
import torch.nn as nn
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from models.DeepConvLSTM_final import DeepConvLSTM
from models.multifreq_deepconvlstm import MultiFreqDeepConvLSTM, FREQ_FOR_CLASS
from models.train_watch import run_inertial_network
from utils.torch_utils_final import fix_random_seed, worker_init_reset_seed


class InertialWindowDataset(Dataset):
    """
    Windowed dataset: (x_windows, y_labels), with optional context features depending on model_name.
    This is the legacy 70/20/10 path.
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray, model_name: str = "deepconvlstm"):
        super().__init__()
        self.windows = windows
        self.labels = labels
        self.model_name = model_name

        # windows expected shape: [N, T, C]
        assert self.windows.ndim == 3, f"Expected windows [N,T,C], got {self.windows.shape}"
        self.window_size = self.windows.shape[1]
        self.channels = self.windows.shape[2]

        # labels are integer class ids
        self.classes = int(np.max(self.labels)) + 1 if len(self.labels) else 0

        self.context_features = 0
        if self.model_name in ["deepconvcontext", "shallow_deepconvlstm"]:
            self.context_features = 4

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.windows[idx]  # [T,C]
        y = self.labels[idx]

        if self.context_features > 0:
            ctx = np.zeros((x.shape[0], self.context_features), dtype=x.dtype)
            x = np.concatenate([x, ctx], axis=1)

        # DeepConvLSTM expects [B,T,C] (train.py pipeline uses InertialDataset);
        # legacy path keeps [1,T,C] per sample then batched -> [B,1,T,C] for old code.
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.tensor(y).long()
        return x, y


def _freq_tag_from_sampling_rate(sampling_rate: float) -> str:
    if abs(sampling_rate - 20.0) < 1e-3:
        return "20hz"
    if abs(sampling_rate - 15.0) < 1e-3:
        return "15hz"
    if abs(sampling_rate - 10.0) < 1e-3:
        return "10hz"
    if abs(sampling_rate - 5.0) < 1e-3:
        return "5hz"
    return f"{sampling_rate:g}hz"


def build_single_freq_splits(
    cfg: Dict,
    sbj_list: List[str],
    freq_tag: str,
    seed: int,
) -> Tuple[InertialWindowDataset, InertialWindowDataset, Tuple[np.ndarray, np.ndarray]]:
    """
    70/20/10 split (legacy):
      - loads all subjects from CSV
      - normalizes features to [-1,1]
      - windows and random split
    """
    sens_folder = cfg["dataset"]["sens_folder"]

    all_rows: List[np.ndarray] = []
    for sbj in sbj_list:
        p_csv = os.path.join(sens_folder, f"{sbj}.csv")
        if not os.path.isfile(p_csv):
            continue

        df = pd.read_csv(p_csv, index_col=False, low_memory=False)
        if "label_dict" in cfg:
            df = df.replace({"label": cfg["label_dict"]})
        df = df.fillna(0)
        arr = df.to_numpy()

        if arr.shape[0] == 0:
            continue
        all_rows.append(arr)

    if not all_rows:
        raise RuntimeError(f"[{freq_tag}] No subject CSVs found in {sens_folder}")

    data = np.concatenate(all_rows, axis=0)

    feat_cols = slice(1, -1)
    feat_min = data[:, feat_cols].min()
    feat_max = data[:, feat_cols].max()
    denom = (feat_max - feat_min) if (feat_max - feat_min) != 0 else 1.0
    data[:, feat_cols] = 2.0 * (data[:, feat_cols] - feat_min) / denom - 1.0

    print(f"[{freq_tag}] All data shape: {data.shape}")
    print(f"[{freq_tag}] Min/Max after normalization: {data[:, feat_cols].min():.2f}, {data[:, feat_cols].max():.2f}")

    window_size = int(cfg["dataset"]["window_size"])
    window_overlap = float(cfg["dataset"]["window_overlap"])

    step = int(window_size * (1 - window_overlap))
    if step <= 0:
        step = 1

    windows = []
    labels = []

    for start in range(0, data.shape[0] - window_size + 1, step):
        win = data[start : start + window_size, feat_cols]
        lab = int(data[start + window_size - 1, -1])
        windows.append(win)
        labels.append(lab)

    windows = np.asarray(windows, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    x_train, y_train = windows[train_idx], labels[train_idx]
    x_val, y_val = windows[val_idx], labels[val_idx]
    x_test, y_test = windows[test_idx], labels[test_idx]

    print(f"[{freq_tag}] Split sizes (train / val / test): {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    train_ds = InertialWindowDataset(x_train, y_train, model_name=cfg.get("name", "deepconvlstm"))
    val_ds = InertialWindowDataset(x_val, y_val, model_name=cfg.get("name", "deepconvlstm"))

    return train_ds, val_ds, (x_test, y_test)


def _prepare_split_from_anno(cfg: Dict, anno_path: str, sbj_list: Optional[List[str]] = None) -> Tuple[str, List[str], List[str]]:
    split_name = os.path.splitext(os.path.basename(anno_path))[0]

    with open(anno_path, "r") as f:
        anno = json.load(f)

    db = anno.get("database", None)
    if not isinstance(db, dict) or len(db) == 0:
        raise KeyError(f"'database' not found or empty in annotation file: {anno_path}")

    train_sbjs: List[str] = []
    val_sbjs: List[str] = []

    for sbj_id, meta in db.items():
        if not isinstance(meta, dict):
            continue

        subset = meta.get("subset", meta.get("Subset", ""))
        subset_l = str(subset).strip().lower()

        if subset_l.startswith("train"):
            train_sbjs.append(sbj_id)
        elif subset_l.startswith("val"):
            val_sbjs.append(sbj_id)

    if len(train_sbjs) > 0 and len(val_sbjs) > 0:
        return split_name, sorted(train_sbjs), sorted(val_sbjs)

    if sbj_list is None or len(sbj_list) == 0:
        raise KeyError(
            f"No per-subject 'subset' entries found in {anno_path}, and sbj_list fallback is unavailable."
        )

    try:
        held_out_id = int(split_name.split("_")[-1])
    except Exception as e:
        raise KeyError(
            f"No per-subject 'subset' entries found in {anno_path}, and could not parse held-out id from '{split_name}'."
        ) from e

    held_out_name = f"sbj_{held_out_id}"
    if held_out_name in sbj_list:
        val_sbjs = [held_out_name]
        train_sbjs = [s for s in sbj_list if s != held_out_name]
    else:
        if held_out_id < 0 or held_out_id >= len(sbj_list):
            raise IndexError(f"held_out_id={held_out_id} out of range for sbj_list (len={len(sbj_list)})")
        val_sbjs = [sbj_list[held_out_id]]
        train_sbjs = [s for i, s in enumerate(sbj_list) if i != held_out_id]

    return split_name, train_sbjs, val_sbjs


def run_inertial_expert_loso(
    cfg: Dict,
    sbj_list: List[str],
    log_dir: str,
    seed: int,
    test_split_dir: str,
    device: torch.device,
    gamma_quant=None,
    quant_bits: int = 4,
    gamma_type: str = "s_shaped",
    apply_gamma: str = "global",
    resume: str = "",
    ckpt_freq: int = -1,
) -> List[str]:
    """
    LOSO per frequency. Uses models/train.py::run_inertial_network so logs/checkpoints match train.py.
    """
    os.makedirs(log_dir, exist_ok=True)
    rng_generator = fix_random_seed(cfg.get("init_rand_seed", seed), include_cuda=True)

    sampling_rate = float(cfg["dataset"].get("sampling_rate", 0.0))
    freq_tag = _freq_tag_from_sampling_rate(sampling_rate)

    # Optional: held-out test windows for cascade eval (don’t crash if legacy build fails)
    os.makedirs(test_split_dir, exist_ok=True)
    test_npz_path = os.path.join(test_split_dir, f"test_windows_{freq_tag}.npz")
    if not os.path.exists(test_npz_path):
        try:
            _, _, (x_test, y_test) = build_single_freq_splits(cfg, sbj_list, freq_tag, seed)
            np.savez_compressed(test_npz_path, x=x_test, y=y_test)
            print(f"[{freq_tag}] Saved held-out test windows to: {test_npz_path}")
        except Exception as e:
            print(f"[{freq_tag}] Warning: could not create held-out test windows ({e}). Continuing training.")

    best_ckpts: List[str] = []
    epochs = int(cfg.get("train_cfg", {}).get("epochs", 0))

    for i, anno_path in enumerate(cfg.get("anno_json", [])):
        split_name, train_sbjs, val_sbjs = _prepare_split_from_anno(cfg, anno_path, sbj_list=sbj_list)
        print(f"Split {i + 1} / {len(cfg['anno_json'])} ({freq_tag}) -> {split_name}")

        ckpt_dir = os.path.join("logs", "continuous_conv", "WISDM_watch_No_GAMMA", split_name)
        last_ckpt = os.path.join(ckpt_dir, f"last_{split_name}.pth.tar")
        best_ckpt = os.path.join(ckpt_dir, f"best_{split_name}.pth.tar")

        fold_resume = resume
        if (fold_resume is None or fold_resume == "") and epochs > 0:
            cand = last_ckpt if os.path.isfile(last_ckpt) else (best_ckpt if os.path.isfile(best_ckpt) else "")
            if cand:
                try:
                    ck = torch.load(cand, map_location="cpu")
                    done_epoch = int(ck.get("epoch", 0)) if isinstance(ck, dict) else 0
                except Exception:
                    done_epoch = 0

                if done_epoch >= epochs and os.path.isfile(best_ckpt):
                    print(f"[SKIP] {split_name} already finished (epoch {done_epoch}/{epochs}).")
                    best_ckpts.append(best_ckpt)
                    continue

                fold_resume = cand
                if done_epoch > 0:
                    print(f"[RESUME] {split_name} from {cand} (epoch {done_epoch}/{epochs}).")

        cfg["dataset"]["json_anno"] = anno_path
        run_inertial_network(
            train_sbjs=train_sbjs,
            val_sbjs=val_sbjs,
            cfg=cfg,
            ckpt_folder=log_dir,
            ckpt_freq=ckpt_freq,
            resume=fold_resume,
            rng_generator=rng_generator,
            run=None,
            gamma_quant=gamma_quant,
            quant_bits=quant_bits,
            gamma_type=gamma_type,
            apply_gamma=apply_gamma,
        )

        ckpt_path = os.path.join("logs", "continuous_conv", "WISDM_watch", split_name, f"best_{split_name}.pth.tar")
        best_ckpts.append(ckpt_path)

    return best_ckpts


def run_inertial_expert_702010(
    cfg: Dict,
    sbj_list: List[str],
    log_dir: str,
    freq_tag: str,
    seed: int,
    test_split_dir: str,
    device: torch.device,
    gamma_quant=None,
    quant_bits: int = 4,
    gamma_type: str = "s_shaped",
    apply_gamma: str = "global",
) -> str:
    """
    Legacy 70/20/10 training (kept for backwards compatibility with main_multifreq imports).
    """
    os.makedirs(log_dir, exist_ok=True)
    rng_generator = fix_random_seed(cfg.get("init_rand_seed", seed), include_cuda=True)

    train_ds, val_ds, (x_test, y_test) = build_single_freq_splits(cfg, sbj_list, freq_tag, seed)

    os.makedirs(test_split_dir, exist_ok=True)
    np.savez_compressed(os.path.join(test_split_dir, f"test_windows_{freq_tag}.npz"), x=x_test, y=y_test)

    channels = train_ds.channels
    n_classes = train_ds.classes
    window_size = train_ds.window_size
    net_cfg = cfg["model"]

    supported_sample_rates = net_cfg.get("supported_sample_rates", (50, 25, 12, 6))
    multirate_training = net_cfg.get("multirate_training", True)

    net = DeepConvLSTM(
        channels=channels,
        classes=n_classes,
        window_size=window_size,
        conv_kernels=net_cfg["conv_kernels"],
        conv_kernel_size=net_cfg["conv_kernel_size"],
        lstm_units=net_cfg["lstm_units"],
        lstm_layers=net_cfg["lstm_layers"],
        dropout=net_cfg["dropout"],
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
        conv_type=net_cfg.get("conv_type", "standard"),
        conv_rank=net_cfg.get("conv_rank", 2),
        conv_mlp_hidden_dim=net_cfg.get("conv_mlp_hidden_dim", 64),
        supported_sample_rates=supported_sample_rates,
        multirate_training=multirate_training,
    )

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of learnable parameters for DeepConvLSTM: {n_params}")
    print(net)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    if cfg.get("weighted_loss", False):
        y_train = train_ds.labels
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Class weights: {weights.detach().cpu().numpy()}")

    opt = torch.optim.Adam(
        net.parameters(),
        lr=cfg["train_cfg"]["lr"],
        weight_decay=cfg["train_cfg"]["weight_decay"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["loader"]["train_batch_size"],
        shuffle=True,
        num_workers=0 if os.name == "nt" else 4,
        worker_init_fn=worker_init_reset_seed,
        generator=rng_generator,
        persistent_workers=False if os.name == "nt" else True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["loader"]["test_batch_size"],
        shuffle=False,
        num_workers=0 if os.name == "nt" else 4,
        worker_init_fn=worker_init_reset_seed,
        generator=rng_generator,
        persistent_workers=False if os.name == "nt" else True,
    )

    best_f1 = -1.0
    best_path = os.path.join(log_dir, f"best_{freq_tag}.pth.tar")

    for epoch in range(cfg["train_cfg"]["epochs"]):
        net.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        net.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = net(xb)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.append(pred)
                gts.append(yb.numpy())

        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        f1 = (pd.Series(preds).groupby(gts).apply(lambda s: (s == s.name).mean()).mean())
        print(f"[{freq_tag}] Epoch {epoch+1:03d}/{cfg['train_cfg']['epochs']:03d} train loss {np.mean(train_losses):.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_f1": best_f1,
                },
                best_path,
            )
            print(f"[{freq_tag}] New best -> {best_path} (F1={best_f1:.4f})")

    return best_path


def test_multifreq_cascade(
    cfg: Dict,
    ckpt_6hz: str,
    ckpt_12hz: str,
    ckpt_25hz: str,
    test_split_dir: str,
    device: torch.device,
    ckpt_50hz: str = "",
    gamma_quant=None,
    quant_bits: int = 4,
    gamma_type: str = "s_shaped",
    apply_gamma: str = "global",
) -> None:
    """
    Robust-ish wrapper: tries to work for both WEAR (6/12/25/50) and WISDM (5/10/15/20),
    depending on which test_windows_*.npz files exist in test_split_dir.

    NOTE: This function depends on your MultiFreqDeepConvLSTM implementation.
    """

    # Detect available test window sets
    sets = [
        ("6hz", "12hz", "25hz", "50hz"),
        ("5hz", "10hz", "15hz", "20hz"),
    ]

    chosen = None
    for a, b, c, d in sets:
        pa = os.path.join(test_split_dir, f"test_windows_{a}.npz")
        pb = os.path.join(test_split_dir, f"test_windows_{b}.npz")
        pc = os.path.join(test_split_dir, f"test_windows_{c}.npz")
        pd = os.path.join(test_split_dir, f"test_windows_{d}.npz")
        if os.path.exists(pa) and os.path.exists(pb) and os.path.exists(pc):
            chosen = (a, b, c, d)
            break

    if chosen is None:
        raise FileNotFoundError(f"Missing test windows in {test_split_dir} (need either 6/12/25[/50] or 5/10/15[/20]).")

    a, b, c, d = chosen
    path_a = os.path.join(test_split_dir, f"test_windows_{a}.npz")
    path_b = os.path.join(test_split_dir, f"test_windows_{b}.npz")
    path_c = os.path.join(test_split_dir, f"test_windows_{c}.npz")
    path_d = os.path.join(test_split_dir, f"test_windows_{d}.npz")

    data_a = np.load(path_a)
    data_b = np.load(path_b)
    data_c = np.load(path_c)

    x_a, y = data_a["x"], data_a["y"]
    x_b = data_b["x"]
    x_c = data_c["x"]

    use_d = bool(ckpt_50hz)
    x_d = None
    if use_d:
        if not os.path.exists(path_d):
            raise FileNotFoundError(f"Missing test windows file for {d}: {path_d}")
        data_d = np.load(path_d)
        x_d = data_d["x"]

    # Instantiate MultiFreqDeepConvLSTM with a signature fallback
    try:
        model = MultiFreqDeepConvLSTM(
            cfg=cfg,
            ckpt_6hz=ckpt_6hz,
            ckpt_12hz=ckpt_12hz,
            ckpt_25hz=ckpt_25hz,
            ckpt_50hz=ckpt_50hz,
            device=device,
            gamma_quant=gamma_quant,
            quant_bits=quant_bits,
            gamma_type=gamma_type,
            apply_gamma=apply_gamma,
        )
    except TypeError:
        model = MultiFreqDeepConvLSTM(
            cfg=cfg,
            ckpt_5hz=ckpt_6hz,
            ckpt_10hz=ckpt_12hz,
            ckpt_15hz=ckpt_25hz,
            ckpt_20hz=ckpt_50hz,
            device=device,
            gamma_quant=gamma_quant,
            quant_bits=quant_bits,
            gamma_type=gamma_type,
            apply_gamma=apply_gamma,
        )

    model.eval()

    ta = torch.from_numpy(x_a).float().unsqueeze(1).to(device)
    tb = torch.from_numpy(x_b).float().unsqueeze(1).to(device)
    tc = torch.from_numpy(x_c).float().unsqueeze(1).to(device)
    td = torch.from_numpy(x_d).float().unsqueeze(1).to(device) if use_d else None

    with torch.no_grad():
        # Forward-call signature fallback
        try:
            logits = model(x_6hz=ta, x_12hz=tb, x_25hz=tc, x_50hz=td)
        except TypeError:
            logits = model(x_5hz=ta, x_10hz=tb, x_15hz=tc, x_20hz=td)

        pred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = float((pred == y).mean()) * 100.0
    print(f"[cascade] Accuracy: {acc:.2f}% (used_highest={use_d}, window_set={a}/{b}/{c}/{d})")
