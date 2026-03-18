import os
import math
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import convert_samples_to_segments, unwindow_inertial_data
from utils.torch_utils_final import (
    init_weights,
    save_checkpoint,
    worker_init_reset_seed,
    InertialDataset,
)
from utils.os_utils import mkdir_if_missing
from models.DeepConvLSTM_final import DeepConvLSTM
from models.map_metric import ANETdetection


def _pretty_freq_tag_from_len(t: int) -> str:
    t = int(t)
    if t in (50, 25, 20, 15, 12, 10, 6, 5):
        return f"{t}hz"
    return f"{t}t"


def _infer_multirate_lengths_from_model(net: nn.Module):
    # For multi-branch models (continuous or standard_multibranch),
    # infer supported input lengths from branch keys.
    if hasattr(net, "branches"):
        b = getattr(net, "branches")
        if isinstance(b, nn.ModuleDict):
            out = []
            for k in b.keys():
                try:
                    out.append(int(k))
                except Exception:
                    pass
            out = sorted(set(out), reverse=True)
            return out
    return []


def _resample_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resample time dimension for inertial windows.

    Input:  x [B, T, C]
    Output:   [B, target_len, C]
    """
    x = x.contiguous()
    target_len = int(target_len)
    if x.shape[1] == target_len:
        return x
    x_t = x.transpose(1, 2)  # [B, C, T]
    x_rs = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
    return x_rs.transpose(1, 2).contiguous()  # [B, T, C]


def _infer_freq_tag_from_sampling_rate(sampling_rate: float) -> str:
    sr = float(sampling_rate)
    if abs(sr - 50.0) < 1e-3:
        return "50hz"
    if abs(sr - 25.0) < 1e-3:
        return "25hz"
    if abs(sr - 20.0) < 1e-3:
        return "20hz"
    if abs(sr - 15.0) < 1e-3:
        return "15hz"
    if abs(sr - 12.0) < 1e-3 or abs(sr - 12.5) < 1e-3:
        return "12hz"
    if abs(sr - 10.0) < 1e-3:
        return "10hz"
    if abs(sr - 6.0) < 1e-3:
        return "6hz"
    if abs(sr - 5.0) < 1e-3:
        return "5hz"
    return f"{sr:g}hz"


def validate_one_epoch_resampled(loader, network, criterion, target_len: int, gpu=None):
    network.eval()
    losses = []
    preds = np.array([], dtype=np.int64)
    gt = np.array([], dtype=np.int64)

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            if gpu is not None:
                inputs = inputs.to(gpu)
                targets = targets.to(gpu)

            inputs = _resample_time(inputs, target_len)

            outputs = network(inputs)
            batch_loss = criterion(outputs, targets.long())
            losses.append(batch_loss.item())

            batch_preds = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
            batch_gt = targets.cpu().numpy().flatten()

            preds = np.concatenate((preds, batch_preds))
            gt = np.concatenate((gt, batch_gt))

    return losses, preds, gt


def _estimate_model_size_bytes(params: int, bits_per_weight: int) -> int:
    # purely a storage estimate for weights
    if bits_per_weight <= 0:
        raise ValueError("bits_per_weight must be > 0")
    return int(math.ceil(params * bits_per_weight / 8.0))


def run_inertial_network(
    train_sbjs,
    val_sbjs,
    cfg,
    ckpt_folder,
    ckpt_freq,
    resume,
    rng_generator,
    run,
    gamma_quant=None,
    quant_bits=4,
    gamma_type="id",
    apply_gamma="global",
):
    split_name = os.path.splitext(os.path.basename(cfg["dataset"]["json_anno"]))[0]
    sampling_rate = float(cfg["dataset"]["sampling_rate"])

    train_cfg = cfg.get("train_cfg", {})
    freq_tag_override = train_cfg.get("freq_tag_override", None)
    if freq_tag_override is not None and str(freq_tag_override).strip():
        freq_tag = str(freq_tag_override).strip()
    else:
        freq_tag = _infer_freq_tag_from_sampling_rate(sampling_rate)

    # Optional: put experiments under logs/deepconvlstm/<log_subdir>/<freq_tag>/<split>
    log_subdir = train_cfg.get("log_subdir", None)
    if log_subdir is not None and str(log_subdir).strip():
        ckpt_folder = os.path.join("logs", "deepconvlstm", str(log_subdir).strip(), freq_tag, split_name)
    else:
        ckpt_folder = os.path.join("logs", "deepconvlstm", freq_tag, split_name)
    mkdir_if_missing(ckpt_folder)

    last_ckpt_path = os.path.join(ckpt_folder, f"last_{split_name}.pth.tar")

    auto_resume = train_cfg.get("auto_resume", True)
    if (resume is None or resume == "") and auto_resume and os.path.isfile(last_ckpt_path):
        resume = last_ckpt_path
        print(f"Auto-resume: using {resume}")

    train_data = np.empty((0, cfg["dataset"]["input_dim"] + 2))
    val_data = np.empty((0, cfg["dataset"]["input_dim"] + 2))

    for t_sbj in train_sbjs:
        t_path = os.path.join(cfg["dataset"]["sens_folder"], t_sbj + ".csv")
        t_df = pd.read_csv(t_path, index_col=False, low_memory=False)
        t_df = t_df.replace({"label": cfg["label_dict"]}).fillna(0)
        t_arr = t_df.to_numpy()
        train_data = np.append(train_data, t_arr, axis=0)

    for v_sbj in val_sbjs:
        v_path = os.path.join(cfg["dataset"]["sens_folder"], v_sbj + ".csv")
        v_df = pd.read_csv(v_path, index_col=False, low_memory=False)
        v_df = v_df.replace({"label": cfg["label_dict"]}).fillna(0)
        v_arr = v_df.to_numpy()
        val_data = np.append(val_data, v_arr, axis=0)

    min_train = np.min(train_data[:, 1:-1])
    max_train = np.max(train_data[:, 1:-1])
    denom = (max_train - min_train) if (max_train - min_train) != 0 else 1.0

    train_data[:, 1:-1] = 2 * (train_data[:, 1:-1] - min_train) / denom - 1
    val_data[:, 1:-1] = 2 * (val_data[:, 1:-1] - min_train) / denom - 1

    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print("Min/ Max values in training data: [{:.2f}, {:.2f}]".format(
        np.min(train_data[:, 1:-1]), np.max(train_data[:, 1:-1])
    ))
    print("Min/ Max values in validation data: [{:.2f}, {:.2f}]".format(
        np.min(val_data[:, 1:-1]), np.max(val_data[:, 1:-1])
    ))

    train_dataset = InertialDataset(
        train_data,
        cfg["dataset"]["window_size"],
        cfg["dataset"]["window_overlap"],
        model=cfg["name"],
    )
    test_dataset = InertialDataset(
        val_data,
        cfg["dataset"]["window_size"],
        cfg["dataset"]["window_overlap"],
        model=cfg["name"],
    )

    label_dict = cfg.get("label_dict", None)
    if label_dict is not None:
        class_id2name = {v: k for k, v in label_dict.items()}
    else:
        class_id2name = None

    num_workers = int(cfg.get("loader", {}).get("num_workers", 4))
    if os.name == "nt":
        num_workers = 0
    persistent_workers = (num_workers > 0) and (os.name != "nt")

    if cfg["name"] in ["shallow_deepconvlstm", "deepconvcontext"]:
        print("DID NOT SHUFFLE (model {})".format(cfg["name"]))
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["loader"]["train_batch_size"],
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_reset_seed,
            generator=rng_generator,
            persistent_workers=persistent_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["loader"]["train_batch_size"],
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_reset_seed,
            generator=rng_generator,
            persistent_workers=persistent_workers,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["loader"]["test_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_reset_seed,
        generator=rng_generator,
        persistent_workers=persistent_workers,
    )

    if cfg["name"] != "deepconvlstm":
        raise ValueError(f"Unsupported model name '{cfg['name']}'")

    conv_type = cfg["model"].get("conv_type", "standard")
    conv_rank = cfg["model"].get("conv_rank", 8)
    conv_mlp_hidden_dim = cfg["model"].get("conv_mlp_hidden_dim", 32)
    kernel_support_s = cfg["model"].get("kernel_support_s", None)
    standard_padding = cfg["model"].get("standard_padding", "valid")

    supported_sample_rates = cfg["model"].get("supported_sample_rates", (50, 25, 12, 6))
    multirate_training = cfg["model"].get("multirate_training", True)

    net = DeepConvLSTM(
        train_dataset.channels,
        train_dataset.classes,
        train_dataset.window_size,
        cfg["model"]["conv_kernels"],
        cfg["model"]["conv_kernel_size"],
        cfg["model"]["lstm_units"],
        cfg["model"]["lstm_layers"],
        cfg["model"]["dropout"],
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
        conv_type=conv_type,
        conv_rank=conv_rank,
        conv_mlp_hidden_dim=conv_mlp_hidden_dim,
        supported_sample_rates=supported_sample_rates,
        multirate_training=multirate_training,
        kernel_support_s=kernel_support_s,
        standard_padding=standard_padding,
    )

    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    size_fp32 = _estimate_model_size_bytes(params, bits_per_weight=32)
    size_q4 = _estimate_model_size_bytes(params, bits_per_weight=4)

    print(f"Number of learnable parameters for DeepConvLSTM: {params}")
    print(f"Estimated weight storage: FP32 {size_fp32/1e6:.3f} MB | 4-bit {size_q4/1e6:.3f} MB (estimate)")
    print(net)

    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    if gpu is not None:
        net = net.to(gpu)

    criterion = nn.CrossEntropyLoss()

    if train_cfg.get("weighted_loss", False):
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(train_dataset.labels), y=train_dataset.labels
        )
        print("Class weights: {}".format(class_weights))
        w_tensor = torch.FloatTensor(class_weights)
        if gpu is not None:
            w_tensor = w_tensor.to(gpu)
        criterion = nn.CrossEntropyLoss(weight=w_tensor)

    opt = torch.optim.Adam(
        net.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = None
    if train_cfg["lr_step"] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=train_cfg["lr_step"],
            gamma=train_cfg["lr_decay"],
        )

    start_epoch = 0
    best_f1 = -1.0
    best_epoch = -1
    best_v_mAP_vec = None

    best_preds = None
    best_gt = None

    if resume is not None and resume != "":
        if os.path.isfile(resume):
            print(f"Loading checkpoint '{resume}'")
            checkpoint = torch.load(resume, map_location="cpu")
            start_epoch = checkpoint.get("epoch", 0)
            net.load_state_dict(checkpoint["state_dict"])
            opt.load_state_dict(checkpoint["optimizer"])
            best_f1 = checkpoint.get("best_f1", -1.0)
            print(f"Loaded checkpoint (epoch {start_epoch}, best_f1 {best_f1:.2f})")
        else:
            print(f"No checkpoint found at '{resume}'")
    else:
        net = init_weights(net, train_cfg["weight_init"])

    det_eval = ANETdetection(
        cfg["dataset"]["json_anno"],
        "validation",
        tiou_thresholds=cfg["dataset"]["tiou_thresholds"],
    )

    t_losses = []
    v_losses = []
    v_mAP_all = []

    mr_cfg = train_cfg.get("multirate_val_lengths", None)
    if mr_cfg is not None:
        multirate_lengths = [int(x) for x in mr_cfg]
    else:
        multirate_lengths = _infer_multirate_lengths_from_model(net)

    base_len = int(cfg["dataset"]["window_size"])
    multirate_lengths = [int(x) for x in multirate_lengths if int(x) > 0]
    multirate_lengths = sorted(set(multirate_lengths), reverse=True)

    save_per_class_multirate = bool(train_cfg.get("multirate_save_per_class", True))

    # Optional: force training/validation to a specific effective input length.
    resample_train_len = train_cfg.get("resample_train_len", None)
    resample_val_len = train_cfg.get("resample_val_len", None)
    if resample_train_len is not None:
        resample_train_len = int(resample_train_len)
    if resample_val_len is not None:
        resample_val_len = int(resample_val_len)

    # If the model itself does multirate resampling (continuous/standard_multibranch with multirate_training),
    # ignore external resampling to avoid "double resampling".
    model_ref0 = net.module if hasattr(net, "module") else net
    model_handles_multirate = bool(getattr(model_ref0, "multirate_training", False)) and \
        (getattr(model_ref0, "conv_type", "") in ("continuous", "standard_multibranch"))

    if model_handles_multirate:
        if resample_train_len is not None or resample_val_len is not None:
            print("[WARN] resample_train_len/resample_val_len set, but model uses internal multirate training. Ignoring external resampling.")
        resample_train_len = None
        resample_val_len = None

    for epoch in range(start_epoch, train_cfg["epochs"]):
        net.train()
        train_epoch_losses = []

        model_ref = net.module if hasattr(net, "module") else net
        train_rate_counts = {}
        if hasattr(model_ref, "branches"):
            try:
                train_rate_counts = {int(k): 0 for k in model_ref.branches.keys()}
            except Exception:
                train_rate_counts = {}
        elif hasattr(model_ref, "supported_sample_rates"):
            try:
                train_rate_counts = {int(k): 0 for k in model_ref.supported_sample_rates}
            except Exception:
                train_rate_counts = {}

        for _, (inputs, targets) in enumerate(train_loader):
            opt.zero_grad()
            if gpu is not None:
                inputs = inputs.to(gpu)
                targets = targets.to(gpu)

            if resample_train_len is not None:
                inputs = _resample_time(inputs, resample_train_len)

            outputs = net(inputs)

            if train_rate_counts:
                hz = getattr(model_ref, "last_sample_rate", None)
                if hz in train_rate_counts:
                    train_rate_counts[hz] += int(inputs.size(0))

            loss = criterion(outputs, targets.long())
            loss.backward()
            opt.step()

            train_epoch_losses.append(loss.item())

        t_losses.append(float(np.nanmean(train_epoch_losses)))

        train_branch_usage_msg = None
        if train_rate_counts:
            total = sum(train_rate_counts.values())
            if total > 0:
                order = sorted(train_rate_counts.keys(), reverse=True)
                train_branch_usage_msg = "TRAIN BRANCH USAGE: " + ", ".join(
                    f"{h}hz {100.0 * train_rate_counts[h] / total:.1f}% ({train_rate_counts[h]})"
                    for h in order
                )

        # ---- validation (primary) ----
        net.eval()
        with torch.no_grad():
            val_epoch_losses = []
            preds = np.array([], dtype=np.int64)
            gt = np.array([], dtype=np.int64)

            for _, (inputs, targets) in enumerate(test_loader):
                if gpu is not None:
                    inputs = inputs.to(gpu)
                    targets = targets.to(gpu)

                if resample_val_len is not None:
                    inputs = _resample_time(inputs, resample_val_len)

                outputs = net(inputs)
                loss = criterion(outputs, targets.long())
                val_epoch_losses.append(loss.item())

                batch_preds = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
                batch_gt = targets.cpu().numpy().flatten()

                preds = np.concatenate((preds, batch_preds))
                gt = np.concatenate((gt, batch_gt))

        v_losses.append(float(np.nanmean(val_epoch_losses)))

        conf_mat = confusion_matrix(gt, preds, normalize="true")
        v_acc = conf_mat.diagonal()
        v_prec = precision_score(gt, preds, average=None, zero_division=1)
        v_rec = recall_score(gt, preds, average=None, zero_division=1)
        v_f1 = f1_score(gt, preds, average=None, zero_division=1)

        avg_acc = float(np.nanmean(v_acc) * 100.0)
        avg_prec = float(np.nanmean(v_prec) * 100.0)
        avg_rec = float(np.nanmean(v_rec) * 100.0)
        avg_f1 = float(np.nanmean(v_f1) * 100.0)

        # ---- map evaluation is always computed at the dataset sampling rate ----
        unseg_preds, _ = unwindow_inertial_data(
            val_data,
            test_dataset.ids,
            preds,
            cfg["dataset"]["window_size"],
            cfg["dataset"]["window_overlap"],
        )
        v_segments = convert_samples_to_segments(
            val_data[:, 0], unseg_preds, cfg["dataset"]["sampling_rate"]
        )

        v_mAP_vec, _ = det_eval.evaluate(v_segments)
        v_mAP_all.append(v_mAP_vec)

        # ---- optional extra eval lengths (multi-rate evaluation) ----
        extra_eval = {}
        if multirate_lengths:
            for L in multirate_lengths:
                L = int(L)
                # If primary val already uses this length, skip.
                if resample_val_len is not None and int(L) == int(resample_val_len):
                    continue
                if resample_val_len is None and int(L) == int(base_len):
                    continue

                v_losses_L, v_preds_L, v_gt_L = validate_one_epoch_resampled(
                    test_loader, net, criterion, target_len=L, gpu=gpu
                )

                conf_L = confusion_matrix(v_gt_L, v_preds_L, normalize="true")
                v_acc_L = conf_L.diagonal()
                v_prec_L = precision_score(v_gt_L, v_preds_L, average=None, zero_division=1)
                v_rec_L = recall_score(v_gt_L, v_preds_L, average=None, zero_division=1)
                v_f1_L = f1_score(v_gt_L, v_preds_L, average=None, zero_division=1)

                avg_acc_L = float(np.nanmean(v_acc_L) * 100.0)
                avg_prec_L = float(np.nanmean(v_prec_L) * 100.0)
                avg_rec_L = float(np.nanmean(v_rec_L) * 100.0)
                avg_f1_L = float(np.nanmean(v_f1_L) * 100.0)

                tagL = _pretty_freq_tag_from_len(L)
                extra_eval[L] = {
                    "eval_len": L,
                    "freq_tag": tagL,
                    "loss": float(np.nanmean(v_losses_L)),
                    "acc_mean": avg_acc_L,
                    "prec_mean": avg_prec_L,
                    "rec_mean": avg_rec_L,
                    "f1_mean": avg_f1_L,
                    "per_class": (v_acc_L, v_prec_L, v_rec_L, v_f1_L),
                }

                print(f"VALIDATION@{tagL}:\tavg. loss {extra_eval[L]['loss']:.4f}")
                print(
                    f"\t\tAcc {avg_acc_L:4.2f} (%) Prec {avg_prec_L:4.2f} (%) "
                    f"Rec {avg_rec_L:4.2f} (%) F1 {avg_f1_L:4.2f} (%)"
                )

                if run is not None:
                    run[f"val_{tagL}/loss"].log(extra_eval[L]["loss"])
                    run[f"val_{tagL}/acc_mean"].log(avg_acc_L)
                    run[f"val_{tagL}/prec_mean"].log(avg_prec_L)
                    run[f"val_{tagL}/rec_mean"].log(avg_rec_L)
                    run[f"val_{tagL}/f1_mean"].log(avg_f1_L)

        # ---- printing ----
        block1 = "Epoch: [{:03d}/{:03d}]".format(epoch + 1, train_cfg["epochs"])
        block2 = "TRAINING:\tavg. loss {:4.4f}".format(t_losses[-1])
        block3 = "VALIDATION:\tavg. loss {:4.4f}".format(v_losses[-1])

        if v_mAP_vec is None or len(v_mAP_vec) == 0 or np.all(np.isnan(v_mAP_vec)):
            block4 = "\t\tAvg. mAP    n/a"
        else:
            avg_map = float(np.nanmean(v_mAP_vec) * 100.0)
            block4 = "\t\tAvg. mAP {:4.2f} (%)".format(avg_map)
            for tiou, tiou_mAP in zip(cfg["dataset"]["tiou_thresholds"], v_mAP_vec):
                block4 += " mAP@{:.1f} {:4.2f} (%)".format(float(tiou), float(tiou_mAP * 100.0))

        block4 += " Acc {:4.2f} (%) Prec {:4.2f} (%) Rec {:4.2f} (%) F1 {:4.2f} (%)".format(
            avg_acc, avg_prec, avg_rec, avg_f1
        )

        print(block1)
        print(block2)
        if train_branch_usage_msg is not None:
            print(train_branch_usage_msg)
        print(block3)
        print(block4)

        if run is not None:
            run["train/loss"].log(t_losses[-1])
            run["val/loss"].log(v_losses[-1])
            if v_mAP_vec is not None and len(v_mAP_vec) > 0 and not np.all(np.isnan(v_mAP_vec)):
                run["val/mAP"].log(float(np.nanmean(v_mAP_vec) * 100.0))
            run["val/acc_mean"].log(avg_acc)
            run["val/prec_mean"].log(avg_prec)
            run["val/rec_mean"].log(avg_rec)
            run["val/f1_mean"].log(avg_f1)

        # ---- best checkpoint & metric export ----
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch = epoch + 1
            best_v_mAP_vec = v_mAP_vec
            best_preds = preds.copy()
            best_gt = gt.copy()

            ckpt_name = f"best_{split_name}.pth.tar"
            print(
                f"New best F1 {best_f1:.2f}% at epoch {best_epoch}, "
                f"saving checkpoint to {os.path.join(ckpt_folder, ckpt_name)}"
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_f1": best_f1,
                },
                False,
                file_folder=ckpt_folder,
                file_name=ckpt_name,
            )

            # ----- per-class metrics (primary) -----
            num_classes = len(v_acc)
            if class_id2name is not None and len(class_id2name) == num_classes:
                class_names = [class_id2name[i] for i in range(num_classes)]
            else:
                class_names = [f"class_{i}" for i in range(num_classes)]

            primary_eval_len = int(resample_val_len) if resample_val_len is not None else int(base_len)
            primary_tag = str(freq_tag)

            meta_common = {
                "split": split_name,
                "dataset_name": cfg.get("dataset_name", ""),
                "sampling_rate_cfg": sampling_rate,
                "window_size_cfg": int(cfg["dataset"]["window_size"]),
                "eval_len": primary_eval_len,
                "freq_tag": primary_tag,
                "epoch": best_epoch,
                "conv_type": conv_type,
                "conv_kernel_size": int(cfg["model"]["conv_kernel_size"]),
                "standard_padding": str(standard_padding),
                "kernel_support_s": float(kernel_support_s) if kernel_support_s is not None else float(getattr(model_ref0, "kernel_support_s", float("nan"))),
                "gamma_quant": str(gamma_quant) if gamma_quant is not None else "none",
                "quant_bits": int(quant_bits),
                "gamma_type": str(gamma_type),
                "apply_gamma": str(apply_gamma),
                "params": int(params),
                "size_fp32_bytes": int(size_fp32),
                "size_4bit_bytes": int(size_q4),
            }

            per_class_df = pd.DataFrame(
                {
                    **{k: [v] * num_classes for k, v in meta_common.items()},
                    "class_id": list(range(num_classes)),
                    "class_name": class_names,
                    "acc": v_acc,
                    "prec": v_prec,
                    "rec": v_rec,
                    "f1": v_f1,
                }
            )
            per_class_path = os.path.join(ckpt_folder, f"per_class_metrics_{split_name}.csv")
            per_class_df.to_csv(per_class_path, index=False)

            # ----- macro metrics summary (rows: primary + extra eval) -----
            macro_rows = [{
                **meta_common,
                "loss": float(np.nanmean(val_epoch_losses)),
                "acc_mean": avg_acc,
                "prec_mean": avg_prec,
                "rec_mean": avg_rec,
                "f1_mean": avg_f1,
            }]

            if save_per_class_multirate and extra_eval:
                for L, d in extra_eval.items():
                    L = int(L)
                    tagL = str(d["freq_tag"])
                    v_acc_L, v_prec_L, v_rec_L, v_f1_L = d["per_class"]
                    num_classes_L = len(v_acc_L)
                    if class_id2name is not None and len(class_id2name) == num_classes_L:
                        class_names_L = [class_id2name[i] for i in range(num_classes_L)]
                    else:
                        class_names_L = [f"class_{i}" for i in range(num_classes_L)]

                    meta_L = dict(meta_common)
                    meta_L["eval_len"] = L
                    meta_L["freq_tag"] = tagL

                    per_class_df_L = pd.DataFrame(
                        {
                            **{k: [v] * num_classes_L for k, v in meta_L.items()},
                            "class_id": list(range(num_classes_L)),
                            "class_name": class_names_L,
                            "acc": v_acc_L,
                            "prec": v_prec_L,
                            "rec": v_rec_L,
                            "f1": v_f1_L,
                        }
                    )
                    per_class_path_L = os.path.join(ckpt_folder, f"per_class_metrics_{split_name}_{tagL}.csv")
                    per_class_df_L.to_csv(per_class_path_L, index=False)

                    macro_rows.append({
                        **meta_L,
                        "loss": float(d["loss"]),
                        "acc_mean": float(d["acc_mean"]),
                        "prec_mean": float(d["prec_mean"]),
                        "rec_mean": float(d["rec_mean"]),
                        "f1_mean": float(d["f1_mean"]),
                    })

            best_macro_path = os.path.join(ckpt_folder, f"best_macro_metrics_{split_name}.csv")
            pd.DataFrame(macro_rows).to_csv(best_macro_path, index=False)

        # always save last
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": opt.state_dict(),
                "best_f1": best_f1,
            },
            False,
            file_folder=ckpt_folder,
            file_name=f"last_{split_name}.pth.tar",
        )

        if scheduler is not None:
            scheduler.step()

    if best_v_mAP_vec is None:
        v_mAP_best = np.array([], dtype=float)
    else:
        v_mAP_best = np.array(best_v_mAP_vec, dtype=float)

    # return BEST epoch preds/gt if available
    if best_preds is None:
        best_preds = preds
    if best_gt is None:
        best_gt = gt

    return t_losses, v_losses, v_mAP_best, best_preds, best_gt, net


def validate_one_epoch(loader, network, criterion, gpu=None):
    network.eval()
    losses = []
    preds = np.array([], dtype=np.int64)
    gt = np.array([], dtype=np.int64)

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            if gpu is not None:
                inputs = inputs.to(gpu)
                targets = targets.to(gpu)

            outputs = network(inputs)
            batch_loss = criterion(outputs, targets.long())
            losses.append(batch_loss.item())

            batch_preds = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
            batch_gt = targets.cpu().numpy().flatten()

            preds = np.concatenate((preds, batch_preds))
            gt = np.concatenate((gt, batch_gt))

    return losses, preds, gt
