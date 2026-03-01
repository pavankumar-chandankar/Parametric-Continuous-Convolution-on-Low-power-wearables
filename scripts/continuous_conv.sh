#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"

SEED="${SEED:-1}"
TEST_SPLIT_DIR="${TEST_SPLIT_DIR:-data/wear/multifreq_test}"
PROTOCOL="${PROTOCOL:-loso}"

BASE_CFG="./configs/continuous_conv/wear_loso_Multirate.yaml"
FREQ_TAG="50hz"

OUT_DIR="checkpoints/continuous_conv_multirate_wear"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "[TRAIN] WEAR ContinuousConv MULTIRATE (single run, 4-branch ckpt) | protocol=${PROTOCOL}"
echo "Config: ${BASE_CFG}"
echo "Test split dir: ${TEST_SPLIT_DIR}"
echo "Gamma-Quant: OFF"
echo "============================================================"

python main_multifreq.py \
  --config "${BASE_CFG}" \
  --mode train_expert \
  --protocol "${PROTOCOL}" \
  --freq-tag "${FREQ_TAG}" \
  --test-split-dir "${TEST_SPLIT_DIR}" \
  --seed "${SEED}"

echo "============================================================"
echo "[COPY] Copying per-split best checkpoints (each contains branches 50/25/12/6) to: ${OUT_DIR}"
echo "============================================================"

python - <<'PY'
import glob
import os
import shutil
import torch

src = sorted(glob.glob(os.path.join("logs","deepconvlstm","Multirate","loso_sbj_*","best_*.pth.tar")))
if not src:
    raise SystemExit("No best_*.pth.tar found under logs/deepconvlstm/50hz/loso_sbj_*/")

dst_dir = "checkpoints/continuous_conv_multirate_wear"
os.makedirs(dst_dir, exist_ok=True)

def has_branch(sd, key):
    if key in sd:
        return True
    if ("module."+key) in sd:
        return True
    return False

need = [
    "branches.50.0.channel_weights",
    "branches.25.0.channel_weights",
    "branches.12.0.channel_weights",
    "branches.6.0.channel_weights",
]

ok = 0
for p in src:
    obj = torch.load(p, map_location="cpu", weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    if not all(has_branch(sd, k) for k in need):
        raise SystemExit(f"[ERROR] {p} is not a 4-branch multirate ckpt (missing branch keys).")

    out = os.path.join(dst_dir, os.path.basename(os.path.dirname(p)) + "_" + os.path.basename(p))
    shutil.copy2(p, out)
    ok += 1

print(f"[OK] Copied {ok} multirate best checkpoints to {dst_dir}")
PY

echo "============================================================"
echo "[DONE] WEAR multirate training complete."
echo "Each copied ckpt contains 4 branches: 50/25/12/6."
echo "============================================================"

