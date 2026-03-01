# scripts/test_deepconvlstm.sh
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"

TEST_SPLIT_DIR="${TEST_SPLIT_DIR:-data/wear/multifreq_test}"

CKPT_6="${CKPT_6:-checkpoints/deepconvlstm/best_6hz.pth.tar}"
CKPT_12="${CKPT_12:-checkpoints/deepconvlstm/best_12hz.pth.tar}"
CKPT_25="${CKPT_25:-checkpoints/deepconvlstm/best_25hz.pth.tar}"
CKPT_50="${CKPT_50:-checkpoints/deepconvlstm/best_50hz.pth.tar}"

CMD=(python main_multifreq.py
  --config ./configs/deepconvlstm/wear_loso_25hz.yaml
  --mode test_multifreq
  --test-split-dir "$TEST_SPLIT_DIR"
  --ckpt-6hz  "$CKPT_6"
  --ckpt-12hz "$CKPT_12"
  --ckpt-25hz "$CKPT_25"
  --ckpt-50hz "$CKPT_50"  # Placeholder, may be overridden below
  --gamma_quant quantize_and_gamma
  --quant_bits 4
  --gamma_type s_shaped
  --apply_gamma global
)

# If 50 Hz ckpt exists, enable 4-frequency cascade (50/25/12/6)
if [[ -f "$CKPT_50" ]]; then
  CMD+=(--ckpt-50hz "$CKPT_50")
else
  echo "[info] 50 Hz checkpoint not found at: $CKPT_50 (running 3-frequency cascade)"
fi

"${CMD[@]}"