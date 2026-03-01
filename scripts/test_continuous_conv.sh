set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"

TEST_SPLIT_DIR="${TEST_SPLIT_DIR:-data/wear/multifreq_test}"

python main_multifreq_final.py \
  --config ./configs/continuous_conv/wear_loso_25hz.yaml \
  --mode test_multifreq \
  --test-split-dir "$TEST_SPLIT_DIR" \
  --ckpt-6hz  logs/deepconvlstm/50hz/loso_sbj_0/best_loso_sbj_0.pth.tar \
  --ckpt-12hz logs/deepconvlstm/12hz/loso_sbj_0/best_loso_sbj_0.pth.tar \
  --ckpt-25hz logs/deepconvlstm/25hz/loso_sbj_0/best_loso_sbj_0.pth.tar \
  --ckpt-50hz logs/deepconvlstm/50hz/loso_sbj_0/best_loso_sbj_0.pth.tar \
  --gamma_quant quantize_and_gamma \
  --quant_bits 4 \
  --gamma_type s_shaped \
  --apply_gamma global