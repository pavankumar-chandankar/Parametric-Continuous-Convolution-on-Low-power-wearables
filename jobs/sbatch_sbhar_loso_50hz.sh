#!/usr/bin/env bash
#SBATCH --job-name=sbhar_loso_50hz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/sbhar_loso_50hz_%j.out
#SBATCH --error=slurm_logs/sbhar_loso_50hz_%j.err

set -euo pipefail
: "${BASHRCSOURCED:=1}"
cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_logs

set +u
source ~/.bashrc
set -u
conda activate gamma

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"

# IMPORTANT: remove incompatible old checkpoints (id-column changed the model input)
rm -rf logs/deepconvlstm_sbhar/50hz

python main_multifreq.py \
  --config configs/continuous_conv/sbhar_loso_50hz.yaml \
  --mode train_expert \
  --protocol loso \
  --freq-tag 50hz \
  --seed 1 \
  --test-split-dir data/sbhar/multifreq_test
