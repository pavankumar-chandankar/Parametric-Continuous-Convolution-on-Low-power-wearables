#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=wisdm_watch_accel_GQ
#SBATCH --output=/home/g263412/GAMMA/slurm_logs/wisdm_watch_accel_%j.out
#SBATCH --error=/home/g263412/GAMMA/slurm_logs/wisdm_watch_accel_%j.err

set -eo pipefail

mkdir -p /home/g263412/GAMMA/slurm_logs
cd /home/g263412/GAMMA

source /etc/profile.d/modules.sh
module load miniconda3/4.9.2
CONDA_BIN=/cm/shared/omni/apps/miniconda3/bin/conda
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate gamma

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Host: $(hostname)"
nvidia-smi || true

python - <<PY
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python main_multifreq.py \
  --config "./configs/continuous_conv/wisdm_watch_accel_loso_Multirate.yaml" \
  --mode train_expert \
  --protocol loso \
  --test-split-dir "data/wisdm/multifreq_test/watch_accel" \
  --gamma_quant "quantize_and_gamma" \
  --quant_bits 4 \
  --gamma_type "s_shaped" \
  --apply_gamma "global" \
  --seed 1
