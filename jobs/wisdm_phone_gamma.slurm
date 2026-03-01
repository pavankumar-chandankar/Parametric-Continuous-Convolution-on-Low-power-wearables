#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=wisdm_phone_multirate_GQ
#SBATCH --output=/home/g263412/GAMMA/slurm_logs/wisdm_phone_gamma.out
#SBATCH --error=/home/g263412/GAMMA/slurm_logs/wisdm_phone_gamma.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pavankumar.chandankar@student.uni-siegen.de

PS1="${PS1:-}"
set -eo pipefail

cd /home/g263412/GAMMA
mkdir -p slurm_logs

# Conda
source /etc/profile.d/modules.sh
module load miniconda3/4.9.2
CONDA_BIN=/cm/shared/omni/apps/miniconda3/bin/conda
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate gamma

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Optional: keep these explicit (only if your script reads them)
export PROTOCOL="${PROTOCOL:-loso}"
export SEED="${SEED:-1}"

# Run training
python main_multifreq.py \
  --config "./configs/continuous_conv/wisdm_phone_loso_Multirate.yaml" \
  --mode train_expert \
  --protocol loso \
  --test-split-dir "data/wisdm/multifreq_test/phone" \
  --gamma_quant "quantize_and_gamma" \
  --quant_bits 4 \
  --gamma_type "s_shaped" \
  --apply_gamma "global" \
  --seed 1
