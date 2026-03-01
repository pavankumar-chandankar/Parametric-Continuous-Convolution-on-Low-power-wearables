#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=sbhar_multirate_noGQ
#SBATCH --output=/home/g263412/SBHAR/slurm_logs/%x_%j.out
#SBATCH --error=/home/g263412/SBHAR/slurm_logs/%x_%j.err
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
bash -u main_multifreq.py \
  --config configs/continuous_conv/sbhar_loso_50hz.yaml \
  --mode train_expert \
  --protocol loso \
  --freq-tag 50hz \
  --seed 1 \
  --test-split-dir data/sbhar/multifreq_test
