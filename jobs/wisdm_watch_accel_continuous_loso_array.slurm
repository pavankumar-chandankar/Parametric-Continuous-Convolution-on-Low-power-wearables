#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=wisdm_accel_cont
#SBATCH --output=/home/g263412/GAMMA/slurm_logs/wisdm_accel_cont_%A_%a.out
#SBATCH --error=/home/g263412/GAMMA/slurm_logs/wisdm_accel_cont_%A_%a.err
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

# Array task id is the LOSO split index (1..51)
SPLIT=${SLURM_ARRAY_TASK_ID}

echo "Running WISDM watch accel continuous | split ${SPLIT}/51"

python scripts/run_loso.py \
  --config configs/continuous_conv/wisdm_watch_accel_loso_Multirate.yaml \
  --seed 1 \
  --start_split ${SPLIT} \
  --end_split ${SPLIT}
