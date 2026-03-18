#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=wisdm_accel_ksweep
#SBATCH --output=/home/g263412/GAMMA/slurm_logs/wisdm_accel_ksweep_%A_%a.out
#SBATCH --error=/home/g263412/GAMMA/slurm_logs/wisdm_accel_ksweep_%A_%a.err
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

# -----------------------
# Sweep settings
# -----------------------

# Kernel list for full sweep (3..21 odd)
KERNELS=(3 5 7 9 11 13 15 17 19 21)

# Total LOSO splits for your WISDM watch accel config
TOTAL_SPLITS=51

# How many splits to process per task (chunk size)
SPLITS_PER_TASK=3

# Derived
CHUNKS_PER_KERNEL=$(( (TOTAL_SPLITS + SPLITS_PER_TASK - 1) / SPLITS_PER_TASK ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

KERNEL_INDEX=$(( TASK_ID / CHUNKS_PER_KERNEL ))
CHUNK_INDEX=$(( TASK_ID % CHUNKS_PER_KERNEL ))

K=${KERNELS[$KERNEL_INDEX]}

START_SPLIT=$(( CHUNK_INDEX * SPLITS_PER_TASK + 1 ))
END_SPLIT=$(( (CHUNK_INDEX + 1) * SPLITS_PER_TASK ))
if [ ${END_SPLIT} -gt ${TOTAL_SPLITS} ]; then
  END_SPLIT=${TOTAL_SPLITS}
fi

echo "TASK_ID=${TASK_ID}"
echo "KERNEL_INDEX=${KERNEL_INDEX}  KERNEL=${K}"
echo "CHUNK_INDEX=${CHUNK_INDEX}   SPLITS=${START_SPLIT}..${END_SPLIT}"

# Run: train once (20Hz), eval at 10Hz + 5Hz at best epoch
python scripts/run_kernel_sweep_evalonly.py \
  --base_config configs/continuous_conv/wisdm_watch_accel_loso_Multirate.yaml \
  --eval_lengths 20,10,5 \
  --exp_tag experiments/kernel_sweep_full/wisdm_watch_accel/std_evalonly \
  --kernel_min ${K} --kernel_max ${K} --kernel_step 2 --odd_only \
  --start_split ${START_SPLIT} --end_split ${END_SPLIT}
