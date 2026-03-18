#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=wisdm_accel_cont_nog
#SBATCH --output=/home/g263412/GAMMA/slurm_logs/wisdm_accel_cont_nog_%A_%a.out
#SBATCH --error=/home/g263412/GAMMA/slurm_logs/wisdm_accel_cont_nog_%A_%a.err
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

SPLIT=${SLURM_ARRAY_TASK_ID}

BASE_CFG="configs/continuous_conv/wisdm_watch_accel_loso_Multirate.yaml"
PATCHED_CFG="/tmp/wisdm_watch_accel_cont_nogamma_split${SPLIT}.yaml"

# Patch config on the fly:
# - ensure conv_type=continuous
# - disable auto_resume to avoid loading old gamma checkpoints
# - write logs under a fresh namespace
python - <<PY
import yaml
base_cfg = "${BASE_CFG}"
patched_cfg = "${PATCHED_CFG}"

with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("dataset_name", "wisdm_watch_accel")
cfg.setdefault("model", {})
cfg["model"]["conv_type"] = "continuous"

cfg.setdefault("train_cfg", {})
cfg["train_cfg"]["auto_resume"] = False
cfg["train_cfg"]["log_subdir"] = "experiments/continuous_nogamma/wisdm_watch_accel"

with open(patched_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote:", patched_cfg)
PY

echo "Running split ${SPLIT}/51 with NO-GAMMA continuous conv"

python scripts/run_loso.py \
  --config "${PATCHED_CFG}" \
  --seed 1 \
  --start_split ${SPLIT} \
  --end_split ${SPLIT}
