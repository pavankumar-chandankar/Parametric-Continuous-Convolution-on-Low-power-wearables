#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=rwhar_gq4
#SBATCH --output=/home/g263412/GAMMA/slurm_logs/rwhar_gq4_%A_%a.out
#SBATCH --error=/home/g263412/GAMMA/slurm_logs/rwhar_gq4_%A_%a.err
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

BASE_CFG="configs/continuous_conv/realworld_waist_accel_loso_Multirate.yaml"
PATCHED_CFG="/tmp/realworld_waist_accel_gamma4_split${SPLIT}.yaml"

# Patch YAML on-the-fly so we:
# - keep same dataset/model
# - enable GammaQuant (4-bit)
# - write logs to a separate folder (no collision with no-gamma)
python - <<PY
import yaml
base_cfg = "${BASE_CFG}"
patched_cfg = "${PATCHED_CFG}"

with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("train_cfg", {})
cfg["train_cfg"]["log_subdir"] = "experiments/continuous_gamma4/realworld_waist_accel"
cfg["train_cfg"]["auto_resume"] = True  # safe because log_subdir is new

with open(patched_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote:", patched_cfg)
PY

echo "Running RealWorld waist-accel LOSO split ${SPLIT}/15 with GammaQuant INT4"

python scripts/run_loso.py \
  --config "${PATCHED_CFG}" \
  --seed 1 \
  --start_split ${SPLIT} \
  --end_split ${SPLIT} \
  --gamma_quant "quantize_and_gamma" \
  --quant_bits 4 \
  --gamma_type "s_shaped" \
  --apply_gamma "global"
