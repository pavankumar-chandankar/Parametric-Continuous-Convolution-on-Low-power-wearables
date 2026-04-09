#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=15
#SBATCH --ntasks=15
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=rwhar_ksg4
#SBATCH --output=/home/g263412/GAMMA/slurm_logs/rwhar_ksg4_%A_%a.out
#SBATCH --error=/home/g263412/GAMMA/slurm_logs/rwhar_ksg4_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pavankumar.chandankar@student.uni-siegen.de

PS1="${PS1:-}"
set -eo pipefail

cd /home/g263412/GAMMA
mkdir -p slurm_logs

source /etc/profile.d/modules.sh
module load miniconda3/4.9.2
CONDA_BIN=/cm/shared/omni/apps/miniconda3/bin/conda
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate gamma

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

KERNELS=(1 3 5 7 9 11)
KIDX=${SLURM_ARRAY_TASK_ID}

if [ "${KIDX}" -lt 0 ] || [ "${KIDX}" -ge "${#KERNELS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${KIDX}"
  exit 1
fi

KS=${KERNELS[$KIDX]}

BASE_CFG="configs/continuous_conv/realworld_waist_accel_loso_Multirate.yaml"
LOG_SUBDIR="experiments/kernel_sweep_standard_gamma4/realworld_waist_accel/ks$(printf "%02d" "${KS}")"

echo "Running RealWorld standard gamma4 kernel sweep"
echo "Kernel size = ${KS}"
echo "Log subdir  = ${LOG_SUBDIR}"
echo "Nodes/tasks = ${SLURM_NNODES}/${SLURM_NTASKS}"

export BASE_CFG LOG_SUBDIR KS

srun --ntasks=15 --ntasks-per-node=1 bash -lc '
set -eo pipefail

cd /home/g263412/GAMMA

source /etc/profile.d/modules.sh
module load miniconda3/4.9.2
CONDA_BIN=/cm/shared/omni/apps/miniconda3/bin/conda
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate gamma

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

SPLIT=${SLURM_PROCID}
PATCHED_CFG="/tmp/realworld_std_gamma4_ks${KS}_split${SPLIT}.yaml"

python - <<PY
import os
import yaml

base_cfg = os.environ["BASE_CFG"]
patched_cfg = os.environ["PATCHED_CFG"] if "PATCHED_CFG" in os.environ else "'"$PATCHED_CFG"'"
log_subdir = os.environ["LOG_SUBDIR"]
ks = int(os.environ["KS"])

with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("model", {})
cfg["model"]["conv_type"] = "standard"
cfg["model"]["conv_kernel_size"] = int(ks)
cfg["model"]["standard_padding"] = cfg["model"].get("standard_padding", "same")

cfg.setdefault("train_cfg", {})
cfg["train_cfg"]["log_subdir"] = log_subdir
cfg["train_cfg"]["auto_resume"] = True
cfg["train_cfg"]["multirate_val_lengths"] = [25, 12, 6]

cfg["train_cfg"].pop("resample_train_len", None)
cfg["train_cfg"].pop("resample_val_len", None)
cfg["train_cfg"].pop("freq_tag_override", None)

with open(patched_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote:", patched_cfg)
print("Kernel size:", ks)
print("Split:", '"$SPLIT"')
print("Log subdir:", log_subdir)
PY

echo "Node ${SLURMD_NODENAME} running split ${SPLIT} with ks=${KS}"

python scripts/run_loso.py \
  --config "${PATCHED_CFG}" \
  --seed 1 \
  --start_split ${SPLIT} \
  --end_split ${SPLIT} \
  --gamma_quant "quantize_and_gamma" \
  --quant_bits 4 \
  --gamma_type "s_shaped" \
  --apply_gamma "global"
'
