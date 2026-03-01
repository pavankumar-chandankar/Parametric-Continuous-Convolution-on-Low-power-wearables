#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=gpuinfo
#SBATCH --output=gpuinfo_%j.out

set -euo pipefail

echo "=== node ==="
hostname
date

echo "=== slurm env ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"

echo "=== nvidia-smi ==="
command -v nvidia-smi || true
nvidia-smi || true

echo "=== job detail (first 120 lines) ==="
scontrol show job "$SLURM_JOB_ID" | sed -n '1,120p'
