#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=gpuinfo2
#SBATCH --output=gpuinfo2_%j.out
#SBATCH --nodelist=gpu-node009,gpu-node010

set -euo pipefail
hostname
date
command -v nvidia-smi || true
nvidia-smi || true
