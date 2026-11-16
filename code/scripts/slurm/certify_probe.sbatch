#!/bin/bash
# 0=invariance test and coarse covering sets, 1=3-token windows (attn), 2=3-token windows (pixel).
# sbatch --exclude=xgpe2,xgpe6 code/scripts/slurm/certify_probe.sbatch

#SBATCH --job-name=fyp-certify
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -e

REPO="${FYP_REPO:-$HOME/fyp}"
VENV="${FYP_VENV:-$HOME/fyp-venv}"

. "$VENV/bin/activate"
cd "$REPO"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export FYP_IMAGENET100="$REPO/code/data/imagenet100"

mkdir -p "$TMPDIR"
test -d "$FYP_IMAGENET100/val" || { echo "ERROR: missing $FYP_IMAGENET100/val"; exit 1; }

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

case "$SLURM_ARRAY_TASK_ID" in
  0) srun python code/scripts/certify_probe.py --out probe_certify ;;
  1) srun python code/scripts/certify_probe.py --sets 3,1 --modes attn \
         --invariance-images 0 --out probe_certify_w3_attn ;;
  2) srun python code/scripts/certify_probe.py --sets 3,1 --modes pixel \
         --invariance-images 0 --out probe_certify_w3_pixel ;;
  *) echo "ERROR: unexpected array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
