#!/bin/bash
# Gate A derivative-only arm. This differs from gate_a_reference.sbatch only by
# `--backward released`. Submit all sizes with:
#   sbatch --exclude=xgpe2,xgpe6 code/scripts/slurm/gate_a_backward.sbatch

#SBATCH --job-name=fyp-gateA-bwd
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
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
test -d "$FYP_IMAGENET100/val" || {
    echo "ERROR: missing $FYP_IMAGENET100/val"; exit 1; }

SIZES="10 20 30 40 50"
SIZE=$(echo "$SIZES" | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun python code/scripts/check_reference_fidelity.py
srun python code/scripts/gate_a.py \
    --sizes "$SIZE" \
    --window-rule ceil+1 \
    --score-frame global \
    --renorm uniform \
    --backward released \
    --tag "bwdrel_${SIZE}px"
