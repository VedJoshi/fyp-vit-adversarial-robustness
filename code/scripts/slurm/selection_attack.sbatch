#!/bin/bash
# 0=auxiliary-weight sweep on development images, 1=nested-support control.
# Freeze the chosen weight before full evaluation through transfer_attack.sbatch.
# sbatch --array=0-1 --exclude=xgpe2,xgpe6 code/scripts/slurm/selection_attack.sbatch

#SBATCH --job-name=fyp-selection
#SBATCH --time=12:00:00
#SBATCH --array=0-1
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
srun python code/scripts/check_selection.py

case "$SLURM_ARRAY_TASK_ID" in
  0)
    srun python code/scripts/selection_attack.py \
        --n-images 128 --size 20 --steps 100 --batch-size 32 \
        --aux-weight 0,0.5,2,8 \
        --locations "60,60 0,0 100,20 140,160" \
        --loss dlr --init uniform --schedule counter \
        --window-rule centred --freeze-mask-after 0 \
        --score-frame global --renorm uniform --backward intended \
        --out selection_attack_dev
    ;;
  1)
    srun python code/scripts/containment_check.py \
        --n-images 256 --inner 10 --outer 20 --steps 100 --batch-size 32 \
        --window-rule centred --freeze-mask-after 0 \
        --score-frame global --renorm uniform --backward intended \
        --out containment_check
    ;;
  *) echo "ERROR: unexpected array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
