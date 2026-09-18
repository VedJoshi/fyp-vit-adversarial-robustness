#!/bin/bash
# Coverage, cover rank and reference-shift for arm 5 under weights 0 and 0.5.
# --capture also writes patch pixels, all 196 scores under both references and
# the reference vectors, under a new name; selection_diagnostic_dev is preserved.
# Development images only. Chooses between top-k union and stable-reference scoring.
# sbatch --exclude=xgpe2,xgpe6 code/scripts/slurm/selection_diagnostic.sbatch

#SBATCH --job-name=fyp-selcap
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

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

srun python code/scripts/selection_diagnostic.py \
    --n-images 128 --size 20 --steps 100 --batch-size 32 \
    --aux-weight 0.5 \
    --locations "60,60 0,0 100,20 140,160" \
    --loss dlr --init uniform --schedule counter \
    --score-frame global --renorm uniform --backward intended \
    --capture \
    --out selection_capture_dev
