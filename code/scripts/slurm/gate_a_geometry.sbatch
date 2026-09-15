#!/bin/bash
# Record per-layer selection geometry on the 20 px evaluation grid.
# sbatch --exclude=xgpe2,xgpe6 code/scripts/slurm/gate_a_geometry.sbatch

#SBATCH --job-name=fyp-gateA-geom
#SBATCH --time=1-00:00:00
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
test -d "$FYP_IMAGENET100/val" || {
    echo "ERROR: missing $FYP_IMAGENET100/val"; exit 1; }

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun python code/scripts/check_geometry.py
srun python code/scripts/gate_a.py \
    --sizes 20 \
    --arm rsa \
    --record-geometry \
    --n-images 512 \
    --steps 100 \
    --loc-stride 1 \
    --backward intended \
    --window-rule ceil+1 \
    --score-frame global \
    --renorm uniform \
    --tag geom20
