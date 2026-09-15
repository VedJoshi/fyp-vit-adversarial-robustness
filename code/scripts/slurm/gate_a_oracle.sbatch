#!/bin/bash
# 20 px oracle comparison: 0=rsa, 1=oracle_window, 2=oracle_extent.
# Arms share initial patches through paired_seed.
# sbatch --exclude=xgpe2,xgpe6 code/scripts/slurm/gate_a_oracle.sbatch

#SBATCH --job-name=fyp-gateA-orc
#SBATCH --time=1-00:00:00
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
test -d "$FYP_IMAGENET100/val" || {
    echo "ERROR: missing $FYP_IMAGENET100/val"; exit 1; }

ARMS="rsa oracle_window oracle_extent"
ARM=$(echo "$ARMS" | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun python code/scripts/check_oracle.py
srun python code/scripts/gate_a.py \
    --sizes 20 \
    --arm "$ARM" \
    --n-images 512 \
    --steps 100 \
    --loc-stride 1 \
    --backward intended \
    --window-rule ceil+1 \
    --score-frame global \
    --renorm uniform \
    --tag "orcfull_${ARM}"
