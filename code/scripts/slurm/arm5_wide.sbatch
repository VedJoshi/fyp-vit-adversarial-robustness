#!/bin/bash
# Arm 5 with eight evasion weights against the finder's five, giving arm 5 the larger
# weight sweep so a prior lopsided comparison (fewer arm-5 runs per location than the
# finder's) cannot explain the finder's lead.
# sbatch --exclude=xgpd0,xgpd5,xgpe1,xgpe2,xgpe6 code/scripts/slurm/arm5_wide.sbatch

#SBATCH --job-name=fyp-arm5wide
#SBATCH --partition=gpu-long
#SBATCH --time=03:00:00
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
nvidia-smi --query-gpu=name,driver_version --format=csv

srun python code/scripts/selection_attack.py --n-images 128 \
    --locations "60,60 100,20" --steps 100 \
    --aux-weight 0,0.25,0.5,1,2,4,8,16 \
    --out selection_attack_arm5_wide
