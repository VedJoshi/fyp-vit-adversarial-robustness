#!/bin/bash
# E1 development run: stages 1-2 (100, 400 steps) x uniform/smooth start, 128 dev images.
# sbatch --exclude=xgpe2,xgpe6,xgpd2 code/scripts/slurm/camouflage_e1_dev.sbatch

#SBATCH --job-name=fyp-e1dev
#SBATCH --array=0-3
#SBATCH --time=12:00:00
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

STEPS=(100 100 400 400)
START=("" "--smooth-init" "" "--smooth-init")
NAME=(u s u s)
T=$SLURM_ARRAY_TASK_ID

# The sealed test set must name the same files on this copy of the dataset.
python code/scripts/draw_test_set.py --check

srun python code/scripts/camouflage_pilot.py --image-set dev --steps "${STEPS[$T]}" ${START[$T]} \
    --save-patches --out "camo_e1_dev_${STEPS[$T]}${NAME[$T]}"
