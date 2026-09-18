#!/bin/bash
# Classification replay of the saved capture through base, multi4 and single6.
# No attack optimisation; multi4 and single6 are measured under transfer.
# sbatch --exclude=xgpd0,xgpe1,xgpe2,xgpe6 code/scripts/slurm/selection_replay.sbatch

#SBATCH --job-name=fyp-selreplay
#SBATCH --partition=gpu-long
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
test -f "$REPO/code/results/selection_capture_dev.pt" || { echo "ERROR: missing capture"; exit 1; }

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun python code/scripts/check_selection.py
srun python code/scripts/check_centred_rule.py
srun python code/scripts/selection_screen.py --out selection_screen
srun python code/scripts/selection_replay.py --capture selection_capture_dev --out selection_replay_dev
