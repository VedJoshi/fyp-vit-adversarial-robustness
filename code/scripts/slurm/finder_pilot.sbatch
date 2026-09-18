#!/bin/bash
# Adaptive pilot against the input-space finder, and arm 5's own adaptive attack on
# matched images, locations and steps, so the comparison is adaptive against adaptive
# on one card. Comparing the finder's adaptive number against arm 5's transfer number
# is the error this branch exists to avoid.
# sbatch --exclude=xgpd0,xgpd5,xgpe1,xgpe2,xgpe6 code/scripts/slurm/finder_pilot.sbatch

#SBATCH --job-name=fyp-finderpilot
#SBATCH --partition=gpu-long
#SBATCH --time=06:00:00
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

srun python code/scripts/check_finder.py
srun python code/scripts/check_selection.py
srun python code/scripts/check_centred_rule.py

LOCS="60,60 100,20"

# The finder under an attack aimed at its own selector.
srun python code/scripts/finder_attack.py --finder tv --n-images 128 \
    --locations "$LOCS" --steps 100 --out finder_attack_dev

# The same finder, started from a blurred patch, in case the default init
# biases the search towards the high-frequency region tv separates best.
srun python code/scripts/finder_attack.py --finder tv --n-images 128 \
    --locations "$LOCS" --steps 100 --smooth-init --out finder_attack_smooth_dev

# Arm 5 under its own block-0 evasion attack, matched on everything else.
# A separate --out: selection_attack_dev.json is the canonical 14 September record.
srun python code/scripts/selection_attack.py --n-images 128 \
    --locations "$LOCS" --steps 100 --aux-weight 0,0.5,2,8 \
    --out selection_attack_arm5_matched
