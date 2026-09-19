#!/bin/bash
# The frozen saturation protocol: 02_direction/PROTOCOL_2026-09-19_saturation.md.
# Stage 1 closes two known gaps (arm 5 never had a smooth start, and its per-weight
# peak sits at 0.25, the smallest weight ever run against it). Stages 2 and 3 are the
# convergence probes: a fresh initial draw at 100 steps, then 200 steps at the original
# draw. Restarts come from separate invocations because patch_autopgd's own restart
# loop does not resample under init="given".
# Do not edit the weights or steps here without revising the protocol first.
# sbatch --exclude=xgpd0,xgpd5,xgpe1,xgpe2,xgpe6 code/scripts/slurm/saturation.sbatch

#SBATCH --job-name=fyp-saturation
#SBATCH --partition=gpu-long
#SBATCH --time=08:00:00
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

# Arrays, not strings: "60,60 100,20" is one argument and word-splitting an
# unquoted string expansion hands argparse a stray "100,20".
LOCS="60,60 100,20"
FINDER=(python code/scripts/finder_attack.py --finder tv --n-images 128 --locations "$LOCS")
ARM5=(python code/scripts/selection_attack.py --n-images 128 --locations "$LOCS")

# Stage 1 — baseline gaps.
srun "${ARM5[@]}" --steps 100 --smooth-init --aux-weight 0,0.25,0.5,1,2,4,8,16 \
    --out selection_attack_arm5_smooth_s1
srun "${ARM5[@]}" --steps 100 --aux-weight 0,0.0625,0.125 \
    --out selection_attack_arm5_low_s1

# Stage 2 — a fresh initial draw at the damaging weights.
srun "${FINDER[@]}" --steps 100 --weights 0,1,10 --seed-offset 1 \
    --out finder_attack_uniform_s2
srun "${FINDER[@]}" --steps 100 --weights 0,1,10 --seed-offset 1 --smooth-init \
    --out finder_attack_smooth_s2
srun "${ARM5[@]}" --steps 100 --aux-weight 0,0.0625,0.125,0.25,1 --seed-offset 1 \
    --out selection_attack_arm5_uniform_s2
srun "${ARM5[@]}" --steps 100 --aux-weight 0,0.0625,0.125,0.25,1 --seed-offset 1 --smooth-init \
    --out selection_attack_arm5_smooth_s2

# Stage 3 — twice the optimisation at the original draw.
srun "${FINDER[@]}" --steps 200 --weights 0,1,10 \
    --out finder_attack_uniform_s3
srun "${FINDER[@]}" --steps 200 --weights 0,1,10 --smooth-init \
    --out finder_attack_smooth_s3
srun "${ARM5[@]}" --steps 200 --aux-weight 0,0.0625,0.125,0.25,1 \
    --out selection_attack_arm5_uniform_s3
srun "${ARM5[@]}" --steps 200 --aux-weight 0,0.0625,0.125,0.25,1 --smooth-init \
    --out selection_attack_arm5_smooth_s3
