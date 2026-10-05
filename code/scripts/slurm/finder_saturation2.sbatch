#!/bin/bash
# The second finder saturation protocol: 02_direction/PROTOCOL_2026-09-20_finder_saturation.md.
# Extends saturation.sbatch with 200 steps at a fresh draw, three more initial draws,
# and 400 steps at the original and a fresh draw. Every stage runs against both
# defences; stage 7 is optional and runs last so an overrun drops it first.
# Do not edit the weights, steps, offsets or locations here without revising the
# protocol first.
# sbatch --exclude=xgpd0,xgpd5,xgpe1,xgpe2,xgpe6 code/scripts/slurm/finder_saturation2.sbatch

#SBATCH --job-name=fyp-finder-sat2
#SBATCH --partition=gpu-long
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

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

srun python code/scripts/check_finder.py
srun python code/scripts/check_selection.py

# Arrays, not strings: "60,60 100,20" is one argument and word-splitting an
# unquoted string expansion hands argparse a stray "100,20".
LOCS="60,60 100,20"
FINDER=(python code/scripts/finder_attack.py --finder tv --n-images 128 --locations "$LOCS")
ARM5=(python code/scripts/selection_attack.py --n-images 128 --locations "$LOCS")

FW=0,1,10                      # the finder's weight grid, as in saturation.sbatch
AW=0,0.0625,0.125,0.25,1       # arm 5's weight grid, as in saturation.sbatch

# Stage 4 - 200 steps at a fresh draw.
srun "${FINDER[@]}" --steps 200 --weights "$FW" --seed-offset 1 \
    --out finder_attack_uniform_s4
srun "${FINDER[@]}" --steps 200 --weights "$FW" --seed-offset 1 --smooth-init \
    --out finder_attack_smooth_s4
srun "${ARM5[@]}" --steps 200 --aux-weight "$AW" --seed-offset 1 \
    --out selection_attack_arm5_uniform_s4
srun "${ARM5[@]}" --steps 200 --aux-weight "$AW" --seed-offset 1 --smooth-init \
    --out selection_attack_arm5_smooth_s4

# Stage 5 - three further initial draws at 100 steps, which makes the draw axis a
# spread rather than a single extra start.
for OFF in 2 3 4; do
    srun "${FINDER[@]}" --steps 100 --weights "$FW" --seed-offset "$OFF" \
        --out "finder_attack_uniform_s5o$OFF"
    srun "${FINDER[@]}" --steps 100 --weights "$FW" --seed-offset "$OFF" --smooth-init \
        --out "finder_attack_smooth_s5o$OFF"
    srun "${ARM5[@]}" --steps 100 --aux-weight "$AW" --seed-offset "$OFF" \
        --out "selection_attack_arm5_uniform_s5o$OFF"
    srun "${ARM5[@]}" --steps 100 --aux-weight "$AW" --seed-offset "$OFF" --smooth-init \
        --out "selection_attack_arm5_smooth_s5o$OFF"
done

# Stage 6 - 400 steps at the original draw.
srun "${FINDER[@]}" --steps 400 --weights "$FW" \
    --out finder_attack_uniform_s6
srun "${FINDER[@]}" --steps 400 --weights "$FW" --smooth-init \
    --out finder_attack_smooth_s6
srun "${ARM5[@]}" --steps 400 --aux-weight "$AW" \
    --out selection_attack_arm5_uniform_s6
srun "${ARM5[@]}" --steps 400 --aux-weight "$AW" --smooth-init \
    --out selection_attack_arm5_smooth_s6

# Stage 7 - 400 steps at a fresh draw. Optional.
srun "${FINDER[@]}" --steps 400 --weights "$FW" --seed-offset 1 \
    --out finder_attack_uniform_s7
srun "${FINDER[@]}" --steps 400 --weights "$FW" --seed-offset 1 --smooth-init \
    --out finder_attack_smooth_s7
srun "${ARM5[@]}" --steps 400 --aux-weight "$AW" --seed-offset 1 \
    --out selection_attack_arm5_uniform_s7
srun "${ARM5[@]}" --steps 400 --aux-weight "$AW" --seed-offset 1 --smooth-init \
    --out selection_attack_arm5_smooth_s7
