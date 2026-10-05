#!/bin/bash
# The knnfar1 adaptive protocol: 02_direction/PROTOCOL_2026-09-20_knnfar.md.
# --select-rule routes the defence and the evasion objective to the same score.
# Stage 1 is the baseline for both defences; stages 2 and 3 add a fresh initial draw,
# then twice the iterations at the original draw, to both.
# Do not edit the weights, steps, offsets or locations here without revising the
# protocol first.
# sbatch --exclude=xgpd0,xgpd5,xgpe1,xgpe2,xgpe6 code/scripts/slurm/knnfar_adaptive.sbatch

#SBATCH --job-name=fyp-knnfar
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

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

srun python code/scripts/check_finder.py
srun python code/scripts/check_selection.py

# Arrays, not strings: "60,60 100,20" is one argument and word-splitting an
# unquoted string expansion hands argparse a stray "100,20".
LOCS="60,60 100,20"
BASE=(python code/scripts/selection_attack.py --n-images 128 --locations "$LOCS")
KNN=("${BASE[@]}" --select-rule knnfar1)

GRID=0,0.0625,0.25,1,4,16   # knnfar1's stage-1 sweep, opened both sides of arm 5's peak
PROBE=0,0.25,1              # the weights both defences are probed at in stages 2 and 3

# Stage 1 - baseline. knnfar1 under an attack aimed at knnfar1, arm 5 under an attack
# aimed at arm 5, at matched images, locations, steps and draw.
srun "${KNN[@]}" --steps 100 --aux-weight "$GRID" \
    --out selection_attack_knnfar1_uniform_s1
srun "${KNN[@]}" --steps 100 --aux-weight "$GRID" --smooth-init \
    --out selection_attack_knnfar1_smooth_s1
srun "${BASE[@]}" --steps 100 --aux-weight "$PROBE" \
    --out selection_attack_arm5_uniform_s1_knncmp
srun "${BASE[@]}" --steps 100 --aux-weight "$PROBE" --smooth-init \
    --out selection_attack_arm5_smooth_s1_knncmp

# Stage 2 - a fresh initial draw, both defences.
srun "${KNN[@]}" --steps 100 --aux-weight "$PROBE" --seed-offset 1 \
    --out selection_attack_knnfar1_uniform_s2
srun "${KNN[@]}" --steps 100 --aux-weight "$PROBE" --seed-offset 1 --smooth-init \
    --out selection_attack_knnfar1_smooth_s2
srun "${BASE[@]}" --steps 100 --aux-weight "$PROBE" --seed-offset 1 \
    --out selection_attack_arm5_uniform_s2_knncmp
srun "${BASE[@]}" --steps 100 --aux-weight "$PROBE" --seed-offset 1 --smooth-init \
    --out selection_attack_arm5_smooth_s2_knncmp

# Stage 3 - twice the optimisation at the original draw, both defences.
srun "${KNN[@]}" --steps 200 --aux-weight "$PROBE" \
    --out selection_attack_knnfar1_uniform_s3
srun "${KNN[@]}" --steps 200 --aux-weight "$PROBE" --smooth-init \
    --out selection_attack_knnfar1_smooth_s3
srun "${BASE[@]}" --steps 200 --aux-weight "$PROBE" \
    --out selection_attack_arm5_uniform_s3_knncmp
srun "${BASE[@]}" --steps 200 --aux-weight "$PROBE" --smooth-init \
    --out selection_attack_arm5_smooth_s3_knncmp
