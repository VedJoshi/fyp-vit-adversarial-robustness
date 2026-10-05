#!/bin/bash
# Sources: 0=undefended, 1=mean, 2=centred, 3=self, 4=selection-aware.
# Target progress must exist in code/results. Run self first to verify replay.
# Task 4 attacks the target directly at the weight frozen on development data.
# Use transfer_attack.py --combine to union finished runs without another sweep.
# sbatch --array=0-2 --exclude=xgpe2,xgpe6 code/scripts/slurm/transfer_attack.sbatch
# Target another arm with FYP_TARGET_TAG, FYP_WINDOW_RULE and FYP_FREEZE_MASK_AFTER.

#SBATCH --job-name=fyp-transfer
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

TARGET="${FYP_TARGET_TAG:-gate_a_diag_centred_frz0}"
RULE="${FYP_WINDOW_RULE:-centred}"
FREEZE="${FYP_FREEZE_MASK_AFTER:-0}"
test -f "code/results/${TARGET}.progress.jsonl" || {
    echo "ERROR: missing code/results/${TARGET}.progress.jsonl, the target's own sweep";
    exit 1; }

case "$SLURM_ARRAY_TASK_ID" in
    0) SOURCE=undefended ;;
    1) SOURCE=mean ;;
    2) SOURCE=centred ;;
    3) SOURCE=self ;;
    4) SOURCE=selection ;;
    5) SOURCE=centred_frz0 ;;
    *) echo "ERROR: unexpected array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "transfer task $SLURM_ARRAY_TASK_ID: source=$SOURCE target=$TARGET rule=$RULE freeze=$FREEZE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun python code/scripts/check_transfer.py
srun python code/scripts/check_selection.py
srun python code/scripts/transfer_attack.py \
    --source "$SOURCE" \
    --tag "transfer_${SOURCE}${FYP_TAG_SUFFIX:-}" \
    --target-tag "$TARGET" \
    --size 20 \
    --n-images 512 \
    --steps 100 \
    --batch-size 32 \
    --loc-stride 1 \
    --loss dlr \
    --init uniform \
    --schedule counter \
    --window-rule "$RULE" \
    --freeze-mask-after "$FREEZE" \
    --score-frame global \
    --renorm uniform \
    --backward intended \
    --aux-weight 0.5
