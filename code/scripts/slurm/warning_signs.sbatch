#!/bin/bash
# Patch warning signs with the defense fixed across attack sizes.
# Defaults to centred, frozen at 0. FYP_FREEZE= selects the unfrozen arm.
# Transfer checks run separately through transfer_attack.sbatch.
# sbatch --exclude=xgpe2,xgpe6 code/scripts/slurm/warning_signs.sbatch

#SBATCH --job-name=fyp-warnsigns
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
test -d "$FYP_IMAGENET100/val" || {
    echo "ERROR: missing $FYP_IMAGENET100/val"; exit 1; }

RULE="${FYP_RULE:-centred}"
PATCH_PX="${FYP_PATCH_PX:-20}"
STEPS="${FYP_STEPS:-100}"
OUT="${FYP_OUT:-gate_a_centred_frz0_warning_signs}"
# Unset defaults to block 0; explicitly empty omits freezing.
FREEZE_ARG=""
if [ -n "${FYP_FREEZE-0}" ]; then
    FREEZE_ARG="--freeze-mask-after ${FYP_FREEZE-0}"
fi

echo "warning signs: rule=$RULE patch=${PATCH_PX}px steps=$STEPS freeze='${FREEZE_ARG}' out=$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun python code/scripts/check_centred_rule.py
srun python code/scripts/warning_signs.py \
    --threat patch \
    --patch-px "$PATCH_PX" \
    --steps "$STEPS" \
    --defense rsa \
    --window-rule "$RULE" \
    --score-frame global \
    --renorm uniform \
    --backward intended \
    $FREEZE_ARG \
    --out "$OUT"
