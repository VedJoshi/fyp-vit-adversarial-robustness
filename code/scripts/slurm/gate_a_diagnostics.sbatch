#!/bin/bash
# 20 px masking diagnostics: 0=max, 1=topk, 2=token, 3=mean frozen,
# 4=centred, 5=centred frozen at 0. Arm 3 reads FYP_FREEZE_LAYER (default 0).
# sbatch --array=4-5 --exclude=xgpe2,xgpe6 code/scripts/slurm/gate_a_diagnostics.sbatch

#SBATCH --job-name=fyp-gateA-diag
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

FREEZE_LAYER="${FYP_FREEZE_LAYER:-0}"

case "$SLURM_ARRAY_TASK_ID" in
    0) RULE=max;    TAG=diag_max_geom;    EXTRA="--record-geometry" ;;
    1) RULE=topk;   TAG=diag_topk_geom;   EXTRA="--record-geometry" ;;
    2) RULE=token;  TAG=diag_token;       EXTRA="" ;;
    3) RULE=ceil+1; TAG="diag_freeze${FREEZE_LAYER}"
       EXTRA="--freeze-mask-after ${FREEZE_LAYER} --record-geometry" ;;
    4) RULE=centred; TAG=diag_centred;    EXTRA="--record-geometry" ;;
    5) RULE=centred; TAG=diag_centred_frz0
       EXTRA="--freeze-mask-after 0 --record-geometry" ;;
    *) echo "ERROR: unexpected array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "arm $SLURM_ARRAY_TASK_ID: rule=$RULE tag=$TAG extra='$EXTRA'"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun python code/scripts/check_pool_diagnostics.py
srun python code/scripts/check_max_rule.py
srun python code/scripts/check_centred_rule.py
srun python code/scripts/gate_a.py \
    --sizes 20 \
    --arm rsa \
    --n-images 512 \
    --steps 100 \
    --loss dlr \
    --init uniform \
    --schedule counter \
    --batch-size 32 \
    --loc-stride 1 \
    --backward intended \
    --window-rule "$RULE" \
    --score-frame global \
    --renorm uniform \
    --tag "$TAG" \
    $EXTRA
