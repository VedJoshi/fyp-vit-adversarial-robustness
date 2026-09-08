#!/bin/bash
# Step 6g: the undefended baseline, measured under Gate A's own protocol.
#
#   sbatch code/scripts/slurm/gate_a_undefended.sbatch          # all five sizes
#   sbatch --array=0 code/scripts/slurm/gate_a_undefended.sbatch  # 10px only, the cheap version
#
# WHY THIS EXISTS. The defended sweep's numbers are read against RSA's own reported
# undefended row (45.70 / 0 / 0 / 0 / 0, Table 1 `ViT-small`). That comparison crosses
# papers: a measurement from this harness against a claim from theirs. This run supplies
# the undefended column from this harness instead, so both sides of the comparison come
# from one protocol.
#
# WHAT `--no-defense` DOES. It forces `RSAConfig.window = 0` and nothing else. The patch
# size still drives the attack and the location grid, and RSAAttention stays installed on
# all twelve blocks, so `disable_fused_attention` and the whole forward path are the ones
# the defended sweeps ran through. Verified 8 September: with window 0 the model's logits
# are bit-identical to the raw classifier's, while window 2 moves them by 1.14.
#
# THE DEFENSE SETTINGS BELOW ARE INERT AND PINNED ANYWAY. With window 0 nothing reads
# them. They are passed so the result file states which sweep this is the baseline for,
# and so it does not silently change meaning if a default moves.
#
# WHAT TO EXPECT. At 20-50px the defended sweep already reached 0.00% and robust accuracy
# cannot go below zero, so those four tasks can only confirm 0.00 or come back positive.
# Positive would mean the defense makes the model *easier* to break at that size, which is
# a finding in its own right and worth the GPU time. The 10px task is the informative one:
# it is the only size where either row is off the floor.
#
# COST. Cheaper per location than the defended sweep, which spent time in `select()` on the
# argmax over window scores. The defended tasks took 3.4-5.2 h each on a TITAN V; budget the
# same and expect less. Walltime is 12 h.
#
# RESUMING. As in gate_a_reference.sbatch: progress is appended per location to
# `code/results/gate_a_undefended_<size>px.progress.jsonl` and replayed on restart. A resume
# MUST reuse the same --n-images and --steps.
#
# OUTPUT. code/results/gate_a_undefended_{10,20,30,40,50}px.json.

#SBATCH --job-name=fyp-gateA-undef
# Partition is left to the environment: export SBATCH_PARTITION for your site.
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# Job mail is left to the environment: export SBATCH_MAIL_USER and
# SBATCH_MAIL_TYPE=END,FAIL in your shell and Slurm applies them here.
# Some nodes advertise a GPU they do not have; a job lands, nvidia-smi exits
# non-zero, and `set -e` ends the script in under a second. The check below
# catches that. Exclude such nodes with SBATCH_EXCLUDE in your environment.

set -e

REPO="${FYP_REPO:-$HOME/fyp}"
VENV="${FYP_VENV:-$HOME/fyp-venv}"

. "$VENV/bin/activate"
cd "$REPO"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

test -d code/data/imagenet100/val || test -n "$FYP_IMAGENET100" || {
    echo "ERROR: code/data/imagenet100/val is missing and \$FYP_IMAGENET100 is unset."; exit 1; }

SIZES="10 20 30 40 50"
SIZE=$(echo $SIZES | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

echo "host  : $(hostname)"
echo "job   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_JOB_PARTITION"
echo "size  : ${SIZE}px"
echo "model : undefended (window forced to 0)"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
    echo "ERROR: nvidia-smi failed on $(hostname). The node advertises a GPU it does not have;"
    echo "       add it to the --exclude line above and resubmit this array task."
    exit 1
fi

srun python code/scripts/check_reference_fidelity.py

srun python code/scripts/gate_a.py \
    --sizes "$SIZE" \
    --no-defense \
    --window-rule "ceil+1" \
    --score-frame global \
    --renorm uniform \
    --tag "undefended_${SIZE}px"

echo "done: $(date)"
