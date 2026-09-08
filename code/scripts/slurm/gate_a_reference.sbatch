#!/bin/bash
# Step 6f, stage 2: the five-size Gate A sweep under the authors' released implementation.
#
#   sbatch code/scripts/slurm/gate_a_reference.sbatch
#
# WHEN TO RUN THIS. Only after pilot_reference_fidelity.sbatch has read out and its task 3
# control returned 0.00%. This is the number that gets written up, so it must not run under
# settings that the 2x2 has not first validated.
#
# WHAT IT IS. gate_a.sbatch with the defense pinned to what wagner-group/robust-self-attention
# actually does, established 7 September from `pycls/models/vision_transformer.py:81-107`:
#
#   --window-rule ceil+1      their line 84, `p // 16 + 1 (+1 if p % 16 > 1)`, and line 94
#                             pools unpadded so only whole-grid windows are candidates
#   --score-frame global      their lines 89-93 centre every head on the head-averaged mean
#   --renorm uniform          their line 107 sets the masked key's column to 1/197 and does
#                             not renormalise the rows
#
# Everything else is RSA's protocol unchanged: 512 images, 100-step PatchAutoPGD(DLR), the
# complete location grid, an image robust only if it survives every location.
#
# THE SETTINGS ARE PINNED, NOT LEFT TO THE DEFAULTS. The defaults are currently these three
# values, but a result file that depends on a default is a result file that changes meaning
# when the default does. That already happened once: gate_a.sbatch and gate_a_rule.sbatch ran
# under `zero` and per-head centring as defaults, and had to be pinned retroactively on
# 7 September so their outputs stayed interpretable.
#
# PREFLIGHT. Each task runs check_reference_fidelity.py first and aborts on failure, so no
# task can spend GPU time on arithmetic that does not match the authors' code.
#
# COST. The `ceil+1` sweep of 3-4 Sep took 53.9 GPU-hours over five array tasks. The `topk`
# sweep of 7 Sep took between 4.9 and 7.3 hours per task. Expect the same order. Walltime is
# 24 hours; the 10px and 20px tasks are the long ones, because the defense holds at those
# sizes and the attack therefore tries more of the location grid before giving up.
#
# RESUMING. Progress is appended per location to
# `code/results/gate_a_reference_<size>px.progress.jsonl` and replayed on restart, so an
# interrupted task redoes one location rather than the run. A resume MUST reuse the same
# --n-images and --steps, since the recorded indices are into the evaluation batch order.
#
# OUTPUT. Five files, code/results/gate_a_reference_{10,20,30,40,50}px.json, merged with
#   python code/scripts/merge_shards.py --tag reference --job <id>

#SBATCH --job-name=fyp-gateA-ref
# Partition is left to the environment: export SBATCH_PARTITION for your site.
#SBATCH --time=1-00:00:00
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
echo "rule  : ceil+1 / global / uniform  (the released implementation)"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
    echo "ERROR: nvidia-smi failed on $(hostname). The node advertises a GPU it does not have;"
    echo "       add it to the --exclude line above and resubmit this array task."
    exit 1
fi

srun python code/scripts/check_reference_fidelity.py

srun python code/scripts/gate_a.py \
    --sizes "$SIZE" \
    --window-rule "ceil+1" \
    --score-frame global \
    --renorm uniform \
    --tag "reference_${SIZE}px"

echo "done: $(date)"
