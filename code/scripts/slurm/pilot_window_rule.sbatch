#!/bin/bash
# Step 6d.3, on the cluster: the 20px location sweep on the full 121-location grid,
# one array task per reading of the window rule.
#
#   sbatch code/scripts/slurm/pilot_window_rule.sbatch
#
# WHAT THIS DECIDES. Gate A ran under `ceil+1` and did not reproduce RSA's Table 1, which
# reports 72.46% at 20px. One candidate cause is the window candidate set: an unpadded
# average pool admits only windows lying wholly inside the 14x14 token grid, so the
# number of candidates containing a token is 1 at a corner, 3 on an edge and 9 in the
# interior, and a corner patch can never be centred. Six readings of the paper's window
# rule are now implemented behind `--window-rule`; this runs all six against Table 1.
#
# THE DECISION RULE, FIXED BEFORE THESE RUN. Adopt the reading whose full-grid robust
# accuracy on 64 images is closest to Table 1's 72.46%. If no reading reaches 50%, the
# reproduction is negative under every reading. Clean accuracy under each reading is
# already measured against Table 2 (`code/results/table2_by_rule.json`); the deviation
# is reported whatever the Table 1 outcome. Do not change this rule after seeing output.
#
# WHY THE FULL GRID HERE. `--loc-stride` keeps every n-th location, which weakens the
# worst-case search and biases the reported number UP. A stride pilot below the target
# is decisive against a reading; a stride pilot at or above it is not. The laptop runs
# the stride-4 screen; this runs the full grid, which is the number that decides.
#
# COST. 121 locations x 2 batches of 32 x ~21 s = about 85 minutes per reading on a
# laptop GPU, and more on a T4. Six tasks run in parallel, so wall clock is one task.
# The walltime below is 6 hours because the cluster T4s are slower than the laptop card
# the 85-minute estimate came from, and a task that runs out of time has to replay its
# progress log rather than resume mid-location.
#
# RESUMING. Identical to gate_a.sbatch: progress is appended per location to
# `code/results/gate_a_<tag>.progress.jsonl` and replayed on restart, so an interrupted
# task redoes one location, not the run. A resume MUST reuse the same --n-images and
# --steps, since the recorded indices are into the evaluation batch order.
#
# OUTPUT. Six files, code/results/gate_a_pilot20full_<rule>.json.

#SBATCH --job-name=fyp-pilot-wr
# Partition is left to the environment: export SBATCH_PARTITION for your site.
#SBATCH --time=0-06:00:00
#SBATCH --array=0-5
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

# `padded` first so the decisive reading finishes even if the array is trimmed.
RULES="padded topk multi ceil token ceil+1"
RULE=$(echo $RULES | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

echo "host  : $(hostname)"
echo "job   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_JOB_PARTITION"
echo "rule  : ${RULE}"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
    echo "ERROR: nvidia-smi failed on $(hostname). The node advertises a GPU it does not have;"
    echo "       add it to the --exclude line above and resubmit this array task."
    exit 1
fi

# 64 images, not 512: this is a screen over six readings, not the reproduction. The
# adopted reading is then re-run at 512 over all five sizes by gate_a_rule.sbatch.
# `--renorm zero --score-frame per-head` are pinned to what this pilot ran under on
# 7 September, so a rerun reproduces the six-reading table rather than silently picking
# up the defaults that changed later that day. See the note in gate_a_rule.sbatch.
srun python code/scripts/gate_a.py \
    --sizes 20 \
    --n-images 64 \
    --window-rule "$RULE" \
    --renorm zero \
    --score-frame per-head \
    --tag "pilot20full_${RULE}"

echo "done: $(date)"
