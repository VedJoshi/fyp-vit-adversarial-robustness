#!/bin/bash
# Step 6f: the 20px location sweep under the authors' released implementation, as a
# 2x2 over the two places this harness differed from it.
#
#   sbatch code/scripts/slurm/pilot_reference_fidelity.sbatch
#
# WHY THIS EXISTS. Norman Mu pointed to wagner-group/robust-self-attention on
# 7 September. Reading `pycls/models/vision_transformer.py:81-107` against `fyp/rsa.py`
# settles the window rule and exposes two implementation differences:
#
#   1. THE WINDOW RULE IS `ceil+1`. Their line 84 computes `p // 16 + 1`, plus one more
#      when `p % 16 > 1`, which equals ceil(p/16) + 1 at every size except p = 1 mod 16.
#      The paper uses none of those. Line 94 pools with stride 1 and no padding, so the
#      candidate set is the unpadded one. Five of the six readings are therefore not
#      what the authors ran, `topk` included, and `ceil+1` is the reading that measured
#      0.00% in the 7 September pilot.
#
#   2. THE ANOMALY SCORE IS CENTRED DIFFERENTLY. Their lines 89-93 average the per-head
#      mean value vector over heads and subtract that single vector from every head.
#      This harness subtracted each head's own mean. Different statistic, different
#      token ranking, different argmax. Now `--score-frame {global,per-head}`.
#
#   3. THE MASKED KEY IS DILUTED, NOT DELETED. Their line 107 sets the masked key's
#      attention column to 1/197 and does not renormalise the rows. This harness ran
#      `--renorm zero`, which deletes it. Now the default is `--renorm uniform`.
#
# WHAT THIS DECIDES, FIXED BEFORE THESE RUN. Task 0 is the authors' implementation.
# If its full-grid robust accuracy on 64 images reaches 50%, the negative Gate A result
# was an artefact of this harness, Gate A reopens, and the five-size sweep is re-run
# under the reference settings before anything is claimed. If it stays below 50%, the
# reproduction is negative under the authors' own code and not merely under a reading
# of their paper. Do not change this rule after seeing output.
#
# WHY A 2x2 AND NOT ONE TASK. Four tasks run in the wall clock of one, and the grid
# says which difference moved the number rather than only that something did.
#
#   task 0  global   / uniform   the released implementation
#   task 1  global   / zero      the score fix alone
#   task 2  per-head / uniform   the renormalisation fix alone
#   task 3  per-head / zero      the control
#
# Task 3 is the `ceil+1` cell of the 7 September pilot and must come back at 0.00%.
# It is here as a check on the harness: if the control moves, something changed that
# was not one of these two knobs, and the other three tasks are uninterpretable.
#
# COST. As pilot_window_rule.sbatch: 121 locations x 2 batches of 32, about 85 minutes
# on a laptop GPU and more on a T4. Walltime is 6 hours for the same reason.
#
# RESUMING. Progress is appended per location to
# `code/results/gate_a_<tag>.progress.jsonl` and replayed on restart, so an interrupted
# task redoes one location, not the run. A resume MUST reuse the same --n-images and
# --steps, since the recorded indices are into the evaluation batch order.
#
# PREFLIGHT. Each task runs code/scripts/check_reference_fidelity.py before the sweep
# and aborts if it fails, so no task can burn GPU time on arithmetic that does not match
# the authors' code. That check cannot run on the login node: importing torch there hits
# the virtual-memory cap.
#
# OUTPUT. Four files, code/results/gate_a_ref20_<frame>_<renorm>.json.

#SBATCH --job-name=fyp-pilot-ref
# Partition is left to the environment: export SBATCH_PARTITION for your site.
#SBATCH --time=0-06:00:00
#SBATCH --array=0-3
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

# Task 0 first so the decisive cell finishes even if the array is trimmed.
FRAMES="global   global per-head per-head"
RENORMS="uniform zero   uniform  zero"
FRAME=$(echo $FRAMES | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))
RENORM=$(echo $RENORMS | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

echo "host  : $(hostname)"
echo "job   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_JOB_PARTITION"
echo "frame : ${FRAME}"
echo "renorm: ${RENORM}"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
    echo "ERROR: nvidia-smi failed on $(hostname). The node advertises a GPU it does not have;"
    echo "       add it to the --exclude line above and resubmit this array task."
    exit 1
fi

# The differential test against the authors' released code, as a preflight. It compares
# this harness's window side, anomaly score, masked token set, replacement value and
# masked attention weight against `pycls/models/vision_transformer.py:81-107`, needs no
# GPU and no dataset, and takes about two seconds. `set -e` aborts the task if it fails,
# so a sync that landed badly costs one second rather than 85 minutes of GPU time.
#
# It runs here rather than on the login node because importing torch there hits the
# virtual-memory cap: `ImportError: libtorch_cuda.so: failed to map segment from shared
# object`. Nothing is wrong with the environment; the login node just cannot map the
# CUDA libraries.
srun python code/scripts/check_reference_fidelity.py

# 64 images and 20px only, matching pilot_window_rule.sbatch so the control cell is
# comparable to that pilot's `ceil+1` row line for line. The window rule is passed
# explicitly rather than left to the default, so the output file records the reading
# even if the default changes again.
srun python code/scripts/gate_a.py \
    --sizes 20 \
    --n-images 64 \
    --window-rule "ceil+1" \
    --score-frame "$FRAME" \
    --renorm "$RENORM" \
    --tag "ref20_${FRAME}_${RENORM}"

echo "done: $(date)"
