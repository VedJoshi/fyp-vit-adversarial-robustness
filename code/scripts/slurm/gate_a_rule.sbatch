#!/bin/bash
# Step 6d.4: the full five-size Gate A sweep, re-run under the adopted window rule.
#
#   sbatch --export=ALL,FYP_WINDOW_RULE=padded code/scripts/slurm/gate_a_rule.sbatch
#
# This is gate_a.sbatch with one addition: the window rule is a parameter rather than
# the `ceil+1` default. Run it only after the 20px pilot has picked a reading under the
# decision rule fixed in pilot_window_rule.sbatch. Everything else is RSA's protocol
# unchanged: 512 images, 100-step PatchAutoPGD(DLR), the complete location grid,
# `renormalise="zero"`, an image robust only if it survives every location.
#
# THE RULE MUST BE PASSED IN. There is no default here on purpose. Re-running the sweep
# under a silently-chosen rule is how a reproduction becomes unreportable, and the tag
# carries the rule so a results file can never be mistaken for another reading's.
#
# COST. The `ceil+1` sweep took 53.9 GPU-hours over five array tasks and finished in
# under a day of wall clock. Expect the same order; `padded` and `multi` do slightly
# more pooling work per layer, which is negligible next to the attack.
#
# RESUMING. As gate_a.sbatch. Progress is per location in
# `code/results/gate_a_rule_<rule>_<size>px.progress.jsonl`. Resubmit one size with
#
#   sbatch --array=2 --export=ALL,FYP_WINDOW_RULE=padded code/scripts/slurm/gate_a_rule.sbatch
#
# OUTPUT. Five files, code/results/gate_a_rule_<rule>_{10,20,30,40,50}px.json. Merge
# them the way `code/results/gate_a_full.json` was merged from the first sweep.

#SBATCH --job-name=fyp-gateA-rule
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

: "${FYP_WINDOW_RULE:?set it, e.g. --export=ALL,FYP_WINDOW_RULE=padded}"

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
echo "rule  : ${FYP_WINDOW_RULE}"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
    echo "ERROR: nvidia-smi failed on $(hostname). The node advertises a GPU it does not have;"
    echo "       add it to the --exclude line above and resubmit this array task."
    exit 1
fi

# `--renorm zero --score-frame per-head` are pinned rather than left to the defaults.
# Both defaults changed on 7 September, when the authors' released code showed the
# masked key's weight is 1/197 and the anomaly score is centred on the head-averaged
# mean. Job 826360 started under the old defaults; a task resubmitted after that change
# would otherwise resume with different arithmetic and append to the same progress log,
# mixing two defenses in one result file. New work belongs in
# pilot_reference_fidelity.sbatch, not here.
srun python code/scripts/gate_a.py \
    --sizes "$SIZE" \
    --window-rule "$FYP_WINDOW_RULE" \
    --renorm zero \
    --score-frame per-head \
    --tag "rule_${FYP_WINDOW_RULE}_${SIZE}px"

echo "done: $(date)"
