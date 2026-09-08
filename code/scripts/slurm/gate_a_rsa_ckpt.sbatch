#!/bin/bash
# Step 6h, stages 3 and 4: Gate A under RSA's own checkpoint and class draw.
#
#   sbatch --array=1 code/scripts/slurm/gate_a_rsa_ckpt.sbatch   # STAGE 3: 20px only
#   sbatch          code/scripts/slurm/gate_a_rsa_ckpt.sbatch    # STAGE 4: all five sizes
#
# RUN STAGE 3 FIRST, AND ONLY RUN STAGE 4 IF STAGE 3 IS NEGATIVE. 20px is where Table 1
# claims 72.46% and this project measures 0.00%: the largest clean gap, and one task
# instead of five. If the defense holds on their weights, four of those five tasks are
# wasted and the answer arrives twenty hours later than it needed to.
#
# RUN checkpoint_control.sbatch BEFORE EITHER. It measures the same value-space geometry
# in minutes. If their checkpoint ranks the patch as this project's does, the checkpoint
# cannot explain the gap and neither stage below is worth its GPU-hours.
#
# THE DECISION RULE, FIXED BEFORE THE RUN (see PREREGISTRATION_6h_2026-09-08.md):
#   - 20px worst-case robust accuracy >= 50%  -> the reproduction SUCCEEDS. The negative
#     result of chapter 5 is withdrawn, the cause is this project's model adaptation, and
#     the finding becomes that RSA is sensitive to the checkpoint it defends.
#   - 20px worst-case robust accuracy < 50%   -> proceed to stage 4. Claim 1 then holds
#     under the authors' code, weights and class subset, and step 6h is closed.
#   No other threshold is to be adopted after seeing the number.
#
# WHAT CHANGES FROM gate_a_reference.sbatch. Two things and nothing else: the weights
# (`--checkpoint`, loaded whole with no head slicing, since the checkpoint is already
# 100-way) and the class subset (RSA's seed-0 draw, which must already be built on disk).
# The defense settings, the attack, the step count, the location grid, the image count and
# the robustness accounting are identical, so any difference in the result is attributable
# to the model and the classes.
#
# gate_a.py REFUSES TO RUN if the wnids on disk are not RSA's seed-0 draw. Their head is
# ordered by their wnid list; scoring it against different classes produces plausible
# numbers rather than an error, which is the failure mode worth a hard stop.
#
# COST. The reference sweep took 3.4-5.2 h per size on a TITAN V. Budget the same.
# Walltime is 12 h. Progress is appended per location to
# code/results/gate_a_rsackpt_<size>px.progress.jsonl and replayed on restart; a resume
# MUST reuse the same --n-images and --steps.
#
# OUTPUT. code/results/gate_a_rsackpt_{10,20,30,40,50}px.json, merged afterwards with
#   python code/scripts/merge_shards.py --tag rsackpt --job <id>

#SBATCH --job-name=fyp-gateA-rsackpt
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

CKPT="${FYP_RSA_CKPT:-$HOME/fyp/code/data/vitsmall_100.pt}"
test -f "$CKPT" || {
    echo "ERROR: no checkpoint at $CKPT. Set \$FYP_RSA_CKPT."; exit 1; }

test -n "$FYP_IMAGENET100" || {
    echo "ERROR: \$FYP_IMAGENET100 is unset. It must point at a build of RSA's seed-0"
    echo "       class draw, not the default seed-1234 build in code/data/imagenet100."
    exit 1; }

SIZES="10 20 30 40 50"
SIZE=$(echo $SIZES | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

echo "host  : $(hostname)"
echo "job   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_JOB_PARTITION"
echo "size  : ${SIZE}px"
echo "ckpt  : $CKPT"
echo "data  : $FYP_IMAGENET100"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
    echo "ERROR: nvidia-smi failed on $(hostname). The node advertises a GPU it does not have;"
    echo "       add it to the --exclude line above and resubmit this array task."
    exit 1
fi

srun python code/scripts/check_reference_fidelity.py

srun python code/scripts/gate_a.py \
    --sizes "$SIZE" \
    --checkpoint "$CKPT" \
    --window-rule "ceil+1" \
    --score-frame global \
    --renorm uniform \
    --tag "rsackpt_${SIZE}px"

echo "done: $(date)"
