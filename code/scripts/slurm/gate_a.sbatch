#!/bin/bash
# Gate A: the full RSA location sweep, split into one array task per patch size.
#
#   sbatch code/scripts/slurm/gate_a.sbatch
#
# Why an array rather than one job. code/README.md budgets the five sizes at 48.3 h
# total on a 6 GB laptop GPU (11.2 h at 10px and 20px, 9.2 h at 30px and 40px,
# 7.5 h at 50px). A partition that caps a job at 3 days would fit 48 h
# but leaves no margin; split per size and every task lands under 1 day, which also
# keeps the job inside the sub-1-day priority band most sites offer.
# Five tasks is well inside the 16 running / 256 submitted per-user limits.
#
# Why a separate --tag per size. gate_a.py appends every completed location to
# `code/results/gate_a_<tag>.progress.jsonl`. One shared tag would have five tasks
# appending to one file concurrently; a distinct tag per size gives each task its own
# progress file and its own results JSON, so nothing interleaves.
#
# RESUMING. A task killed at the time limit is safe to resubmit: the jsonl line for a
# location is written only after that location finishes, so an interrupted location is
# simply redone, and finished ones are skipped. Resubmit ONE size with
#
#   sbatch --array=2 code/scripts/slurm/gate_a.sbatch      # just 30px
#
# The recorded "wrong" values are indices into the evaluation batch order, so a resume
# MUST use the same --n-images and --steps. Do not edit those flags mid-sweep.
#
# OUTPUT. Five files, not one: code/results/gate_a_full_{10,20,30,40,50}px.json, each
# with a single entry under `per_size`. HANDOFF.md's `--tag full` would produce one
# combined file; nothing in the repo reads either name, so merge them by hand when all
# five are done.

#SBATCH --job-name=fyp-gateA
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
# /tmp has a 10MB per-user quota on some clusters (confirmed via `quota -s`) - redirect
# anything that stages through TMPDIR to $HOME instead.
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
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
    echo "ERROR: nvidia-smi failed on $(hostname). The node advertises a GPU it does not have;"
    echo "       add it to the --exclude line above and resubmit this array task."
    exit 1
fi

# No --n-images / --steps: the defaults are config.SCALE.n_eval_images = 512 and
# config.SCALE.attack_steps = 100, which is exactly RSA's protocol. No --renorm:
# the default is "zero", the mode measured to match RSA's Table 2.
# The defense settings are pinned to what this sweep originally ran under, not left to
# the defaults, which changed on 7 September when the authors' code became available.
# See the note in gate_a_rule.sbatch.
srun python code/scripts/gate_a.py --sizes "$SIZE" \
    --window-rule "ceil+1" --renorm zero --score-frame per-head \
    --tag "full_${SIZE}px"

echo "done: $(date)"
