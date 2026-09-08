#!/bin/bash
# Short GPU job: confirms the environment, the dataset and the GPU all work
# before anything long is queued. Run smoke_min.sbatch first.
#
#   sbatch code/scripts/slurm/smoke_test.sbatch
#
# `gpu` is the short partition: 3 h maximum, 15 min default. Asking for 15 min or
# less earns the +200 "nice" bonus documented on the slurm-info page, so this job
# gets the highest partition priority available (800 + 200 = 1000).
#
# `--no-save` is REQUIRED here. Without it validate_patch_fool.py overwrites
# code/results/m4_patch_fool_validation.json, which is a real, git-tracked result
# (is_result: true, written 29 Aug under commit a50d24b on the laptop GPU).

#SBATCH --job-name=fyp-smoke
# Partition is left to the environment: export SBATCH_PARTITION for your site.
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
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

echo "host    : $(hostname)"
echo "job     : $SLURM_JOB_ID on $SLURM_JOB_PARTITION"
nvidia-smi

# config.SCALE is FULL, so this reads ImageNet-100, not Imagenette.
test -d code/data/imagenet100/val || test -n "$FYP_IMAGENET100" || {
    echo "ERROR: code/data/imagenet100/val is missing and \$FYP_IMAGENET100 is unset."; exit 1; }

# 32 images, one batch: PatchAutoPGD at 100 steps then Patch-Fool twice at 250.
# ~7 min on a 6 GB laptop card (code/README.md), so 15 min is comfortable here.
srun python code/scripts/validate_patch_fool.py --n-images 32 --steps 250 --no-save

echo "done: $(date)"
