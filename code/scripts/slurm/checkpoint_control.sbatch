#!/bin/bash
# Step 6h, stages 1 and 2: does RSA's own checkpoint change the value-space geometry?
#
#   sbatch code/scripts/slurm/checkpoint_control.sbatch
#
# WHY THIS EXISTS. Every explanation for the Gate A gap has been excluded by measurement
# except one: RSA evaluate `vitsmall_100.pt`, weights produced on ImageNet-100, while this
# harness uses timm's ImageNet-1k DeiT-small with the classifier head sliced to 100
# classes -- which is what RSA section 4.3 instructs, but is not necessarily what they ran.
# Their class draw also differs: `pycls/datasets/imagenet.py` sorts the wnid directories
# and takes `np.random.seed(0); np.sort(np.random.permutation(1000)[:100])`, which shares
# only 14 of 100 classes with this project's seed-1234 draw.
#
# WHY IT RUNS BEFORE THE SWEEP. The checkpoint can only matter through the geometry RSA
# ranks on: a_i = ||v_i - mu||_2 over value vectors, argmax over window means. This job
# measures that geometry directly on both models, in minutes, instead of inferring it from
# a 22 GPU-hour sweep. If their weights rank the patch the same way, the checkpoint cannot
# explain the gap and gate_a_rsa_ckpt.sbatch is not worth submitting.
#
# STAGE 1 is the clean-accuracy gate: RSA report 93.16% undefended. A load path that
# misses it by more than a few points is wrong and nothing after it means anything.
#
# STAGE 2 is the geometry, at 20px, at one location where the patch owns a token cell
# outright (0,0) and one where it straddles two (20,20), against a 100-step PatchAutoPGD
# patch: per layer, the median rank of the best patch token among 196, the median rank of
# the best patch-containing window, and the rate at which the selected window fully
# contains the patch.
#
# PREREQUISITES, both of which this script checks before spending GPU time:
#   - $FYP_RSA_CKPT points at vitsmall_100.pt (README of wagner-group/robust-self-attention
#     links the ImageNet-100 weights).
#   - $FYP_IMAGENET100 points at a build of RSA's seed-0 class draw, made with
#     `python code/scripts/build_imagenet100.py --classes rsa --out <dir> --zip <archive>`.
#     The seed-1234 build will NOT do: their checkpoint's head is ordered by their wnid
#     list, and scoring it against different classes is silently wrong rather than an error.
#
# OUTPUT. code/results/checkpoint_control_{ours,rsa}.json. Compare the window-rank rows.

#SBATCH --job-name=fyp-ckpt-ctrl
# Partition is left to the environment: export SBATCH_PARTITION for your site.
#SBATCH --time=01:00:00
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
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CKPT="${FYP_RSA_CKPT:-$HOME/fyp/code/data/vitsmall_100.pt}"
test -f "$CKPT" || {
    echo "ERROR: no checkpoint at $CKPT."
    echo "       Download the ImageNet-100 weights linked from the README of"
    echo "       github.com/wagner-group/robust-self-attention and set \$FYP_RSA_CKPT."
    exit 1; }

test -n "$FYP_IMAGENET100" -o -d code/data/imagenet100/val || {
    echo "ERROR: no ImageNet-100 build found."; exit 1; }

echo "host  : $(hostname)"
echo "job   : ${SLURM_JOB_ID} on $SLURM_JOB_PARTITION"
echo "ckpt  : $CKPT"
echo "data  : ${FYP_IMAGENET100:-code/data/imagenet100}"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
    echo "ERROR: nvidia-smi failed on $(hostname). The node advertises a GPU it does not have;"
    echo "       add it to the --exclude line above and resubmit."
    exit 1
fi

srun python code/scripts/check_reference_fidelity.py

echo "=== our model, for comparison ==="
srun python code/scripts/checkpoint_control.py --model ours

echo "=== RSA's checkpoint ==="
srun python code/scripts/checkpoint_control.py --model rsa --checkpoint "$CKPT"

echo "done: $(date)"
