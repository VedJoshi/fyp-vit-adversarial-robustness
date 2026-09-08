#!/bin/bash
# Step 6h, half one: Gate A on RSA's own 100 classes, this project's model.
#
#   sbatch --array=1 code/scripts/slurm/gate_a_rsa_classes.sbatch   # 20px only, run this first
#   sbatch          code/scripts/slurm/gate_a_rsa_classes.sbatch    # all five sizes
#
# WHY THIS EXISTS. Step 6h is two variables at once: the model and the class subset.
#
#   model              classes            status
#   ours (sliced)      ours (seed 1234)   measured, gate_a_reference.sbatch
#   ours (sliced)      theirs (seed 0)    THIS JOB
#   theirs (vitsmall)  theirs (seed 0)    blocked: their weights are not downloadable
#
# Their published weights are not obtainable -- the link in their README points at an
# empty folder as of 8 September 2026 -- so the model half cannot be run until the authors
# send the checkpoint. The class half can, and it is worth running on its own: if robust
# accuracy is still 0.00% at 20px on their exact class subset, the class draw is
# eliminated and step 6h reduces to the checkpoint alone.
#
# WHAT CHANGES FROM gate_a_reference.sbatch. One thing: $FYP_IMAGENET100 points at the
# seed-0 build instead of the seed-1234 one. Same weights, same head slicing, same defense
# settings, same attack, same grid, same image count. gate_a.py reads the class draw off
# the wnids on disk and records it, so the result file names which subset produced it.
#
# BUILD THE DATA FIRST, on the laptop, then upload:
#   python code/scripts/build_imagenet100.py --classes rsa \
#       --src <imagenet-val>/imagenet-val --out code/data/imagenet100_rsa
# 5000 images, 100 classes, 50 each. The two builds share 14 of 100 classes, so they must
# live in separate directories.
#
# WHAT TO EXPECT. Clean accuracy will move: a different 100 classes is a different task,
# and RSA's 93.16% was measured on theirs. Clean accuracy landing closer to 93.16% than
# this project's own draw did is itself evidence about the class subset. Robust accuracy
# at 20px is the number that matters.
#
# COST. As the reference sweep: 3.4-5.2 h per size on a TITAN V. Walltime 12 h. Resumable
# per location via code/results/gate_a_rsaclasses_<size>px.progress.jsonl; a resume MUST
# reuse the same --n-images and --steps.
#
# OUTPUT. code/results/gate_a_rsaclasses_{10,20,30,40,50}px.json, merged with
#   python code/scripts/merge_shards.py --tag rsaclasses --job <id> --compare reference

#SBATCH --job-name=fyp-gateA-rsacls
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

export FYP_IMAGENET100="${FYP_IMAGENET100:-$REPO/code/data/imagenet100_rsa}"
test -d "$FYP_IMAGENET100/val" || {
    echo "ERROR: no seed-0 build at $FYP_IMAGENET100/val."
    echo "       Build it with build_imagenet100.py --classes rsa and upload it."
    exit 1; }

SIZES="10 20 30 40 50"
SIZE=$(echo $SIZES | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

echo "host  : $(hostname)"
echo "job   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURM_JOB_PARTITION"
echo "size  : ${SIZE}px"
echo "data  : $FYP_IMAGENET100  (RSA's seed-0 class draw)"
echo "model : timm DeiT-S, head sliced -- unchanged from gate_a_reference"
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
    --tag "rsaclasses_${SIZE}px"

echo "done: $(date)"
