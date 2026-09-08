#!/bin/bash
# The two-minute check. Run this BEFORE smoke_test.sbatch.
#
#   sbatch code/scripts/slurm/smoke_min.sbatch
#
# It exercises exactly the Gate A code path -- dataset, sliced head, RSA install,
# patch_autopgd, the progress-jsonl write, results.save -- at the smallest size that
# still runs all of it:
#
#   --sizes 50        81 grid locations, the cheapest of the five patch sizes
#   --loc-stride 40   keeps locations 0, 40 and 80 of those 81, so three attacks
#   --n-images 32     one batch
#   --steps 10        instead of the 100 RSA specifies
#   --tag smoke       writes results/gate_a_smoke*.json, never touching a real tag
#
# Everything it produces is throwaway; delete it afterwards:
#   rm code/results/gate_a_smoke.json code/results/gate_a_smoke.progress.jsonl
#
# `gpu` is the short partition: 3 h maximum. Asking for 15 min or less earns the
# +200 "nice" bonus documented on the slurm-info page, so this queues at the highest
# partition priority available.

#SBATCH --job-name=fyp-min
# Partition is left to the environment: export SBATCH_PARTITION for your site.
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "host  : $(hostname)"
echo "job   : $SLURM_JOB_ID on $SLURM_JOB_PARTITION"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
    echo "ERROR: nvidia-smi failed on $(hostname). The node advertises a GPU it does not have;"
    echo "       add it to the --exclude line above and resubmit this array task."
    exit 1
fi

# 1. Imports, CUDA, cached weights, dataset. Fails in seconds if any of these is wrong.
#    Run without `srun` so the heredoc goes straight to python's stdin. The batch step
#    already holds the job's GPU allocation, so CUDA is visible here.
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("code").resolve()))
import torch
print("torch        :", torch.__version__)
print("cuda avail   :", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA device visible - wrong torch build or no --gpus"
print("device       :", torch.cuda.get_device_name(0))
from fyp import attacks, config, data, models, results, rsa
print(config.describe())
ds, wnids, class_indices = data.build_dataset()
print(f"dataset      : {len(ds)} images, {len(wnids)} classes")
clf, info = models.load_model(class_indices=class_indices, device=config.get_device())
print(f"model        : {info['n_params_m']}M params, {info['num_classes']}-way, "
      f"fused_attn disabled on {info['fused_attn_disabled']} modules")
with rsa.enable(clf, rsa.RSAConfig.for_patch(50)) as h:
    print("rsa          :", h)
    print("logits       :", tuple(clf(torch.rand(2, 3, 224, 224, device=config.get_device())).shape))
print("preflight OK")
PY

# 2. The Gate A path itself, three locations at 10 steps.
srun python code/scripts/gate_a.py --sizes 50 --loc-stride 40 --n-images 32 --steps 10 --tag smoke

echo
echo "progress file:"
cat code/results/gate_a_smoke.progress.jsonl
echo "done: $(date)"
