#!/bin/bash
# Run ONCE on a login node (a login node), not through Slurm.
#
# Creates the Python environment in $HOME and pre-populates the timm/HuggingFace
# weight cache. Compute nodes are not guaranteed outbound internet, so
# `timm.create_model(..., pretrained=True)` must find the checkpoint already on
# disk when the job runs.
#
# Usage:  bash code/scripts/slurm/setup_env.sh
#
# Overridable:
#   FYP_REPO         vault root that contains code/   (default $HOME/fyp)
#   FYP_VENV         venv location                    (default $HOME/fyp-venv)
#   FYP_PYTHON       interpreter to build the venv    (default python3)
#   FYP_TORCH_INDEX  PyTorch wheel index              (default the cu118 index)
set -e

REPO="${FYP_REPO:-$HOME/fyp}"
VENV="${FYP_VENV:-$HOME/fyp-venv}"
PY="${FYP_PYTHON:-python3}"
TORCH_INDEX="${FYP_TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"

# /tmp on some clusters is a node-local disk with a 10MB PER-USER QUOTA (confirmed via
# `quota -s`), not a size limit on /tmp itself. Any tool that stages large files through
# /tmp -- pip's build/download staging included -- fails after ~5-10MB with no obvious
# quota-related error message. Redirect to $HOME, which has terabytes of headroom.
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
if [ ! -d "$REPO/code/fyp" ]; then
    echo "ERROR: $REPO/code/fyp not found."
    echo "       FYP_REPO must point at the vault root (the folder containing code/)."
    exit 1
fi

# The package uses PEP 585 builtin generics (dict[...], list[...]) in annotations
# under `from __future__ import annotations`, plus `X | None` annotations. 3.9 is
# the floor; the repo was developed on 3.13.
"$PY" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit(f"ERROR: need Python >= 3.9, this is {sys.version.split()[0]}. "
                     "Set FYP_PYTHON to a newer interpreter (try `python3.11`, `python3.12`).")
print(f"python : {sys.version.split()[0]}  ({sys.executable})")
PY

if DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) && [ -n "$DRIVER" ]; then
    echo "driver : $DRIVER"
else
    echo "driver : no GPU on this login node - checked inside the job instead"
fi

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
"$PY" -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install --upgrade pip

# There is no requirements.txt in this repo. The third-party imports across
# code/fyp/ and code/scripts/ are exactly: torch, torchvision, timm, numpy.
# matplotlib is used only by the notebooks; jupyterlab is not needed on the cluster.
#
# Versions are PINNED to code/README.md's "Verified on ... torch 2.7.0+cu118,
# timm 1.0.15" line, on purpose. An unpinned `pip install timm` grabbed 1.0.29
# during cluster setup, which changed timm's Block.forward() to pass attn_mask/
# is_causal kwargs into the attention module - RSAAttention.forward() (rsa.py)
# predates that change and doesn't accept them, so every RSA-active forward pass
# raised `TypeError: RSAAttention.forward() got an unexpected keyword argument
# 'attn_mask'`. README.md itself warns "Re-run after any timm upgrade" - pinning
# to the verified set is the correct fix here, not patching rsa.py to silently
# swallow new kwargs from a version that was never validated against RSA's
# published numbers.
#
# cu118 is the most permissive build driver-wise; if `torch.cuda.is_available()`
# comes back False inside a job, re-run with e.g.
#   FYP_TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash .../setup_env.sh
# --resume-retries and --timeout guard against some clusters's flaky path to
# download.pytorch.org, which has been observed to stall mid-wheel-download.
pip install --resume-retries 20 --timeout 120 torch==2.7.0 torchvision==0.22.0 --index-url "$TORCH_INDEX"
pip install --resume-retries 20 --timeout 120 timm==1.0.15 numpy matplotlib

# --------------------------------------------------------------------------
# Checkpoint cache and import verification CANNOT run here.
# --------------------------------------------------------------------------
# This login node caps interactive shells at `ulimit -v` = 1GB virtual memory
# (confirmed: `ulimit -Hv` is also 1GB, so it cannot be raised - this is a hard
# policy limit, not a mistake). libtorch_cuda.so alone exceeds that when mapped,
# so `import torch` - and therefore `import timm`, which imports torch itself -
# fails here with "ImportError: libtorch_cuda.so: failed to map segment from
# shared object". This is expected: it is the cluster stopping heavy work on
# login nodes, per the login banner. Slurm jobs get their own resource limits,
# so torch loads fine once actually submitted.
#
# What that means in practice:
#   - The HF/timm checkpoint cache is NOT pre-warmed by this script. If compute
#     nodes have outbound internet, `models.load_model()`'s first call to
#     `timm.create_model(..., pretrained=True)` downloads it there instead, and
#     every job after that hits the now-populated $HOME/.cache/huggingface.
#   - If a job instead fails on a download/connection error, compute nodes do
#     NOT have internet, and checkpoints must be fetched on a machine that does
#     (e.g. your laptop) and copied into the same $HF_HOME path via scp/rsync.
#   - The real import/CUDA/dataset verification now happens inside
#     smoke_min.sbatch, which runs as an actual Slurm job with proper limits.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# --------------------------------------------------------------------------
# Dataset check (plain filesystem check - no Python import needed for this)
# --------------------------------------------------------------------------
DATA_DIR="${FYP_DATA_ROOT:-$REPO/code/data}/imagenet100/val"
if [ -d "$DATA_DIR" ]; then
    N=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "imagenet100  : $DATA_DIR  OK  ($N class folders)"
    [ "$N" -eq 100 ] || echo "  warning: expected 100 class folders, found $N"
else
    echo "ERROR: $DATA_DIR is missing."
    echo "       rsync/scp code/data/imagenet100 across, or set FYP_DATA_ROOT."
    exit 1
fi

echo
echo "venv : $VENV"
echo "repo : $REPO"
echo "next : sbatch code/scripts/slurm/smoke_min.sbatch"
echo "       (this is where torch/CUDA import and the checkpoint download are"
echo "        actually verified - they cannot run on this login node)"
