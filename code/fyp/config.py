"""Central configuration: paths, model, scale, seeding.

`SCALE` selects between DEV (Imagenette) and FULL (ImageNet-100, RSA's protocol).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
CODE_ROOT = Path(__file__).resolve().parent.parent
VAULT_ROOT = CODE_ROOT.parent
DATA_ROOT = Path(os.environ.get("FYP_DATA_ROOT", CODE_ROOT / "data"))
RESULTS_ROOT = Path(os.environ.get("FYP_RESULTS_ROOT", CODE_ROOT / "results"))
FIGURES_ROOT = RESULTS_ROOT / "figures"

for _p in (DATA_ROOT, RESULTS_ROOT, FIGURES_ROOT):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
# RSA calls its model "ViT-small ... the DeiT-small model (Touvron et al. 2020)".
# In timm that is `deit_small_patch16_224`: 22M params, 16x16 patches, 384-dim,
# 6 heads, 12 blocks.
MODEL_NAME = "deit_small_patch16_224"

IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID = IMAGE_SIZE // PATCH_SIZE          # 14
N_IMAGE_TOKENS = GRID * GRID             # 196
N_TOKENS = N_IMAGE_TOKENS + 1            # 197, including the CLS token at index 0
CLS_INDEX = 0


# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scale:
    """Run size. `dev` uses Imagenette; `full` matches RSA's protocol."""

    name: str
    dataset: str            # "imagenette" (10 classes, 98 MB) or "imagenet100"
    n_eval_images: int      # RSA evaluates on 512 sampled images
    batch_size: int
    attack_steps: int
    n_mechanism_images: int  # for the Stage-M measurements
    notes: str = ""


DEV = Scale(
    name="dev",
    dataset="imagenette",
    n_eval_images=64,
    batch_size=16,
    attack_steps=10,
    n_mechanism_images=32,
    notes="Imagenette, 64 images. Numbers are not comparable to RSA.",
)

FULL = Scale(
    name="full",
    dataset="imagenet100",
    n_eval_images=512,      # RSA: "accuracy is evaluated on 512 randomly sampled images"
    batch_size=32,          # raise on a bigger GPU; 6 GB laptop cards want <= 32
    attack_steps=100,       # RSA: PatchAutoPGD, 100 steps
    n_mechanism_images=256,
    notes="Matches RSA's evaluation protocol. Needs the ImageNet-100 subset on disk.",
)

# Active scale.
SCALE: Scale = FULL


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 0

# RSA says only "a random 100-class subset of ImageNet" and never publishes which
# 100. The subset here is a seeded draw from the sorted wnid list, so clean accuracy
# will differ from RSA's by a point or two because the classes differ.
CLASS_SUBSET_SEED = 1234
N_CLASSES = 100


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe() -> str:
    dev = get_device()
    lines = [
        f"model        : {MODEL_NAME}",
        f"scale        : {SCALE.name}  ({SCALE.dataset}, {SCALE.n_eval_images} eval images, "
        f"batch {SCALE.batch_size}, {SCALE.attack_steps} attack steps)",
        f"device       : {dev}",
        f"data root    : {DATA_ROOT}",
        f"results root : {RESULTS_ROOT}",
    ]
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        lines.append(
            f"gpu          : {torch.cuda.get_device_name(0)} "
            f"({free / 1e9:.1f} GB free / {total / 1e9:.1f} GB)"
        )
    if SCALE.notes:
        lines.append(f"note         : {SCALE.notes}")
    return "\n".join(lines)
