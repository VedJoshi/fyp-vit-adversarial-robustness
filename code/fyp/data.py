"""Datasets and class mappings for ImageNet-100 and Imagenette.

Sorted wnids keep dataset labels aligned with the sliced model head.
"""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from . import config

# Keep pixels in [0, 1]; normalization lives inside the model.
to_unit = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.IMAGE_SIZE),
    transforms.ToTensor(),
])


def wnid_index_map() -> dict[str, int]:
    """{'n01440764': 0, ...} for all 1000 classes, offline, straight from timm."""
    from timm.data import ImageNetInfo

    info = ImageNetInfo()
    return {info.index_to_label_name(i): i for i in range(info.num_classes())}


def wnid_descriptions() -> dict[str, str]:
    from timm.data import ImageNetInfo

    info = ImageNetInfo()
    return {info.index_to_label_name(i): info.index_to_description(i) for i in range(info.num_classes())}


def imagenet100_wnids(seed: int = config.CLASS_SUBSET_SEED, n: int = config.N_CLASSES) -> list[str]:
    """Cache a seeded class draw so later runs use the same subset."""
    cache = config.DATA_ROOT / f"imagenet100_wnids_seed{seed}.json"
    if cache.exists():
        wnids = json.loads(cache.read_text())
        if len(wnids) == n:
            return wnids

    all_wnids = sorted(wnid_index_map())
    wnids = sorted(random.Random(seed).sample(all_wnids, n))
    cache.write_text(json.dumps(wnids, indent=1))
    return wnids


def rsa_class_wnids(n: int = config.N_CLASSES) -> list[str]:
    """Reproduce the released RSA class draw: NumPy seed 0 over sorted wnids.

    Python's random.Random uses a different generator and does not reproduce it.
    """
    import numpy as np

    cache = config.DATA_ROOT / "imagenet100_wnids_rsa_seed0.json"
    if cache.exists():
        wnids = json.loads(cache.read_text())
        if len(wnids) == n:
            return wnids

    all_wnids = np.array(sorted(wnid_index_map()))
    np.random.seed(0)
    classes = np.sort(np.random.permutation(len(all_wnids))[:n])
    wnids = all_wnids[classes].tolist()
    cache.write_text(json.dumps(wnids, indent=1))
    return wnids


def ensure_imagenette(root: Path | None = None, size: str = "160px", split: str = "val") -> Path:
    """Download Imagenette if absent; return the directory of wnid subfolders."""
    from torchvision.datasets import Imagenette

    root = Path(root or config.DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    suffix = {"full": "imagenette2", "320px": "imagenette2-320", "160px": "imagenette2-160"}[size]
    split_dir = root / suffix / split
    if not split_dir.is_dir():
        print(f"downloading Imagenette ({size}) to {root} - about 98 MB, one time ...")
        Imagenette(str(root), split=split, size=size, download=True)
    return split_dir


def imagenet100_dir(split: str = "val") -> Path:
    """Dataset split directory, overridden by FYP_IMAGENET100.

    Expected layout: <root>/val/<wnid>/*.JPEG (or train/<wnid>/*.JPEG).
    """
    import os

    root = Path(os.environ.get("FYP_IMAGENET100", config.DATA_ROOT / "imagenet100"))
    return root / split


def build_dataset(
    dataset: str | None = None,
    split: str = "val",
    transform=to_unit,
    expect_wnids: Sequence[str] | None = None,
) -> tuple[ImageFolder, list[str], list[int]]:
    """Return (dataset, wnids, class_indices) in matching label order.

    Pass class_indices to models.load_model. expect_wnids also validates class
    identities; matching the class count alone is insufficient.
    """
    dataset = dataset or config.SCALE.dataset

    if dataset == "imagenette":
        directory = ensure_imagenette(split=split)
    elif dataset == "imagenet100":
        directory = imagenet100_dir(split)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"ImageNet-100 not found at {directory}.\n"
                f"Either set $FYP_IMAGENET100 to point at it, or stay on "
                f"config.DEV (Imagenette) until the data is in place. "
                f"See imagenet100_dir.__doc__ for the expected layout."
            )
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    ds = ImageFolder(str(directory), transform=transform)
    wnids = list(ds.classes)  # ImageFolder sorts these

    w2i = wnid_index_map()
    missing = [w for w in wnids if w not in w2i]
    if missing:
        raise ValueError(f"folder names are not ImageNet wnids: {missing[:5]}")
    class_indices = [w2i[w] for w in wnids]

    assert wnids == sorted(wnids), "wnid order must be sorted for targets to line up"
    assert len(class_indices) == len(wnids) == len(ds.classes)

    if dataset == "imagenet100" and len(wnids) != config.N_CLASSES:
        raise ValueError(
            f"{directory} has {len(wnids)} classes, expected {config.N_CLASSES}. The "
            f"head width follows this count, so every accuracy below it would be "
            f"against a different problem. Rebuild with scripts/build_imagenet100.py."
        )

    if expect_wnids is not None:
        expected = sorted(expect_wnids)
        if wnids != expected:
            overlap = len(set(wnids) & set(expected))
            raise ValueError(
                f"{directory} holds a different class subset: {overlap} of "
                f"{len(expected)} wnids overlap.\n"
                f"  on disk  : {wnids[:4]} ...\n"
                f"  expected : {expected[:4]} ...\n"
                f"Class order fixes the label indices, so this would evaluate a "
                f"different 100-way problem while every shape and order check passes. "
                f"Rebuild with scripts/build_imagenet100.py against the expected list."
            )

    return ds, wnids, class_indices


def eval_subset(
    ds: ImageFolder,
    n: int | None = None,
    seed: int = config.SEED,
) -> Subset:
    """Draw n images deterministically and sort them for stable batching.

    Request the required size directly: a prefix of a sorted draw is biased.
    This uses our sampler, not the released RSA image-sampling sequence.
    """
    n = n or config.SCALE.n_eval_images
    n = min(n, len(ds))
    idx = random.Random(seed).sample(range(len(ds)), n)
    return Subset(ds, sorted(idx))


def dev_subset(ds: ImageFolder,
               n: int | None = None,
               eval_n: int | None = None,
               seed: int = config.SEED) -> Subset:
    """Draw tuning images from the complement of eval_subset(ds, eval_n)."""
    eval_n = eval_n or config.SCALE.n_eval_images
    held = set(random.Random(seed).sample(range(len(ds)), min(eval_n, len(ds))))
    pool = [i for i in range(len(ds)) if i not in held]
    n = min(n or len(pool), len(pool))
    return Subset(ds, sorted(random.Random(seed + 1).sample(pool, n)))


def matched_positions(n_small: int, n_large: int, dataset_size: int,
                      seed: int = config.SEED) -> list[int]:
    """Map the smaller draw's positions to matching images in the larger draw.

    Returns positions in the smaller draw's sorted order. Raises ValueError if
    the draws do not nest; using the same seed does not guarantee nesting.
    """
    if n_small > n_large:
        n_small, n_large = n_large, n_small
    small = random.Random(seed).sample(range(dataset_size), n_small)
    large = sorted(random.Random(seed).sample(range(dataset_size), n_large))
    pos = {d: i for i, d in enumerate(large)}
    missing = [d for d in small if d not in pos]
    if missing:
        raise ValueError(
            f"the {n_small}-image draw is not a subset of the {n_large}-image draw at "
            f"seed {seed} over {dataset_size} images: {len(missing)} of {n_small} "
            f"images are absent. The {n_small - len(missing)} shared images are still a "
            f"common population, but the smaller sample cannot be recovered whole, so "
            f"any comparison would run over a reduced and self-selected subset"
        )
    return [pos[d] for d in sorted(small)]


def make_loader(
    ds,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Build a loader; default to no worker processes for Windows notebooks."""
    return DataLoader(
        ds,
        batch_size=batch_size or config.SCALE.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.get_device().type == "cuda",
    )


def describe(ds, wnids, class_indices) -> str:
    desc = wnid_descriptions()
    head = ", ".join(desc[w].split(",")[0] for w in wnids[:6])
    more = f", ... (+{len(wnids) - 6} more)" if len(wnids) > 6 else ""
    return (
        f"{len(ds)} images across {len(wnids)} classes\n"
        f"  classes  : {head}{more}\n"
        f"  1k indices: {class_indices[:6]}{' ...' if len(class_indices) > 6 else ''}"
    )
