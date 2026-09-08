"""ImageNet-100 and Imagenette, with RSA's class-slicing contract.

A dataset here is a directory of WordNet-id subfolders, so `ImageFolder` orders
targets by `sorted(wnids)`. The same wnids map to their ImageNet-1k indices in that
order and are passed to `models.slice_head`, so loader target `t` lines up with logit
column `t`. `build_dataset` asserts that ordering.

``imagenette`` is 10 ImageNet classes, ~98 MB; ``imagenet100`` matches RSA. Both use
the same wnid-folder layout and the same slicing path.

Head slicing involves no training (`models.slice_head`), so only the validation split
is used until Act 2's patch adversarial training.
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

# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
# Produces a [0,1] tensor and NOTHING else. Normalisation lives inside the model
# (see models.NormalizedModel) so that attacks operate in natural pixel space.
# Resize(256) -> CenterCrop(224) is the standard ImageNet eval transform.
to_unit = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.IMAGE_SIZE),
    transforms.ToTensor(),
])


# --------------------------------------------------------------------------
# wnid <-> ImageNet-1k index
# --------------------------------------------------------------------------
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
    """The ImageNet-100 class subset: a seeded draw from the sorted 1000 wnids.

    RSA never publishes which 100 classes it used, so clean accuracy will differ from
    theirs by a point or two. The subset is cached to disk so it does not drift if
    this function changes.
    """
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
    """The 100 wnids RSA's released code evaluates on.

    `_construct_imdb` at `pycls/datasets/imagenet.py` of
    wagner-group/robust-self-attention sorts the wnid directories of the split and
    then draws::

        np.random.seed(0)
        classes = np.sort(np.random.permutation(len(self._class_ids))[:cfg.MODEL.NUM_CLASSES])
        self._class_ids = self._class_ids[classes]
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

    so the label index of a wnid is its position in the sorted 100. That matches the
    ordering contract `build_dataset` asserts, and the draw is over the full 1000-wnid
    directory listing.

    The draw uses numpy, not `random.Random` as `imagenet100_wnids` does: the two
    generators return different subsets from the same seed, so the seed alone does not
    reproduce it.
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


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------
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
    """Where the real ImageNet-100 lives. Override with $FYP_IMAGENET100.

    Expected layout::

        <root>/val/n01440764/*.JPEG
        <root>/train/n01440764/*.JPEG      # only needed for Act 2

    To build it from ILSVRC2012: download the validation set (6.3 GB), arrange it
    into wnid folders, then keep only the wnids in `imagenet100_wnids()`. The
    100-class val split is ~5000 images, about 650 MB.
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
    """Return `(dataset, wnids, class_indices)` with all three in the same order.

    `class_indices[t]` is the ImageNet-1k index of the class the loader calls `t`.
    Pass it straight to `models.load_model(class_indices=...)`.

    `expect_wnids` asserts *which* 100 classes are on disk, not merely that there are
    100 of them in sorted order. Without it every internal check still passes on the
    wrong task: a directory of any 100 valid wnids builds a correctly ordered 100-way
    head and reaches plausible clean accuracy, so shape, order and accuracy cannot tell
    RSA's seed-0 draw from this project's seed-1234 draw, which share 14 of 100 classes.
    Pass `data.rsa_class_wnids()` for the authors' task and `data.imagenet100_wnids()`
    for this project's.

    A class count other than `config.N_CLASSES` is an error, not a warning: it changes
    the head width and therefore every accuracy below it.
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

    # The contract, asserted rather than assumed.
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
    """A deterministic random sample of `n` images.

    RSA evaluates on "512 randomly sampled images". The fixed seed gives every method
    in a comparison the same sample.

    **Ask for the size you are going to measure.** The returned indices are sorted, so
    the first k items of `eval_subset(ds, 512)` are the k *lowest* of 512 order
    statistics, not a random k. `ImageFolder` orders by class directory, so a prefix of
    a sorted sample over-represents early wnids: the first 32 of a 512-draw over 100
    classes reach about class 6 of 100. A run that wants 32 images and slices a 512-draw
    is estimating that prefix population, not a 32-image random sample. Pass `n=32`.

    The sort itself is deliberate and stays: it makes batch order stable, so an index
    recorded by one run names the same image in another.

    **This is not the authors' 512 images.** Their loader continues the same NumPy
    stream that drew the classes (`rsa_class_wnids`) to permute the assembled image
    database, so their sample depends on the class draw, on the directory listing order,
    and on how many images each class contributed; this one is an independent
    `random.Random(config.SEED)` draw over `ImageFolder`'s own ordering. Matching
    `rsa_class_wnids` fixes the *task*; it does not fix which 512 images are evaluated,
    and no transcription of their image sampling exists here to fix it with. A
    reproduction claim has to say so.
    """
    n = n or config.SCALE.n_eval_images
    n = min(n, len(ds))
    idx = random.Random(seed).sample(range(len(ds)), n)
    return Subset(ds, sorted(idx))


def make_loader(
    ds,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """A DataLoader over `ds`.

    `num_workers` defaults to 0: on Windows, worker processes re-import the module and
    interact badly with notebooks.
    """
    return DataLoader(
        ds,
        batch_size=batch_size or config.SCALE.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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
