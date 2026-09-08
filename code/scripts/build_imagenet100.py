"""Build the ImageNet-100 validation split from an ILSVRC2012 validation set.

Reads the 100 wnids from `data.imagenet100_wnids()`, or from `data.rsa_class_wnids()`
under ``--classes rsa``, and writes::

    <out>/val/<wnid>/*.JPEG

Two input forms are accepted:

- ``--zip`` a Kaggle-style archive whose members are ``<prefix>/<wnid>/*.JPEG``.
  Only the 100 needed wnids are extracted, so the full 50k images are never written
  to disk.
- ``--src`` a directory already laid out as ``<wnid>/*.JPEG``, which is copied.

The ILSVRC2012 validation tarball from image-net.org is a flat directory of 50,000
files with no class information, and needs the devkit ground-truth file to sort. It
is not a valid input here; use a source that is already in wnid folders.

Usage::

    python scripts/build_imagenet100.py --zip data/imagenet1k-val.zip
    python scripts/build_imagenet100.py --src /path/to/imagenet/val
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fyp import config, data  # noqa: E402

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def from_zip(zip_path: Path, out: Path, wnids: set[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        wanted = []
        for name in members:
            parts = name.split("/")
            if len(parts) < 2 or name.endswith("/"):
                continue
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            wnid = parts[-2]
            if wnid in wnids:
                wanted.append((name, wnid))

        if not wanted:
            raise SystemExit(
                f"no members of {zip_path.name} matched the 100 wnids.\n"
                f"first few archive entries: {members[:3]}"
            )

        print(f"extracting {len(wanted)} images for {len({w for _, w in wanted})} classes")
        for i, (name, wnid) in enumerate(wanted, 1):
            dest_dir = out / wnid
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(name).name
            if not dest.exists():
                with zf.open(name) as src, open(dest, "wb") as fh:
                    shutil.copyfileobj(src, fh)
            counts[wnid] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(wanted)}")
    return counts


def from_dir(src: Path, out: Path, wnids: set[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for wnid in sorted(wnids):
        srcd = src / wnid
        if not srcd.is_dir():
            continue
        dest_dir = out / wnid
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in srcd.iterdir():
            if f.suffix.lower() in IMAGE_SUFFIXES:
                dest = dest_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                counts[wnid] += 1
    if not counts:
        raise SystemExit(f"no wnid folders from the subset were found under {src}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", type=Path, help="archive laid out as <prefix>/<wnid>/*.JPEG")
    g.add_argument("--src", type=Path, help="directory laid out as <wnid>/*.JPEG")
    ap.add_argument("--out", type=Path, default=config.DATA_ROOT / "imagenet100",
                    help="destination root; the split is written to <out>/val")
    ap.add_argument("--split", default="val")
    ap.add_argument("--classes", choices=("local", "rsa"), default="local",
                    help="'local' is data.imagenet100_wnids(), this project's seed-1234 "
                         "draw; 'rsa' is data.rsa_class_wnids(), the seed-0 draw their "
                         "released code makes. The two overlap in 14 of 100 classes, so "
                         "they need separate destinations")
    args = ap.parse_args()

    wnids = set(data.rsa_class_wnids() if args.classes == "rsa" else data.imagenet100_wnids())
    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    draw = ("RSA's seed-0 draw (data.rsa_class_wnids)" if args.classes == "rsa"
            else f"local seed-{config.CLASS_SUBSET_SEED} draw (data.imagenet100_wnids)")
    print(f"subset: {len(wnids)} wnids, {draw}")
    print(f"destination: {out}")

    counts = from_zip(args.zip, out, wnids) if args.zip else from_dir(args.src, out, wnids)

    found = sorted(counts)
    missing = sorted(wnids - set(found))
    total = sum(counts.values())
    per_class = sorted(counts.values())

    print(f"\n{total} images across {len(found)} classes")
    print(f"images per class: min {per_class[0]}, max {per_class[-1]}")
    if missing:
        print(f"MISSING {len(missing)} classes: {missing[:8]}")
        raise SystemExit(1)
    print("\nall 100 classes present. Point the loader at it with:")
    print(f'  $env:FYP_IMAGENET100 = "{args.out}"')


if __name__ == "__main__":
    main()
