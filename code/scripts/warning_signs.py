"""Run Athalye's warning signs at `FULL` in either threat model and record the result.

The notebooks run these too. This script exists so the two `FULL` records can be
regenerated without opening a notebook, which matters whenever the scoring convention
changes and every recorded number has to move with it.

    linf   -> `diagnostics.warning_signs`, PGD over an eps ladder.
              Rewrites the `athalye_warning_signs` block of `results/m2_attacks.json`
              and leaves the rest of that record alone.
    patch  -> `diagnostics.patch_warning_signs` driven by `attacks.patch_autopgd`,
              over a patch-size ladder. Writes `results/m4_patch_warning_signs.json`.

Both run over `config.SCALE.n_mechanism_images` images at `config.SCALE.attack_steps`.
At `FULL` that is 256 images at 100 steps: about 18 minutes for `linf` and 15 for
`patch` on a 6 GB laptop GPU.

Usage::

    python scripts/warning_signs.py --threat linf
    python scripts/warning_signs.py --threat patch
    python scripts/warning_signs.py --threat both
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fyp import attacks, config, data, diagnostics, models, results  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threat", default="both", choices=("linf", "patch", "both"))
    p.add_argument("--patch-px", type=int, default=10,
                   help="patch size for signs 1 and 4 in the patch threat model")
    p.add_argument("--steps", type=int, default=config.SCALE.attack_steps)
    p.add_argument("--n-images", type=int, default=config.SCALE.n_mechanism_images)
    return p.parse_args()


def main():
    args = parse_args()
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    ds, wnids, class_indices = data.build_dataset()
    clf, info = models.load_model(class_indices=class_indices, device=device)

    n_batches = max(args.n_images // config.SCALE.batch_size, 1)
    print(f"\n{n_batches} batches, {n_batches * config.SCALE.batch_size} images, "
          f"{args.steps} steps\n")

    def batches():
        # A fresh loader per run, so each threat model sees the same images.
        return islice(data.make_loader(data.eval_subset(ds)), n_batches)

    if args.threat in ("linf", "both"):
        t0 = time.time()
        res = diagnostics.warning_signs(clf, batches(), steps=args.steps)
        res["minutes"] = round((time.time() - t0) / 60, 2)
        print()
        diagnostics.print_report(res)
        rec = results.load("m2_attacks")
        rec.pop("_meta", None)
        rec["athalye_warning_signs"] = res
        results.save("m2_attacks", rec)

    if args.threat in ("patch", "both"):
        t0 = time.time()
        res = diagnostics.patch_warning_signs(
            clf, batches(), size=args.patch_px, steps=args.steps,
            attack=attacks.patch_autopgd,
        )
        res["minutes"] = round((time.time() - t0) / 60, 2)
        res["model_info"] = info
        print()
        diagnostics.print_patch_report(res)
        results.save("m4_patch_warning_signs", res)


if __name__ == "__main__":
    main()
