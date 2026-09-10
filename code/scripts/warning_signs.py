"""Run Athalye's warning signs at `FULL` in either threat model and record the result.

The notebooks run these too. This script exists so the two `FULL` records can be
regenerated without opening a notebook, which matters whenever the scoring convention
changes and every recorded number has to move with it.

    linf   -> `diagnostics.warning_signs`, PGD over an eps ladder.
              Rewrites the `athalye_warning_signs` block of `results/m2_attacks.json`
              and leaves the rest of that record alone.
    patch  -> `diagnostics.patch_warning_signs` driven by `attacks.patch_autopgd`,
              over a patch-size ladder. Writes `results/m4_patch_warning_signs.json`.

`--defense rsa` installs RSA for the whole run at one fixed config, taken from
`--patch-px`. The config is fixed rather than following the size sweep because sign 5
asks whether robust accuracy falls as the attacker's budget grows against a **fixed**
defense; re-tuning the window per size would change the defense and the attacker's
budget together, and the monotonicity premise would not apply. `--backward` selects
which derivative the attack differentiates through, so the same defense can be measured
under both. `--out` names the record, so the two runs do not overwrite each other.

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
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fyp import attacks, config, data, diagnostics, models, results, rsa  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threat", default="both", choices=("linf", "patch", "both"))
    p.add_argument("--patch-px", type=int, default=10,
                   help="patch size for signs 1 and 4 in the patch threat model")
    p.add_argument("--steps", type=int, default=config.SCALE.attack_steps)
    p.add_argument("--n-images", type=int, default=config.SCALE.n_mechanism_images)
    p.add_argument("--defense", default="none", choices=("none", "rsa"))
    p.add_argument("--backward", default=rsa.DEFAULT_BACKWARD, choices=rsa.BACKWARDS,
                   help="which derivative the attack differentiates through; the "
                        "forward pass is identical either way")
    p.add_argument("--renorm", default="uniform", choices=rsa.RENORMALISATIONS)
    p.add_argument("--window-rule", default="ceil+1", choices=rsa.WINDOW_RULES)
    p.add_argument("--score-frame", default="global", choices=rsa.SCORE_FRAMES)
    p.add_argument("--out", default="m4_patch_warning_signs",
                   help="record name under results/")
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

    n_total = n_batches * config.SCALE.batch_size

    def batches():
        # A fresh loader per run, so each threat model sees the same images. The draw
        # is at `n_total`, not the front of a 512-draw: sorted indices make a prefix the
        # lowest order statistics of the sample, which over-represents early wnids.
        return islice(data.make_loader(data.eval_subset(ds, n_total)), n_batches)

    if args.threat in ("linf", "both"):
        t0 = time.time()
        res = diagnostics.warning_signs(clf, batches(), steps=args.steps)
        res["minutes"] = round((time.time() - t0) / 60, 2)
        print()
        diagnostics.print_report(res)
        rec = results.load("m2_attacks")
        rec.pop("_meta", None)
        rec["athalye_warning_signs"] = res
        results.save("m2_attacks", rec,
                     n_eval_images=n_total, attack_steps=args.steps)

    if args.threat in ("patch", "both"):
        if args.defense == "rsa":
            cfg = rsa.RSAConfig.for_patch(args.patch_px, renormalise=args.renorm,
                                          window_rule=args.window_rule,
                                          score_frame=args.score_frame,
                                          backward=args.backward)
            print(f"RSA active, fixed for the run: {cfg}\n")
            defense = rsa.enable(clf, cfg)
        else:
            cfg, defense = None, nullcontext()

        t0 = time.time()
        with defense:
            res = diagnostics.patch_warning_signs(
                clf, batches(), size=args.patch_px, steps=args.steps,
                attack=attacks.patch_autopgd,
            )
        res["minutes"] = round((time.time() - t0) / 60, 2)
        res["model_info"] = info
        res["defense"] = args.defense
        res["rsa_config"] = cfg.as_dict() if cfg is not None else None
        print()
        diagnostics.print_patch_report(res)
        results.save(args.out, res,
                     n_eval_images=n_total, attack_steps=args.steps)


if __name__ == "__main__":
    main()
