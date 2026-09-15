"""Run and save warning-sign checks for linf, patch, or both threat models.

Keep the defense fixed as the attack budget changes. --patch-px sets its
mask size; --backward selects the derivative and --out names the record.
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
    p.add_argument("--freeze-mask-after", type=int, default=None, metavar="K",
                   help="blocks after K reuse block K's mask; omit for the unfrozen "
                        "defense. Required to point this battery at a frozen arm")
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
                                          backward=args.backward,
                                          freeze_mask_after=args.freeze_mask_after)
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
