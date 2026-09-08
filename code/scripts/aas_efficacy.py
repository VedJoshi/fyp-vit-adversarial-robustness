"""Does pre-softmax attention scaling strengthen the attack on these checkpoints?

The bounded efficacy check rung L2 needs before it comes off the attack ladder.

Measurement 1 settled a *different* question. It found that the mechanism Jain & Dutta
propose is absent here: DeiT-S peaks at a pre-softmax logit gap of 11.19 against the
~103 float32 underflow threshold, so no attention row is saturated hard enough for its
non-maximal entries to vanish, and there is no lost gradient for a smaller `lambda` to
restore. That says the *stated* mechanism does not apply. It does not say the technique
fails to help, because a flatter softmax could smooth the loss surface whether or not
anything underflowed. Removing the rung on efficacy grounds needs this.

**What this implements, and what it does not.** AAS is *Adaptive* Attention Scaling
(Jain & Dutta, CVPR 2024, Algorithm 1): one scaling factor per attention block, found
by ten steps of gradient ascent on an LPIPS feature distance between the normal and the
scaled model, clamped to `[1e-7, 1]` so only scaling down is allowed, after which the
attack is generated on the scaled model. This script implements their **`+Scale`
baseline** instead: one factor shared by every block, taken as the best of their own set
`{10, 1, 1e-1, ..., 1e-4}`, which is exactly how they describe that row - they say
per-block tuning is `O(m^t)` and they do not do it either. By their own Table 3 that
baseline carries **1.6 points of AAS's 2.1**, with the learned factors adding ~0.51%.
The question here is whether the technique does anything at all on this checkpoint, and
the baseline answers that at a fraction of the cost. Full Algorithm 1 is left unwritten.

Their Table 1, CIFAR-10, PGD-100 robust accuracy, for orientation: `lambda = 1` gives
61.10, `1e-2` gives 59.43 - the strongest - and `10` gives 66.78. So the effect they
report is **1.67 points**, and scaling down is the direction that helps.

**Two numbers per lambda, because scaling changes the model.** They report the clean
accuracy of the scaled model and note it moves by up to 1%; on a saturated network,
scaling down barely disturbs the argmax. `robust_on_scaled` follows that protocol.
`robust_on_unscaled` scores the same adversarial images on the untouched model, which is
the question an attack ladder actually asks: does routing gradients through a scaled
model break the real one better? If scaling collapses clean accuracy here, the technique
is inapplicable on this checkpoint for a concrete and measurable reason, and that is the
finding rather than the robust-accuracy column.

The one trap in the design: **the baseline must not already be at 0%.** A 32 px patch at
100 steps drives PatchAutoPGD to 0.00% on this model, and against a floor no scaling can
show an improvement - the same flat-ladder reading the 29 Aug review flagged for the eps
sweep. A 16 px patch at token 90 leaves about 9% robust, which is headroom in both
directions, and it is the setting `validate_patch_fool.py` already uses.

The seed is reset before every attack, so a given batch starts every variant from the
same random patch. `unscaled` and `lambda = 1.0` are the same computation by
construction and are both run: if they disagree, the context manager is not restoring
state and every other number here is suspect.

Usage::

    python scripts/aas_efficacy.py                  # 256 images, ~16 min
    python scripts/aas_efficacy.py --n-images 64    # quicker
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import islice
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fyp import attacks, config, data, models, results  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-images", type=int, default=config.SCALE.n_mechanism_images)
    p.add_argument("--steps", type=int, default=config.SCALE.attack_steps)
    p.add_argument("--patch-px", type=int, default=config.PATCH_SIZE)
    p.add_argument("--token", type=int, default=90, help="fixed patch token; 90 is row 6, col 6")
    # Jain & Dutta Table 1 sweeps exactly this set.
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    ds, wnids, class_indices = data.build_dataset()
    clf, info = models.load_model(class_indices=class_indices, device=device)

    top = (args.token // config.GRID) * config.PATCH_SIZE
    left = (args.token % config.GRID) * config.PATCH_SIZE
    n_batches = max(args.n_images // config.SCALE.batch_size, 1)
    n_total = n_batches * config.SCALE.batch_size

    # `None` is the unscaled control, run outside the context manager entirely.
    variants: list[tuple[str, float | None]] = [("unscaled", None)]
    variants += [(f"lambda={lam:g}", lam) for lam in args.lambdas]

    print(f"\n{n_batches} batches, {n_total} images, PatchAutoPGD(DLR) at {args.steps} steps, "
          f"{args.patch_px} px patch at ({top},{left})")
    print(f"variants: {', '.join(n for n, _ in variants)}\n")

    n_clean = 0
    n_clean_scaled = {name: 0 for name, _ in variants}
    n_rob_scaled = {name: 0 for name, _ in variants}
    n_rob_unscaled = {name: 0 for name, _ in variants}
    identity_gap: float | None = None
    t0 = time.time()

    loader = data.make_loader(data.eval_subset(ds, n_total))
    for bi, (x, y) in enumerate(islice(loader, n_batches)):
        x, y = x.to(device), y.to(device)
        clean = attacks.accuracy(clf, x, y)
        n_clean += int(clean.sum())
        first: torch.Tensor | None = None

        for name, lam in variants:
            # Same starting patch for every variant on this batch.
            config.seed_everything()
            if lam is None:
                adv = attacks.patch_autopgd(clf, x, y, top, left, args.patch_px,
                                            steps=args.steps)
                first = adv
                clean_s = clean
            else:
                with models.attention_scaling(clf, lam) as n_scaled:
                    assert n_scaled == info["depth"], n_scaled
                    # Their protocol reports the scaled model's own clean accuracy.
                    clean_s = attacks.accuracy(clf, x, y)
                    adv = attacks.patch_autopgd(clf, x, y, top, left, args.patch_px,
                                                steps=args.steps)
                    rob_s = attacks.robust_correct(clf, adv, y, clean_s)
                if lam == 1.0 and first is not None and identity_gap is None:
                    identity_gap = float((adv - first).abs().max())
                n_rob_scaled[name] += int(rob_s.sum())

            if lam is None:
                n_rob_scaled[name] += int(attacks.robust_correct(clf, adv, y, clean).sum())
            # Always score the same images on the untouched model as well.
            n_rob_unscaled[name] += int(attacks.robust_correct(clf, adv, y, clean).sum())
            n_clean_scaled[name] += int(clean_s.sum())

        print(f"  batch {bi + 1}/{n_batches}  {(time.time() - t0) / 60:.1f} min")

    pct = lambda d, name: 100.0 * d[name] / n_total
    clean_acc = 100.0 * n_clean / n_total
    base = pct(n_rob_unscaled, "unscaled")

    print(f"\nclean accuracy of the untouched model: {clean_acc:.2f}% over {n_total} images\n")
    print(f"{'variant':<14} {'clean(scaled)':>14} {'robust(scaled)':>15} "
          f"{'robust(unscaled)':>17} {'vs base':>9}")
    for name, _ in variants:
        d = pct(n_rob_unscaled, name) - base
        mark = "" if name == "unscaled" else f"{d:+.2f}"
        print(f"{name:<14} {pct(n_clean_scaled, name):13.2f}% {pct(n_rob_scaled, name):14.2f}% "
              f"{pct(n_rob_unscaled, name):16.2f}% {mark:>9}")

    if identity_gap is not None:
        ok = identity_gap == 0.0
        print(f"\nidentity check, unscaled vs lambda=1: max |diff| {identity_gap:.2e}  "
              f"{'OK' if ok else 'FAIL - the scale is not being restored'}")

    # Does scaling preserve the model at all? Theirs moves by at most ~1 point.
    drops = {name: clean_acc - pct(n_clean_scaled, name)
             for name, lam in variants if lam is not None and lam != 1.0}
    worst = max(drops, key=drops.get)
    print(f"\nclean-accuracy cost of scaling: up to {drops[worst]:.2f} points ({worst}); "
          f"Jain & Dutta report at most ~1 point")

    scaled_only = {n: pct(n_rob_unscaled, n) for n, l in variants if l is not None and l != 1.0}
    best = min(scaled_only, key=scaled_only.get)
    gain = base - scaled_only[best]
    print(f"best scaled variant on the untouched model: {best} at {scaled_only[best]:.2f}%, "
          f"{gain:+.2f} points against the unscaled attack "
          f"(their Table 1 effect is 1.67 points)")
    if gain > 1.0:
        verdict = "scaling strengthens the attack here - rung L2 stays on the ladder"
    elif gain < -1.0:
        verdict = "scaling weakens the attack here - rung L2 comes off on efficacy as well as mechanism"
    else:
        verdict = "no material difference - rung L2 comes off on efficacy as well as mechanism"
    print(verdict)

    minutes = round((time.time() - t0) / 60, 2)
    print(f"\n{minutes} min")

    payload = {
        "question": ("does pre-softmax attention scaling beat standard PatchAutoPGD on this "
                     "checkpoint; implements Jain & Dutta's `+Scale` baseline, not Algorithm 1"),
        "implements": "shared-lambda +Scale baseline",
        "not_implemented": ("full AAS Algorithm 1: per-block factors learned by 10 steps of "
                            "LPIPS ascent, clamped to [1e-7, 1]; adds ~0.51 points over "
                            "+Scale by their Table 3"),
        "clean_acc_untouched": clean_acc,
        "clean_acc_scaled_by_variant": {n: pct(n_clean_scaled, n) for n, _ in variants},
        "robust_acc_on_scaled_model": {n: pct(n_rob_scaled, n) for n, _ in variants},
        "robust_acc_on_untouched_model": {n: pct(n_rob_unscaled, n) for n, _ in variants},
        "baseline_unscaled": base,
        "best_scaled_variant": best,
        "gain_points_vs_unscaled": gain,
        "max_clean_accuracy_drop_from_scaling": drops[worst],
        "verdict": verdict,
        "identity_check_max_abs_diff": identity_gap,
        "reference_table1_cifar10_pgd100": {"1": 61.10, "1e-4": 60.23, "1e-3": 60.14,
                                            "1e-2": 59.43, "1e-1": 60.87, "10": 66.78},
        "lambdas": args.lambdas,
        "patch_px": args.patch_px,
        "patch_token": args.token,
        "patch_top_left": [top, left],
        "steps": args.steps,
        "n_images": n_total,
        "headroom_note": ("16 px at this location leaves the unscaled attack well above 0%, "
                          "so an improvement is detectable; a 32 px patch saturates at 0.00% "
                          "and would make the comparison vacuous"),
        "mechanism_note": ("measurement 1 found peak logit gap 11.19 on DeiT-S against the "
                           "~103 underflow threshold, so the saturation the technique exists "
                           "to relieve is absent; this measures efficacy regardless"),
        "minutes": minutes,
        "model_info": info,
        "invocation": " ".join(sys.argv),
    }
    if not args.no_save:
        results.save("m7_aas_efficacy", payload)


if __name__ == "__main__":
    main()
