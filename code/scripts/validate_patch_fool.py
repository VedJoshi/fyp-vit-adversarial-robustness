"""Validate `attacks.patch_fool` on the undefended model and record the result.

Patch-Fool is the attention-aware adversary measurements 2 and 3 are to be re-run
against, so it has to be shown to work before it carries a number. Three runs on one
batch, all at a 16 px token-aligned patch, which is Patch-Fool's own threat model:

    autopgd_fixed   PatchAutoPGD at one fixed token. The reference point.
    fool_fixed      Patch-Fool at that same token, so the only difference from the
                    line above is the objective - attention-aware against not.
    fool_selected   Patch-Fool choosing its own token by attention, which is the
                    attack as published.

The pass condition is the same one PatchAutoPGD was held to: **Patch-Fool at a fixed
location must be at least as strong as PatchAutoPGD at that location.** If it is
weaker, the implementation is wrong rather than the model robust. Choosing its own
location should then be stronger again; if it is not, patch selection is wrong.

Robust accuracy is conditioned on the clean prediction throughout - an image the model
already misclassifies is never robust, since the patch may hold the original pixels.

The helper checks below run first and cost nothing: they pin `token_patch_mask` and
`_pcgrad` against hand-computed answers, so a numerical failure downstream is not
silently blamed on the attack.

Usage::

    python scripts/validate_patch_fool.py                 # 32 images, ~7 min
    python scripts/validate_patch_fool.py --steps 50      # quicker smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fyp import attacks, config, data, models, results  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-images", type=int, default=config.SCALE.batch_size)
    p.add_argument("--steps", type=int, default=250, help="Patch-Fool iterations")
    p.add_argument("--autopgd-steps", type=int, default=100)
    p.add_argument("--token", type=int, default=90,
                   help="fixed image token for the matched comparison; 90 is row 6, col 6")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def check_helpers(device) -> dict:
    """Pin the two new helpers against answers worked out by hand."""
    out = {}

    # token_patch_mask: token 90 is row 6, col 6 -> pixels [96:112, 96:112].
    m = attacks.token_patch_mask(torch.tensor([[90]], device=device))
    assert m.shape == (1, 1, 224, 224), m.shape
    assert m.sum().item() == 256, m.sum().item()
    assert m[0, 0, 96:112, 96:112].min().item() == 1.0
    rows, cols = m[0, 0].nonzero(as_tuple=True)
    out["mask_bounds"] = [rows.min().item(), rows.max().item(),
                          cols.min().item(), cols.max().item()]
    assert out["mask_bounds"] == [96, 111, 96, 111], out["mask_bounds"]

    # Two tokens give two disjoint cells.
    m2 = attacks.token_patch_mask(torch.tensor([[0, 195]], device=device))
    assert m2.sum().item() == 512
    assert m2[0, 0, 0, 0].item() == 1.0 and m2[0, 0, 223, 223].item() == 1.0

    # _pcgrad: aligned rows pass through untouched, conflicting rows get projected.
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    c = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])          # row 0 aligned, row 1 opposed
    ref = attacks._pcgrad(a, c, normalise=False)
    nrm = attacks._pcgrad(a, c, normalise=True)
    assert torch.allclose(ref[0], a[0]) and torch.allclose(nrm[0], a[0])
    # ||c|| = 1 here, so the two forms agree and both give exact orthogonality.
    assert torch.allclose(nrm[1], torch.zeros(2), atol=1e-6), nrm[1]
    out["pcgrad_unit_norm_forms_agree"] = bool(torch.allclose(ref[1], nrm[1]))

    # ||c|| != 1 is where they separate: only `normalise=True` is orthogonal to c.
    a2 = torch.tensor([[1.0, 1.0]])
    c2 = torch.tensor([[-2.0, 0.0]])
    r_ref = attacks._pcgrad(a2, c2, normalise=False)
    r_nrm = attacks._pcgrad(a2, c2, normalise=True)
    out["pcgrad_residual_reference"] = float((r_ref * c2).sum())
    out["pcgrad_residual_normalised"] = float((r_nrm * c2).sum())
    assert abs(out["pcgrad_residual_normalised"]) < 1e-5
    assert abs(out["pcgrad_residual_reference"]) > 1e-3

    print("  helper checks pass  "
          f"mask {out['mask_bounds']}, "
          f"pcgrad residual reference {out['pcgrad_residual_reference']:+.3f} "
          f"vs normalised {out['pcgrad_residual_normalised']:+.1e}")
    return out


def main():
    args = parse_args()
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    print("\nhelper checks")
    helpers = check_helpers(device)

    ds, wnids, class_indices = data.build_dataset()
    clf, info = models.load_model(class_indices=class_indices, device=device)

    loader = data.make_loader(data.eval_subset(ds, args.n_images),
                              batch_size=args.n_images)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    clean = attacks.accuracy(clf, x, y)
    print(f"\n{x.shape[0]} images, clean accuracy {100.0 * clean.float().mean():.2f}%")

    top = (args.token // config.GRID) * config.PATCH_SIZE
    left = (args.token % config.GRID) * config.PATCH_SIZE
    fixed = torch.full((x.shape[0], 1), args.token, device=device)
    runs = {}

    def record(name, adv, seconds, **extra):
        rep = attacks.report(clean, attacks.robust_correct(clf, adv, y, clean))
        rep["minutes"] = round(seconds / 60, 2)
        rep.update(extra)
        runs[name] = rep
        print(f"  {name:<16} robust {rep['robust_acc']:6.2f}%   "
              f"asr {rep['asr']:6.2f}%   {rep['minutes']:.2f} min")
        return rep

    print(f"\n16 px patch at token {args.token} = pixels [{top}:{top + 16}, {left}:{left + 16}]")

    t0 = time.time()
    adv = attacks.patch_autopgd(clf, x, y, top, left, config.PATCH_SIZE,
                                steps=args.autopgd_steps)
    record("autopgd_fixed", adv, time.time() - t0, steps=args.autopgd_steps)

    t0 = time.time()
    adv, _ = attacks.patch_fool(clf, x, y, steps=args.steps, patches=fixed)
    record("fool_fixed", adv, time.time() - t0, steps=args.steps)

    t0 = time.time()
    adv, tokens = attacks.patch_fool(clf, x, y, steps=args.steps)
    sel = record("fool_selected", adv, time.time() - t0, steps=args.steps)
    sel["tokens"] = tokens[:, 0].tolist()
    sel["distinct_tokens"] = len(set(sel["tokens"]))

    # The attack must not have touched a pixel outside the token cell it reported.
    outside = ((adv - x).abs() *
               (1 - attacks.token_patch_mask(tokens))).max().item()
    print(f"\n  perturbation outside the selected patch: {outside:.2e}  "
          f"({'ok' if outside == 0.0 else 'LEAK'})")
    print(f"  {sel['distinct_tokens']} distinct tokens selected across {x.shape[0]} images")

    weaker = runs["fool_fixed"]["robust_acc"] > runs["autopgd_fixed"]["robust_acc"]
    print("\n" + ("  FAIL  Patch-Fool is weaker than PatchAutoPGD at the same location "
                  "- the implementation is wrong, not the model robust"
                  if weaker else
                  "  PASS  Patch-Fool at a fixed location is at least as strong as "
                  "PatchAutoPGD there"))

    payload = {
        "runs": runs,
        "helper_checks": helpers,
        "leak_outside_patch": outside,
        "fixed_token": args.token,
        "patch_px": config.PATCH_SIZE,
        "pass_at_least_as_strong_as_autopgd": not weaker,
        "reference": "GATECH-EIC/Patch-Fool, checked 29 Aug 2026",
        "model_info": info,
        "invocation": " ".join(sys.argv),
    }
    if not args.no_save:
        results.save("m4_patch_fool_validation", payload)


if __name__ == "__main__":
    main()
