"""Adaptive pilot against the input-space finder, on development images.

The finder ranks tokens by a property of their 16x16 pixel tile and RSA masks the
centred 3x3 block on the winner. Under attacks optimised against the block-0
argmax it reaches oracle-level replay accuracy, but those attacks had no term
controlling their frequency content, so that number is a transfer measurement
and carries no robustness claim.

This attacks the finder's own selector. With C the anchors whose edge-adjusted
block covers every patch-intersected token, N the rest, and s_a the finder's
score for anchor a, the auxiliary objective is

    m(x) = max_{a in N} s_a(x) - max_{a in C} s_a(x)

maximised alongside the classification loss. A strictly positive margin means the
selected block leaves some patch token exposed; it does **not** mean the image is
misclassified, and the two are reported separately. A margin of exactly zero is
decided by argmax tie-breaking and is inconclusive, so ties are counted and
excluded from the evasion diagnostic rather than scored either way.

Selection is recomputed on every attacked input through a forward pre-hook, so
the mask tracks each attack iterate. Freezing across transformer blocks is the
defence; freezing across attack iterations would be attacking something else.

Weights are chosen here, on development images, and frozen before any evaluation
run. The 0.5 chosen for the block-0 objective has no meaning on this score scale,
so the sweep is logarithmic and predeclared.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import inputspace_screen  # noqa: E402
from fyp import attacks, config, data, metrics, models, results, rsa  # noqa: E402
from gate_a import paired_seed  # noqa: E402

WEIGHTS = (0.0, 0.1, 1.0, 10.0, 100.0)   # predeclared; 0.0 is classification-only


def finder_scores(name: str, x: torch.Tensor) -> torch.Tensor:
    """(B, 196) finder score map, differentiable in the pixels."""
    fn = inputspace_screen.RULES[name]
    return fn(x, inputspace_screen.blur_residual(x))


def install_finder(clf, handle, name: str, window: int):
    """Recompute the mask from the current input before every forward pass.

    Returns the hook handle. The mask is a discrete function of the input, so it
    is built under no_grad; the evasion margin is what carries gradient to the
    selection, exactly as the block-0 objective does.
    """
    def pre(module, args):
        with torch.no_grad():
            s = finder_scores(name, args[0])
            handle.set_mask(rsa.centred_token_mask(s.argmax(1), window))
    return clf.register_forward_pre_hook(pre)


def evasion_loss(name: str, top: int, left: int, size: int, window: int, device):
    """Exposing-minus-covering margin on the finder's own score map."""
    covers = metrics.centred_coverage(top, left, size, window).to(device)
    if not covers.any():
        raise SystemExit(f"no anchor at ({top},{left}) covers the patch with window={window}")

    def aux(adv: torch.Tensor) -> torch.Tensor:
        s = finder_scores(name, adv)
        return (s.masked_fill(covers, float("-inf")).max(1).values
                - s.masked_fill(~covers, float("-inf")).max(1).values)
    return aux


def hard_escape(name: str, adv: torch.Tensor, top: int, left: int, size: int,
                window: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(escaped, tied) per example, read from the real argmax rather than the margin."""
    covers = metrics.centred_coverage(top, left, size, window).to(device)
    with torch.no_grad():
        s = finder_scores(name, adv)
        best_n = s.masked_fill(covers, float("-inf")).max(1).values
        best_c = s.masked_fill(~covers, float("-inf")).max(1).values
        chosen = s.argmax(1)
    return ~covers[chosen], (best_n == best_c)


def smooth_init(x: torch.Tensor, top: int, left: int, size: int) -> torch.Tensor:
    """Patch initialised to a blurred draw, so the search does not start high-frequency."""
    adv = x.clone()
    noise = torch.rand_like(x[:, :, top:top + size, left:left + size])
    for _ in range(3):
        noise = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(noise, (1, 1, 1, 1), mode="reflect"), 3, 1)
    adv[:, :, top:top + size, left:left + size] = noise
    return adv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finder", default="tv", choices=list(inputspace_screen.RULES))
    ap.add_argument("--n-images", type=int, default=128)
    ap.add_argument("--locations", default="60,60 100,20")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--weights", default=",".join(str(w) for w in WEIGHTS))
    ap.add_argument("--smooth-init", action="store_true",
                    help="start the patch from a blurred draw (init='given') instead of "
                         "AutoPGD's uniform draw; without init='given' the blurred patch "
                         "is overwritten and the arm is only a reseed")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="finder_attack_dev")
    args = ap.parse_args()

    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    weights = [float(v) for v in args.weights.split(",")]
    locs = [tuple(int(v) for v in p.split(",")) for p in args.locations.split()]
    ds, wnids, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    cfg = rsa.RSAConfig.for_patch(args.size, window_rule="centred", freeze_mask_after=0)

    sub = data.dev_subset(ds, args.n_images)
    xs, ys = [], []
    for x, y in data.make_loader(sub):
        xs.append(x.to(device))
        ys.append(y.to(device))
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    n, bs = len(x_all), args.batch_size
    idx = list(sub.indices)

    print(f"\nfinder {args.finder}, {n} development images, {len(locs)} locations, "
          f"{args.steps} steps, weights {weights}"
          f"{', smooth init' if args.smooth_init else ''}\n")

    t_start = time.time()
    with rsa.enable(clf, cfg) as handle:
        hook = install_finder(clf, handle, args.finder, cfg.window)
        clean_ok = torch.cat([attacks.accuracy(clf, x_all[i:i + bs], y_all[i:i + bs])
                              for i in range(0, n, bs)])
        print(f"clean accuracy under the finder {100.0 * clean_ok.float().mean():.2f}%\n")

        rec: dict = {}
        for w in weights:
            per_loc = {}
            for top, left in locs:
                aux = evasion_loss(args.finder, top, left, args.size, cfg.window, device)
                ok, esc, tie = [], [], []
                for i in range(0, n, bs):
                    xb, yb = x_all[i:i + bs], y_all[i:i + bs]
                    torch.manual_seed(paired_seed(args.size, top, left, i))
                    start = smooth_init(xb, top, left, args.size) if args.smooth_init else xb
                    adv = attacks.patch_autopgd(clf, start, yb, top=top, left=left,
                                                size=args.size, steps=args.steps,
                                                init="given" if args.smooth_init else "uniform",
                                                aux_loss=aux, aux_weight=w)
                    ok.append(attacks.accuracy(clf, adv, yb))
                    e, t = hard_escape(args.finder, adv, top, left, args.size,
                                       cfg.window, device)
                    esc.append(e)
                    tie.append(t)
                ok, esc, tie = (torch.cat(v) for v in (ok, esc, tie))
                per_loc[f"{top},{left}"] = {
                    "correct": ok.tolist(),
                    "accuracy": 100.0 * float(ok.float().mean()),
                    "escaped_pct": 100.0 * float(esc[~tie].float().mean()) if (~tie).any() else None,
                    "tied": int(tie.sum())}
                print(f"  w={w:<6} ({top:>3},{left:>3})  acc {100.0 * float(ok.float().mean()):6.2f}%"
                      f"   escaped {per_loc[f'{top},{left}']['escaped_pct'] or float('nan'):5.1f}%"
                      f"   ties {int(tie.sum()):>3}")
            rec[str(w)] = per_loc
        hook.remove()

    # Union over weights: an image fails if any setting breaks it anywhere.
    print(f"\n{'weight':>8} {'mean acc':>9} {'survives both loc':>18}")
    summary = {}
    for w in weights:
        accs = [rec[str(w)][k]["accuracy"] for k in rec[str(w)]]
        surv = [i for i in range(n) if clean_ok[i] and all(
            rec[str(w)][k]["correct"][i] for k in rec[str(w)])]
        summary[str(w)] = {"mean_accuracy": sum(accs) / len(accs),
                           "survives": len(surv), "survives_pct": 100.0 * len(surv) / n}
        print(f"{w:>8} {sum(accs) / len(accs):8.2f}% {100.0 * len(surv) / n:17.2f}%")

    union = [i for i in range(n) if clean_ok[i] and all(
        rec[str(w)][k]["correct"][i] for w in weights for k in rec[str(w)])]
    broken_by_evasion = [idx[i] for i in range(n) if clean_ok[i]
                         and all(rec["0.0"][k]["correct"][i] for k in rec["0.0"])
                         and not all(rec[str(w)][k]["correct"][i]
                                     for w in weights if w for k in rec[str(w)])]
    print(f"\nunion over every weight and location: {len(union)}/{n} = "
          f"{100.0 * len(union) / n:.2f}% survive")
    print(f"images the classification-only attack keeps but an evasion weight breaks: "
          f"{len(broken_by_evasion)}")
    print("\naccuracy is measured through the real hard selector on the returned patch. "
          "escape is a diagnostic, not a success criterion; ties are excluded from it.")

    results.save(args.out, dict(
        purpose="adaptive pilot against the input-space finder on development images",
        finder=args.finder,
        caveat="development images and a subset of locations; not the 121-location "
               "protocol and not comparable with full-protocol figures. Weights are "
               "selected here and must be frozen before any evaluation run.",
        objective="max over non-covering anchors minus max over covering anchors, on the "
                  "finder's own score map; positive means the selected block leaves a "
                  "patch token exposed, which does not imply misclassification",
        settings=dict(weights=weights, locations=[f"{t},{l}" for t, l in locs],
                      steps=args.steps, size=args.size, smooth_init=args.smooth_init,
                      n_images=n, window=cfg.window, config=cfg.as_dict()),
        index=idx,
        indexing="index[i] is the dataset index of position i; every per-image list "
                 "in by_weight is in this order",
        clean_accuracy=100.0 * float(clean_ok.float().mean()),
        by_weight=rec, summary=summary,
        union_survives=len(union), union_survives_pct=100.0 * len(union) / n,
        broken_only_by_evasion=broken_by_evasion,
        hours=(time.time() - t_start) / 3600), n_eval_images=n, attack_steps=args.steps)


if __name__ == "__main__":
    main()
