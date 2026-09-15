"""Test whether a smaller patch finds failures missed inside a larger patch budget.

Hold the defense fixed and embed the smaller attack unchanged in the larger
region. These are freshly generated attacks, not a replay of the warning battery.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fyp import attacks, config, data, diagnostics, models, results, rsa  # noqa: E402


def contains(inner: int, outer: int, image_size: int = config.IMAGE_SIZE) -> bool:
    """Whether the centred `inner`-px square lies inside the centred `outer`-px one."""
    ti, to = diagnostics.centred(inner, image_size)[0], diagnostics.centred(outer, image_size)[0]
    return to <= ti and ti + inner <= to + outer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-images", type=int, default=256)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--outer", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--loss", default="dlr")
    ap.add_argument("--init", default="uniform")
    ap.add_argument("--schedule", default="counter")
    ap.add_argument("--window-rule", default="centred")
    ap.add_argument("--freeze-mask-after", type=int, default=0)
    ap.add_argument("--score-frame", default="global")
    ap.add_argument("--renorm", default="uniform")
    ap.add_argument("--backward", default="intended")
    ap.add_argument("--out", default="containment_check")
    args = ap.parse_args()

    if not contains(args.inner, args.outer):
        raise SystemExit(
            f"the centred {args.inner} px square is not inside the centred {args.outer} px "
            f"one, so embedding is not a legal {args.outer} px attack and this test does "
            f"not apply.")

    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    ds, wnids, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    cfg = rsa.RSAConfig.for_patch(args.outer, renormalise=args.renorm,
                                  window_rule=args.window_rule,
                                  score_frame=args.score_frame, backward=args.backward,
                                  freeze_mask_after=args.freeze_mask_after)

    sub = data.eval_subset(ds, args.n_images)
    xs, ys = [], []
    for x, y in data.make_loader(sub):
        xs.append(x.to(device))
        ys.append(y.to(device))
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    n = len(x_all)
    bs = args.batch_size

    ti, li = diagnostics.centred(args.inner)
    to, lo = diagnostics.centred(args.outer)
    print(f"\n{args.inner} px at ({ti},{li}) spans [{ti},{ti + args.inner}); "
          f"{args.outer} px at ({to},{lo}) spans [{to},{to + args.outer}) -- contained")
    print(f"{n} images, {args.steps} steps, defense fixed at window {cfg.window} "
          f"({cfg.window_rule}), freeze {cfg.freeze_mask_after}\n")

    t_start = time.time()
    wrong: dict[str, set[int]] = {k: set() for k in
                                  ("inner", "outer", "embedded")}
    with rsa.enable(clf, cfg) as handle:
        clean_ok = torch.cat([attacks.accuracy(clf, x_all[i:i + bs], y_all[i:i + bs])
                              for i in range(0, n, bs)])
        clean_wrong = set((~clean_ok).nonzero(as_tuple=True)[0].tolist())
        print(f"clean accuracy {100.0 * float(clean_ok.float().mean()):.2f}%\n")

        for s in range(0, n, bs):
            take = torch.arange(s, min(s + bs, n), device=device)
            x, y = x_all[take], y_all[take]

            config.seed_everything(config.SEED + s)
            adv_in = attacks.patch_autopgd(clf, x, y, top=ti, left=li, size=args.inner,
                                           steps=args.steps, loss=args.loss,
                                           init=args.init, schedule=args.schedule).detach()
            config.seed_everything(config.SEED + s)
            adv_out = attacks.patch_autopgd(clf, x, y, top=to, left=lo, size=args.outer,
                                            steps=args.steps, loss=args.loss,
                                            init=args.init, schedule=args.schedule).detach()

            # Rebuild and verify that the smaller attack is unchanged and within the outer budget.
            m_out = attacks.patch_mask(x.shape, to, lo, args.outer, device).bool()
            m_in = attacks.patch_mask(x.shape, ti, li, args.inner, device).bool()
            embedded = torch.where(m_in, adv_in, x)
            if not torch.equal(embedded, adv_in):
                raise SystemExit("embedding changed the inner attack's image")
            if not ((embedded != x) <= m_out).all():
                raise SystemExit("the embedded image perturbs pixels outside the "
                                 f"{args.outer} px region; it is not a legal attack")

            for key, adv in (("inner", adv_in), ("outer", adv_out),
                             ("embedded", embedded)):
                bad = ~attacks.accuracy(clf, adv, y)
                wrong[key] |= set(take[bad].tolist())
            print(f"  batch {s // bs + 1}/{(n + bs - 1) // bs}  "
                  f"{(time.time() - t_start) / 60:.1f} min")

    for k in wrong:
        wrong[k] |= clean_wrong
    union = wrong["embedded"] | wrong["outer"]

    def pct(sset):
        return 100.0 * (n - len(sset)) / n

    print(f"\n  robust accuracy, defense fixed at the {args.outer} px config")
    print(f"    {args.inner} px attack alone          {pct(wrong['inner']):6.2f}%")
    print(f"    {args.outer} px attack alone          {pct(wrong['outer']):6.2f}%")
    print(f"    {args.inner} px embedded in {args.outer} px    {pct(wrong['embedded']):6.2f}%")
    print(f"    union of the two {args.outer} px attacks {pct(union):6.2f}%")

    missed = wrong["embedded"] - wrong["outer"]
    print(f"\n  images the {args.outer} px attack missed that were inside its own budget: "
          f"{len(missed)}")
    if missed:
        print(f"  The {args.outer} px optimiser left {pct(wrong['outer']) - pct(union):.2f} points on the "
              f"table inside its own budget, so at least that much of the ladder is the "
              f"attack and not the defense.")
    else:
        print(f"  This rerun did not reproduce additional failures from the embedded "
              f"{args.inner} px attack. That does not explain the battery's inversion: "
              f"these are freshly generated attacks on a different seeding, and the "
              f"budget-inclusion argument is unaffected by how they came out.")

    results.save(args.out, dict(
        purpose="is the 10-vs-20px inversion an attack failure? centred containment test",
        settings=dict(inner=args.inner, outer=args.outer, steps=args.steps,
                      window=cfg.window, window_rule=cfg.window_rule,
                      freeze_mask_after=cfg.freeze_mask_after,
                      inner_at=[ti, li], outer_at=[to, lo]),
        robust_acc_pct={k: pct(v) for k, v in wrong.items()},
        union_pct=pct(union),
        missed_by_outer=sorted(missed),
        clean_wrong=sorted(clean_wrong),
        hours=round((time.time() - t_start) / 3600, 3)),
        n_eval_images=n, attack_steps=args.steps)


if __name__ == "__main__":
    main()
