"""Tune a selection-aware attack against the centred, frozen-at-0 defense.

Add an exposing-minus-covering score margin to the classification loss.
Coverage uses the shifted mask, not token membership in the patch.

Uses disjoint development images only. Freeze the chosen weight before
full evaluation through transfer_attack.py --source selection.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import attacks, config, data, hooks, metrics, models, results, rsa  # noqa: E402
from gate_a import paired_seed  # noqa: E402


def selection_scores(clf, x: torch.Tensor, cfg) -> torch.Tensor:
    """Block-0 value-anomaly scores before replacement, with gradients; shape (B, N_img)."""
    with hooks.AttentionCapture(clf, layers=[0], store=("v",), detach=False) as cap:
        clf(x)
    return metrics.rsa_token_scores(cap.data[0].v, frame=cfg.score_frame)


def evasion_loss(clf, cfg, top: int, left: int, size: int, device):
    """Exposing-minus-covering score margin for block-0 centred selection.

    Positive forces exposure; zero leaves the outcome to argmax tie-breaking.
    """
    covers = metrics.centred_coverage(top, left, size, cfg.window).to(device)
    if not covers.any():
        raise SystemExit(
            f"no token at ({top},{left}) anchors a block covering the whole patch, so "
            f"the mask cannot cover it and there is nothing to evade; "
            f"window={cfg.window} is too small for this location.")

    def aux(adv: torch.Tensor) -> torch.Tensor:
        s = selection_scores(clf, adv, cfg)
        return (s.masked_fill(covers, float("-inf")).max(1).values
                - s.masked_fill(~covers, float("-inf")).max(1).values)

    return aux


def anchor_contains(window: int, rule: str, grid: int = config.GRID) -> torch.Tensor:
    """(A, T) boolean: whether anchor a's block, after the border shift, holds token t."""
    out = torch.zeros(grid * grid, grid * grid, dtype=torch.bool)
    for a in range(grid * grid):
        s = torch.full((1, grid * grid), -1.0)
        s[0, a] = 1.0
        r0, c0, h, w = rsa.RSAAttention._window_corner(None, s, window, grid, rule)
        for r in range(int(r0), int(r0) + int(h)):
            for c in range(int(c0), int(c0) + int(w)):
                out[a, r * grid + c] = True
    return out


def union_margins(scores: torch.Tensor, contains: torch.Tensor, foot: Sequence[int],
                  k: int, radius: int, grid: int = config.GRID) -> torch.Tensor:
    """(B, len(foot)) smallest step margin for keeping each patch token exposed.

    Walks the defence's greedy selection restricted to anchors that miss the token,
    applying the real suppression after each choice. A row entry above zero means
    those anchors also win unrestricted, so the real selection leaves that token
    outside the union; zero is inconclusive.

    Masked reductions use a finite -1e4 rather than -inf so no infinity reaches
    backward, which assumes scores stay above it. Value-anomaly scores are norms
    and cannot go negative.
    """
    ar = torch.arange(grid, device=scores.device)
    floor = -1e4                                  # finite, so no inf reaches backward
    out = []
    for t in foot:
        cov = contains[:, t]                                             # (A,)
        avail = torch.ones_like(scores, dtype=torch.bool)
        # A step with no covering anchor left cannot cover t, so it must not enter the
        # minimum: substituting a constant there would cap a larger real margin and
        # cut its gradient. Only steps that could still cover t bind.
        worst = torch.ones(len(scores), device=scores.device, dtype=scores.dtype)
        bound = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
        for _ in range(k):
            free_non, free_cov = avail & ~cov, avail & cov
            exposing, pick = scores.masked_fill(~free_non, floor).max(1)
            covering = scores.masked_fill(~free_cov, floor).max(1).values
            binds = free_cov.any(1)
            m = exposing - covering
            worst = torch.where(binds & bound, torch.minimum(worst, m),
                                torch.where(binds, m, worst))
            bound = bound | binds
            # No exposing anchor left: this token cannot be kept out of the union.
            worst = torch.where(free_non.any(1), worst, -torch.ones_like(worst))
            near_r = (ar[None, :] - (pick // grid)[:, None]).abs() < radius
            near_c = (ar[None, :] - (pick % grid)[:, None]).abs() < radius
            avail = avail & ~(near_r[:, :, None] & near_c[:, None, :]).flatten(1)
        out.append(worst)
    return torch.stack(out, dim=1)


def union_evasion_loss(clf, cfg, top: int, left: int, size: int, device):
    """Margin that drives the greedy union to leave one patch token exposed.

    For one patch token, walk the defence's own selection: at each step take the
    highest-scoring available anchor whose block misses that token, then apply the
    real suppression. The step's margin is that anchor's score minus the best
    still-available anchor that does cover the token. Every margin strictly
    positive makes each of those anchors the highest-scoring available anchor
    outright, so the real greedy selection makes the same choices and the token
    stays outside the union. Exposing one token is enough; the returned value is
    the largest such margin over the patch's tokens.

    Unlike a rank-k margin over individually non-covering anchors, this cannot be
    satisfied by windows that miss the patch singly and cover it jointly.

    A tie between a covering and a missing anchor gives margin zero, so only zero
    margins are inconclusive; ties among missing anchors resolve to the same lower
    index in both this walk and the defence. Exposing a token is still not
    misclassification, so success is read from the prediction and escape from
    `union_mask_escaped` on the returned candidate.
    """
    if cfg.n_windows < 2:
        raise ValueError("union_evasion_loss is for a multi-window config; "
                         "use evasion_loss for a single block")
    k, radius = cfg.n_windows, cfg.radius
    contains = anchor_contains(cfg.window, cfg.window_rule).to(device)
    foot = metrics.patch_token_indices(top, left, size, include_cls_offset=False)

    def aux(adv: torch.Tensor) -> torch.Tensor:
        s = selection_scores(clf, adv, cfg)                              # (B, A)
        return union_margins(s, contains, foot, k, radius).max(1).values

    return aux


def union_mask_escaped(clf, x: torch.Tensor, cfg, top: int, left: int,
                       size: int) -> torch.Tensor:
    """Whether the defence's union mask leaves any patch token exposed."""
    contains = anchor_contains(cfg.window, cfg.window_rule).to(x.device)
    foot = metrics.patch_token_indices(top, left, size, include_cls_offset=False)
    with torch.no_grad():
        sel = rsa.nms_anchors(selection_scores(clf, x, cfg), cfg.n_windows, cfg.radius)
    return ~contains[sel][:, :, foot].any(1).all(1)


def mask_escaped(clf, x: torch.Tensor, cfg, top: int, left: int, size: int) -> torch.Tensor:
    """Whether the supplied image's block-0 centred mask exposes part of the patch.

    Called on returned attack candidates, not intermediate iterates.
    """
    covers = metrics.centred_coverage(top, left, size, cfg.window).to(x.device)
    with torch.no_grad():
        s = selection_scores(clf, x, cfg)
    return ~covers[s.argmax(1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-images", type=int, default=64,
                    help="development images; this script always draws from "
                         "data.dev_subset, so a larger number buys more development "
                         "images and never the evaluation sample")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--aux-weight", default="0,0.5,2,8",
                    help="comma-separated; 0 is the existing classification-only attack")
    ap.add_argument("--locations", default="60,60 0,0 100,20 140,160",
                    help="space-separated top,left; development locations")
    ap.add_argument("--loss", default="dlr")
    ap.add_argument("--init", default="uniform")
    ap.add_argument("--schedule", default="counter")
    ap.add_argument("--window-rule", default="centred")
    ap.add_argument("--freeze-mask-after", type=int, default=0)
    ap.add_argument("--score-frame", default="global")
    ap.add_argument("--renorm", default="uniform")
    ap.add_argument("--backward", default="intended")
    ap.add_argument("--out", default="selection_attack_dev")
    args = ap.parse_args()

    weights = [float(w) for w in args.aux_weight.split(",") if w.strip()]
    if 0.0 not in weights:
        raise SystemExit("--aux-weight must include 0, the paired baseline arm")
    locations = [tuple(int(v) for v in loc.split(",")) for loc in args.locations.split()]

    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    ds, wnids, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    cfg = rsa.RSAConfig.for_patch(args.size, renormalise=args.renorm,
                                  window_rule=args.window_rule,
                                  score_frame=args.score_frame, backward=args.backward,
                                  freeze_mask_after=args.freeze_mask_after)

    sub = data.dev_subset(ds, args.n_images)
    xs, ys = [], []
    for x, y in data.make_loader(sub):
        xs.append(x.to(device))
        ys.append(y.to(device))
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    n = len(x_all)
    bs = args.batch_size

    print(f"\nselection-aware attack on development images, disjoint from the "
          f"{config.SCALE.n_eval_images}-image protocol sample")
    print(f"{n} images, {len(locations)} locations, {args.steps} steps, "
          f"window {cfg.window} ({cfg.window_rule}), freeze {cfg.freeze_mask_after}")
    print(f"aux weights {weights}\n")

    t_start = time.time()
    per_loc: dict[str, dict[float, dict]] = {}
    with rsa.enable(clf, cfg) as handle:
        clean_ok = torch.cat([attacks.accuracy(clf, x_all[i:i + bs], y_all[i:i + bs])
                              for i in range(0, n, bs)])
        clean_wrong = set((~clean_ok).nonzero(as_tuple=True)[0].tolist())
        print(f"clean accuracy {100.0 * float(clean_ok.float().mean()):.2f}%, "
              f"{len(clean_wrong)} clean errors folded into every arm\n")

        for top, left in locations:
            covers = int(metrics.centred_coverage(top, left, args.size, cfg.window).sum())
            print(f"({top:>3},{left:>3})  {covers} of {config.GRID ** 2} tokens anchor a "
                  f"covering block")
            by_w: dict[float, dict] = {}
            for w in weights:
                aux = evasion_loss(clf, cfg, top, left, args.size, device) if w else None
                wrong, escaped = set(), set()
                t0 = time.time()
                for s in range(0, n, bs):
                    take = torch.arange(s, min(s + bs, n), device=device)
                    # Pair initial patches across auxiliary weights.
                    config.seed_everything(paired_seed(args.size, top, left, s))
                    adv = attacks.patch_autopgd(clf, x_all[take], y_all[take],
                                                top=top, left=left, size=args.size,
                                                steps=args.steps, loss=args.loss,
                                                init=args.init, schedule=args.schedule,
                                                aux_loss=aux, aux_weight=w)
                    adv = adv.detach()
                    bad = ~attacks.accuracy(clf, adv, y_all[take])
                    wrong |= set(take[bad].tolist())
                    esc = mask_escaped(clf, adv, cfg, top, left, args.size)
                    escaped |= set(take[esc].tolist())
                lost = wrong | clean_wrong
                by_w[w] = {"wrong": sorted(lost), "escaped": sorted(escaped),
                           "seconds": round(time.time() - t0, 1)}
                base = set(by_w[0.0]["wrong"]) if 0.0 in by_w else set()
                print(f"    weight {w:>5}  mask escaped {100.0 * len(escaped) / n:6.2f}%   "
                      f"wrong {100.0 * len(lost) / n:6.2f}%   "
                      f"added over baseline {len(lost - base):>3}   "
                      f"{by_w[w]['seconds']:.0f}s")
            per_loc[f"{top},{left}"] = by_w

    print(f"\n{'location':>10}  {'weight':>6}  {'escaped':>8}  {'wrong':>8}  "
          f"{'added':>6}  {'lost':>5}")
    summary = {}
    for loc, by_w in per_loc.items():
        base = set(by_w[0.0]["wrong"])
        for w, rec in by_w.items():
            got = set(rec["wrong"])
            summary[f"{loc}@{w}"] = {
                "escaped_pct": 100.0 * len(rec["escaped"]) / n,
                "wrong_pct": 100.0 * len(got) / n,
                "added_over_baseline": sorted(got - base),
                "lost_vs_baseline": sorted(base - got),
            }
            print(f"{loc:>10}  {w:>6}  {100.0 * len(rec['escaped']) / n:7.2f}%  "
                  f"{100.0 * len(got) / n:7.2f}%  {len(got - base):>6}  "
                  f"{len(base - got):>5}")

    base_all = set().union(*(set(by_w[0.0]["wrong"]) for by_w in per_loc.values()))
    print(f"\nunion over all development locations, images broken at least once:")
    for w in weights:
        got = set().union(*(set(by_w[w]["wrong"]) for by_w in per_loc.values()))
        print(f"    weight {w:>5}   {len(got):>3} images   "
              f"{len(got - base_all):>3} the baseline never breaks")
    print("\n**Choose the weight on that last column**, images added to the union with the "
          "baseline over the development locations. Not the escape rate, which is a means "
          "and not the goal, and not a sum of per-location counts, which counts one image "
          "once per location it fails at. These are development images: report nothing "
          "from this table as a result.")
    results.save(args.out, dict(
        purpose="selection-aware attack against arm 5, development images",
        settings=dict(size=args.size, steps=args.steps, weights=weights,
                      locations=[list(l) for l in locations],
                      window=cfg.window, window_rule=cfg.window_rule,
                      freeze_mask_after=cfg.freeze_mask_after,
                      dev_disjoint_from=config.SCALE.n_eval_images),
        clean_wrong=sorted(clean_wrong),
        per_location={k: {str(w): v for w, v in d.items()} for k, d in per_loc.items()},
        summary=summary,
        hours=round((time.time() - t_start) / 3600, 3)),
        n_eval_images=n, attack_steps=args.steps)


if __name__ == "__main__":
    main()
