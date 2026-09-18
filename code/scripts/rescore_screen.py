"""Screen rescoring rules against the saved capture, offline.

A rescoring rule maps the 196 block-0 anomaly scores to a new map and takes the
argmax as the anchor. The mask geometry is unchanged: the same centred block on
whatever anchor wins. Only the ranking changes.

Any strictly monotone per-token transform leaves the argmax alone, so the rules
worth screening use spatial structure. The monotone entries are kept as nulls
that must reproduce the baseline exactly.

Coverage here is a geometric screening quantity on saved attacks, not recovered
accuracy, and says nothing about an attacker adapted to the rule.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import config, metrics, results  # noqa: E402

G = config.GRID


def pool(s: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of the k x k neighbourhood of each token, averaging over valid cells only."""
    g = s.view(-1, 1, G, G)
    lo = (k - 1) // 2
    pad = (lo, k - 1 - lo) * 2
    num = F.avg_pool2d(F.pad(g, pad), k, 1)
    den = F.avg_pool2d(F.pad(torch.ones_like(g), pad), k, 1)
    return (num / den).flatten(1)


def rules(lambdas: list[float]) -> dict:
    """Name to callable on an (n, 196) score map. The first entry is arm 5."""
    out = {
        "max": lambda s: s,
        "monotone_null": lambda s: 3.0 * s + 7.0,
        "pool2": lambda s: pool(s, 2),
        "pool3": lambda s: pool(s, 3),
        "pool5": lambda s: pool(s, 5),
    }
    for lam in lambdas:
        # Centre-surround: reward a token that stands above its wider neighbourhood.
        out[f"contrast{lam:g}"] = lambda s, lam=lam: s - lam * (pool(s, 5) - pool(s, 3))
        # Blob: reward a token whose 2x2 neighbourhood is high, not just itself.
        out[f"blob{lam:g}"] = lambda s, lam=lam: s + lam * pool(s, 2)
    return out


def features(s: torch.Tensor) -> torch.Tensor:
    """(n, 196, F) score-derived features. No position, which would leak the location prior."""
    med = s.median(1, keepdim=True).values
    mad = (s - med).abs().median(1, keepdim=True).values + 1e-6
    rank = s.argsort(1, descending=True).argsort(1).float() / (G * G)
    return torch.stack([s, pool(s, 2), pool(s, 3), pool(s, 5),
                        pool(s, 5) - pool(s, 3), (s - med) / mad, rank], -1)


def ceiling(cap, covers, w, seeds: list[int], steps: int) -> list[dict]:
    """Fit a small network on the score map and score it on held-out images.

    A probe, not a defence and not a bound: one architecture on one feature set at
    this sample size. A poor held-out column says these features did not yield a
    rule worth developing here, not that no function of them could. The held-out
    column is the one that means anything, since fitting 512 cases in sample
    overstates badly.
    """
    X, Y, img = [], [], []
    for loc, byw in cap["by"].items():
        sc = byw[str(float(w))]["scores_own"]
        X.append(features(sc))
        Y.append(covers[loc].unsqueeze(0).expand(len(sc), -1).float())
        img.append(torch.arange(len(sc)))
    X, Y, img = torch.cat(X), torch.cat(Y), torch.cat(img)
    X = (X - X.mean((0, 1))) / (X.std((0, 1)) + 1e-6)
    n = int(img.max()) + 1
    out = []
    for seed in seeds:
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        held = set(perm[: n // 2].tolist())
        tr = torch.tensor([int(i) not in held for i in img])
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[-1], 64), torch.nn.GELU(),
                                  torch.nn.Linear(64, 64), torch.nn.GELU(),
                                  torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(net(X[tr]).squeeze(-1), Y[tr])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = net(X).squeeze(-1)
            out.append({
                "seed": seed,
                "fitted_pct": 100.0 * float(Y[tr].gather(1, p[tr].argmax(1, True)).mean()),
                "held_out_pct": 100.0 * float(Y[~tr].gather(1, p[~tr].argmax(1, True)).mean()),
                "arm5_held_out_pct": 100.0 * float(
                    Y[~tr].gather(1, X[~tr, :, 0].argmax(1, True)).mean())})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="selection_capture_dev")
    ap.add_argument("--lambdas", default="0.5,1,2,4,8")
    ap.add_argument("--ceiling", action="store_true",
                    help="also fit a small network on the score map and score it on "
                         "held-out images, bounding what any rule of these features can do")
    ap.add_argument("--ceiling-seeds", default="0,1,2")
    ap.add_argument("--ceiling-steps", type=int, default=4000)
    ap.add_argument("--out", default="rescore_screen")
    args = ap.parse_args()

    cap = torch.load(config.RESULTS_ROOT / f"{args.capture}.pt", weights_only=True)
    m = cap["meta"]
    lam = [float(v) for v in args.lambdas.split(",")]
    rule = rules(lam)
    n = len(cap["index"])
    print(f"{args.capture}: {n} images, {len(cap['by'])} locations, weights {m['weights']}\n")

    covers = {loc: metrics.centred_coverage(*(int(v) for v in loc.split(",")),
                                            m["size"], m["window"]) for loc in cap["by"]}
    base_clean = cap["clean_scores"].argmax(1)

    head = f"{'rule':>14} |" + "".join(f"  cover w={w}" for w in m["weights"])
    print(head + f" | {'clean moved':>11} {'clean cover':>11}")
    print("-" * len(head + " |             "))
    rec: dict[str, dict] = {}
    for name, fn in rule.items():
        row: dict[str, float] = {}
        for w in m["weights"]:
            hit = tot = 0
            for loc, byw in cap["by"].items():
                chosen = fn(byw[str(float(w))]["scores_own"]).argmax(1)
                hit += int(covers[loc][chosen].sum())
                tot += len(chosen)
            row[f"cover_w{w}"] = 100.0 * hit / tot
        clean_sel = fn(cap["clean_scores"]).argmax(1)
        row["clean_moved_pct"] = 100.0 * float((clean_sel != base_clean).float().mean())
        row["clean_cover_pct"] = 100.0 * sum(
            int(covers[loc][clean_sel].sum()) for loc in cap["by"]) / (n * len(cap["by"]))
        rec[name] = row
        print(f"{name:>14} |" +
              "".join(f"{row[f'cover_w{w}']:10.1f}%" for w in m["weights"]) +
              f" | {row['clean_moved_pct']:10.1f}% {row['clean_cover_pct']:10.1f}%")

    assert rec["monotone_null"] == rec["max"], \
        "a monotone transform changed the argmax; the screen is not measuring what it claims"
    print("\ncover is the share of cases whose centred block fully covers the patch. "
          "clean moved is how often the rule picks a different anchor on the clean image, "
          "a disturbance proxy and not a clean-accuracy cost. monotone_null must equal max.")

    best = max((r for k, r in rec.items() if k != "monotone_null"),
               key=lambda r: r[f"cover_w{m['weights'][1]}"])
    print(f"\nbest coverage under the selection-aware attack: "
          f"{best[f'cover_w{m['weights'][1]}']:.1f}% against arm 5's "
          f"{rec['max'][f'cover_w{m['weights'][1]}']:.1f}%")

    bound = None
    if args.ceiling:
        seeds = [int(v) for v in args.ceiling_seeds.split(",")]
        bound = ceiling(cap, covers, m["weights"][1], seeds, args.ceiling_steps)
        print(f"\n{'seed':>5} {'fitted':>8} {'held out':>9} {'arm 5 there':>12}")
        print("-" * 37)
        for r in bound:
            print(f"{r['seed']:>5} {r['fitted_pct']:7.1f}% {r['held_out_pct']:8.1f}% "
                  f"{r['arm5_held_out_pct']:11.1f}%")
        print("fitted on the attacks it is scored against, so only the held-out column "
              "bounds anything.")

    results.save(args.out, dict(
        purpose="offline screen of rescoring rules on the saved capture; mask geometry "
                "is unchanged and only the ranking differs",
        source=f"{args.capture}.pt",
        caveat="coverage is geometric screening on existing attacks, not recovered "
               "accuracy and not robustness against an attacker adapted to the rule",
        note="any strictly monotone per-token transform leaves the argmax unchanged, so "
             "score-map rescoring can only act through spatial structure",
        settings=dict(lambdas=lam, **{k: m[k] for k in
                      ("size", "steps", "weights", "window", "window_rule",
                       "freeze_mask_after", "score_frame")}),
        rules=rec, ceiling=bound),
        n_eval_images=n, attack_steps=m["steps"])


if __name__ == "__main__":
    main()
