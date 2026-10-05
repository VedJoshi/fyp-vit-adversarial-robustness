"""Screen neighbour-aware statistics on the block-0 value vectors, offline.

RSA's score is the distance from one global mean, per head, averaged. It cannot
see whether an unusual token has near neighbours among the image's own tokens.
The hypothesis is that a natural region outscoring the patch is ordinary content
with company elsewhere in the image, while an optimised patch is more isolated.

Two families, predeclared:

  knn{k}     distance to the k-th nearest token, self excluded
  knnfar{k}  the same, additionally excluding the token's surrounding 3x3 block,
             so that the three other tokens a 20 px patch spans cannot supply the
             company that would hide it

Distances are per head and averaged over heads, matching the frame the anomaly
score is computed in. Selection is the argmax and the mask geometry is unchanged
throughout: the same centred 3x3 block on whatever anchor wins.

Coverage here is a geometric screening quantity measured on attacks optimised
against the argmax. It is a transfer measurement, it is not recovered accuracy,
and it says nothing about an attacker adapted to one of these rules.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import config, metrics, results  # noqa: E402

G = config.GRID
GATE_W0 = 90.0    # classification-only coverage must not fall below this
GATE_W05 = 35.0   # selection-aware coverage must reach this

# The distance helpers live in fyp.metrics so the live defence and this screen
# cannot drift apart.


def rules(ks: list[int], far_ks: list[int]) -> dict:
    """Name to callable on (scores, distances). The first two entries are nulls."""
    out = {
        "max": lambda s, d: s,
        "monotone_null": lambda s, d: 3.0 * s + 7.0,
    }
    for k in ks:
        out[f"knn{k}"] = lambda s, d, k=k: metrics.kth_distance(d, k, 0)
    for k in far_ks:
        out[f"knnfar{k}"] = lambda s, d, k=k: metrics.kth_distance(d, k, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="selection_capture_dev")
    ap.add_argument("--values", default="selection_values_dev")
    ap.add_argument("--ks", default="4,6,8", help="k for self-excluded neighbours, k >= 4")
    ap.add_argument("--far-ks", default="1,4,8", help="k for 3x3-excluded neighbours")
    ap.add_argument("--out", default="neighbour_screen")
    args = ap.parse_args()

    device = config.get_device()
    print(config.describe())

    cap = torch.load(config.RESULTS_ROOT / f"{args.capture}.pt", weights_only=True)
    val = torch.load(config.RESULTS_ROOT / f"{args.values}.pt", weights_only=True)
    m = cap["meta"]
    assert val["index"].tolist() == cap["index"].tolist(), \
        "the vectors and the capture are not the same development draw"

    ks = [int(v) for v in args.ks.split(",")]
    far_ks = [int(v) for v in args.far_ks.split(",")]
    assert all(k >= 4 for k in ks), "self-excluded k must be at least 4; see DO_THIS_NEXT 3a"
    rule = rules(ks, far_ks)
    n, locs, weights = len(cap["index"]), list(cap["by"]), m["weights"]
    print(f"\n{args.values}: {n} images, {len(locs)} locations, weights {weights}")
    print(f"rules: {', '.join(rule)}\n")

    covers = {loc: metrics.centred_coverage(*(int(v) for v in loc.split(",")),
                                            m["size"], m["window"]).to(device)
              for loc in locs}

    # Distances are the expensive part; compute each set once and reuse per rule.
    dist = {"clean": metrics.head_mean_distances(val["clean_v"].to(device))}
    score = {"clean": cap["clean_scores"].to(device)}
    for loc in locs:
        for w in weights:
            key = (loc, w)
            dist[key] = metrics.head_mean_distances(val["by"][loc][str(w)].to(device))
            score[key] = cap["by"][loc][str(float(w))]["scores_own"].to(device)
    base_clean = score["clean"].argmax(1)

    rec: dict[str, dict] = {}
    for name, fn in rule.items():
        row: dict = {"by_location": {}}
        for w in weights:
            hit = tot = 0
            for loc in locs:
                chosen = fn(score[(loc, w)], dist[(loc, w)]).argmax(1)
                c = int(covers[loc][chosen].sum())
                row["by_location"].setdefault(loc, {})[f"w{w}"] = 100.0 * c / n
                hit += c
                tot += n
            row[f"cover_w{w}"] = 100.0 * hit / tot
        clean_sel = fn(score["clean"], dist["clean"]).argmax(1)
        row["clean_moved_pct"] = 100.0 * float((clean_sel != base_clean).float().mean())
        row["passes_gate"] = bool(row[f"cover_w{weights[1]}"] >= GATE_W05
                                  and row[f"cover_w{weights[0]}"] >= GATE_W0)
        rec[name] = row

    assert rec["monotone_null"] == rec["max"], \
        "a monotone transform changed the argmax; the screen is not measuring what it claims"

    head = f"{'rule':>14} |" + "".join(f"  cover w={w}" for w in weights)
    print(head + f" | {'clean moved':>11} {'gate':>6}")
    print("-" * (len(head) + 22))
    for name, row in rec.items():
        print(f"{name:>14} |" + "".join(f"{row[f'cover_w{w}']:10.1f}%" for w in weights) +
              f" | {row['clean_moved_pct']:10.1f}% "
              f"{'PASS' if row['passes_gate'] else 'fail':>6}")

    print(f"\nper location, coverage at weight {weights[1]}:")
    print(f"{'rule':>14} |" + "".join(f"{loc:>12}" for loc in locs))
    for name, row in rec.items():
        print(f"{name:>14} |" +
              "".join(f"{row['by_location'][loc][f'w{weights[1]}']:11.1f}%" for loc in locs))

    passed = [k for k, r in rec.items() if r["passes_gate"] and k not in ("max", "monotone_null")]
    print(f"\ngate: cover w={weights[1]} >= {GATE_W05}% and cover w={weights[0]} >= {GATE_W0}%. "
          f"{'PASSED: ' + ', '.join(passed) if passed else 'no candidate passed.'}")
    print("clean moved is a disturbance proxy, not a clean-accuracy cost, and is a "
          "tie-breaker rather than a veto: the oracle arms move every clean anchor and "
          "cost 1.56 and 2.35 points of clean accuracy, not more.")

    results.save(args.out, dict(
        purpose="offline screen of neighbour-aware statistics on block-0 value vectors; "
                "mask geometry unchanged, only the ranking differs",
        source=f"{args.values}.pt and {args.capture}.pt",
        caveat="coverage is geometric and measured on attacks optimised against the "
               "argmax, so it is a transfer quantity; it is not recovered accuracy and "
               "no rule here has been attacked adaptively",
        gate=dict(cover_w0_min=GATE_W0, cover_w05_min=GATE_W05,
                  note="discretionary project-management gate fixed before the vectors "
                       "existed; passing it justifies classification replay only"),
        settings=dict(ks=ks, far_ks=far_ks, distance="per-head L2, averaged over heads",
                      **{k: m[k] for k in ("size", "steps", "weights", "window",
                                           "window_rule", "score_frame", "freeze_mask_after")}),
        rules=rec, passed=passed), n_eval_images=n, attack_steps=m["steps"])


if __name__ == "__main__":
    main()
