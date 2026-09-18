"""CPU-only coverage screen of fixed attacks; does not evaluate classifier accuracy.

NMS greedily keeps anchors in descending score order (lower index breaks ties),
rejecting candidates within Chebyshev distance <= radius of a kept anchor.
Radius 0 is ordinary top-k. Windows use RSA's inward-shifted centred geometry.
Neither NMS nor k guarantees a fixed union size, especially at image edges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from selection_diagnostic import block_tokens
from fyp import config, metrics, rsa

KS = (1, 2, 3, 4, 8)
RADII = (0, 1, 2)


def select(scores, radius, k):
    chosen = []
    for t in sorted(range(len(scores)), key=lambda t: (-scores[t], t)):
        r, c = divmod(t, config.GRID)
        if all(max(abs(r - u // config.GRID), abs(c - u % config.GRID)) > radius
               for u in chosen):
            chosen.append(t)
            if len(chosen) == k:
                break
    if len(chosen) != k:
        raise ValueError(f"only {len(chosen)} anchors available for k={k}, radius={radius}")
    return chosen


def union(chosen, windows):
    mask = set()
    for t in chosen:
        mask.update(windows[t])
    return mask


def budget(sizes):
    return dict(mean=statistics.mean(sizes), median=statistics.median(sizes),
                minimum=min(sizes), maximum=max(sizes))


def self_check(windows):
    # Two individually non-covering windows jointly cover a 2x2 footprint.
    scores = [0.] * 196
    scores[46], scores[47], scores[89] = 10., 9., 8.
    assert select(scores, 0, 2) == [46, 47]
    assert select(scores, 1, 2) == select(scores, 2, 2) == [46, 89]
    foot = {60, 61, 74, 75}
    assert not foot <= windows[46] and not foot <= windows[89]
    assert foot <= union([46, 89], windows)
    assert select([0.] * 196, 0, 3) == [0, 1, 2]
    assert len(union([0, 1], windows)) == 9  # duplicate edge windows
    print("self-check: tie order, NMS, collective coverage and edge union size PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "results/selection_capture_dev.pt"
    ap.add_argument("--capture", type=Path, default=default)
    ap.add_argument("--out", type=Path, default=default.with_name("selection_screen_dev.json"))
    args = ap.parse_args()
    cfg = rsa.RSAConfig.for_patch(20, window_rule="centred", freeze_mask_after=0)
    mod = rsa.RSAAttention.__new__(rsa.RSAAttention)
    mod.cfg = cfg
    windows = [block_tokens(mod, t, 3, "centred") for t in range(196)]
    self_check(windows)
    cap = torch.load(args.capture, map_location="cpu", weights_only=True)
    meta = cap["meta"]
    assert (meta["size"], meta["window"], meta["window_rule"], meta["freeze_mask_after"]) == (
        20, 3, "centred", 0), "this screen is for the captured arm-5 protocol"
    ids = cap["index"].tolist()
    assert len(ids) == len(set(ids))
    n = len(ids)
    clean = cap["clean_scores"].tolist()
    assert cap["clean_scores"].shape == (n, 196) and torch.isfinite(cap["clean_scores"]).all()
    groups = []
    for loc, by_weight in cap["by"].items():
        top, left = map(int, loc.split(","))
        foot = set(metrics.patch_token_indices(top, left, 20, include_cls_offset=False))
        for w, rec in by_weight.items():
            assert rec["scores_own"].shape == (n, 196) and torch.isfinite(rec["scores_own"]).all()
            assert torch.equal(rec["scores_own"].argmax(1), rec["chosen"])
            new = cap["clean_ok"] & rec["wrong"] & ~by_weight["0.0"]["wrong"]
            groups.append((loc, w, rec, foot, new.tolist()))
    rows = []
    for radius in RADII:
        clean_orders = [select(s, radius, max(KS)) for s in clean]
        attack_orders = {(loc, w): [select(s, radius, max(KS))
                                   for s in rec["scores_own"].tolist()]
                         for loc, w, rec, foot, new in groups}
        for k in KS:
            per_location = {}
            totals = {}
            for loc, w, rec, foot, new in groups:
                masks = [union(a[:k], windows) for a in attack_orders[loc, w]]
                full = [foot <= m for m in masks]
                if k == 1:
                    assert full == [c == "full" for c in rec["coverage"]]
                stats = dict(full=sum(full), cases=n,
                             new_full=sum(f and b for f, b in zip(full, new)),
                             new_cases=sum(new),
                             masked_tokens=budget([len(m) for m in masks]))
                per_location.setdefault(loc, {})[w] = stats
                total = totals.setdefault(w, dict(full=0, cases=0, new_full=0,
                                                  new_cases=0, sizes=[]))
                for name in ("full", "cases", "new_full", "new_cases"):
                    total[name] += stats[name]
                total["sizes"].extend(len(m) for m in masks)
            for total in totals.values():
                total["masked_tokens"] = budget(total.pop("sizes"))
            row = dict(radius=radius, k=k,
                       clean_masked_tokens=budget([len(union(a[:k], windows))
                                                   for a in clean_orders]),
                       aggregate=totals, per_location=per_location)
            rows.append(row)
            a = totals["0.5"]
            print(f"radius={radius} k={k}: new coverage {a['new_full']}/{a['new_cases']}; "
                  f"all {a['full']}/{a['cases']}; tokens attack/clean "
                  f"{a['masked_tokens']['mean']:.1f}/{row['clean_masked_tokens']['mean']:.1f}")
    record = dict(purpose="fixed-attack geometry screening, not classifier accuracy",
                  capture=args.capture.name,
                  capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                  rule=__doc__, k_values=KS, radii=RADII, n_images=n,
                  new_cases="clean-correct and wrong under weight 0.5 but not 0 at the same location",
                  counts="image-location cases, not independent or distinct images",
                  rows=rows)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
