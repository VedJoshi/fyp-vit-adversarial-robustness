"""Screen top-k / NMS selection against the saved capture, offline.

Reads selection_capture_dev.pt and asks, for each (k, suppression radius):
what fraction of cases have the patch inside the union of the selected
blocks, how many distinct tokens that union actually masks, and what the
same rule costs on clean images.

Also decomposes the score change. clean_scores is the clean image under its
own reference, scores_ref is the attacked image under the clean reference,
and scores_own is the attacked image under its own. So scores_ref minus
clean_scores is the change in token values at a fixed reference, and
scores_own minus scores_ref is the reference's contribution at fixed values.

Coverage is a geometric screening quantity. It is not recovered accuracy and
says nothing about an attacker adapted to the rule being screened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import config, metrics, results, rsa  # noqa: E402
from selection_diagnostic import block_tokens  # noqa: E402

G = config.GRID


def anchor_blocks(window: int, rule: str = "centred") -> list[int]:
    """Bitmask of masked tokens per anchor, through the defense's corner rule."""
    mod = rsa.RSAAttention.__new__(rsa.RSAAttention)
    mod.cfg = rsa.RSAConfig.for_patch(20, window_rule=rule, freeze_mask_after=0)
    return [sum(1 << t for t in block_tokens(mod, a, window, rule)) for a in range(G * G)]


def nms_select(scores: torch.Tensor, k: int, radius: int) -> torch.Tensor:
    """Greedy top-k anchors, suppressing anchors within Chebyshev distance < radius.

    Ties break on the lower token index, matching argmax. radius=1 suppresses
    only the chosen anchor, which is plain top-k. radius=w separates anchors
    enough for interior w x w blocks to be disjoint, but border blocks shift
    inward and can still overlap, so the union sizes are the authority.
    Returns (B, k) anchor indices.
    """
    order = scores.argsort(dim=1, descending=True, stable=True)
    rows, cols = order // G, order % G
    out = torch.full((len(scores), k), -1, dtype=torch.long)
    for b in range(len(scores)):
        taken_r, taken_c, n = [], [], 0
        for j in range(G * G):
            r, c = int(rows[b, j]), int(cols[b, j])
            if any(max(abs(r - tr), abs(c - tc)) < radius
                   for tr, tc in zip(taken_r, taken_c)):
                continue
            out[b, n] = int(order[b, j])
            taken_r.append(r)
            taken_c.append(c)
            n += 1
            if n == k:
                break
    return out


def union_stats(sel: torch.Tensor, blocks: list[int], foot: int) -> tuple[list[bool], list[int]]:
    covered, sizes = [], []
    for row in sel.tolist():
        u = 0
        for a in row:
            if a >= 0:
                u |= blocks[a]
        covered.append(foot & ~u == 0)
        sizes.append(bin(u).count("1"))
    return covered, sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="selection_capture_dev")
    ap.add_argument("--k", default="1,2,3,4,6,8")
    ap.add_argument("--radius", default="1,2,3")
    ap.add_argument("--single", default="3,5,6,7",
                    help="sides of the one-block control, for area against placement")
    ap.add_argument("--out", default="selection_screen")
    args = ap.parse_args()

    ks = [int(v) for v in args.k.split(",")]
    radii = [int(v) for v in args.radius.split(",")]
    sides = [int(v) for v in args.single.split(",")]
    cap = torch.load(config.RESULTS_ROOT / f"{args.capture}.pt", weights_only=True)
    m = cap["meta"]
    blocks = anchor_blocks(m["window"], m["window_rule"])
    n = len(cap["index"])
    print(f"{args.capture}: {n} images, {len(cap['by'])} locations, weights {m['weights']}, "
          f"window {m['window']} ({m['window_rule']})\n")

    # newly broken by the selection term, per location, as image positions
    newly = {}
    for loc, byw in cap["by"].items():
        w0 = byw[str(float(m["weights"][0]))]["wrong"]
        w5 = byw[str(float(m["weights"][1]))]["wrong"]
        newly[loc] = (w5 & ~w0 & cap["clean_ok"]).nonzero().flatten().tolist()
    print(f"newly broken cases: {sum(len(v) for v in newly.values())} "
          f"over {len(newly)} locations\n")

    grid: dict[str, dict] = {}
    print(f"{'k':>2} {'r':>2} | {'cover all':>10} {'cover newly':>12} "
          f"{'union tokens':>13} {'clean tokens':>13}")
    print("-" * 60)
    for k in ks:
        for r in radii:
            cov_all, cov_new, size_all, size_clean = [], [], [], []
            for loc, byw in cap["by"].items():
                top, left = (int(v) for v in loc.split(","))
                foot = sum(1 << t for t in metrics.patch_token_indices(
                    top, left, m["size"], include_cls_offset=False))
                g = byw[str(float(m["weights"][1]))]
                sel = nms_select(g["scores_own"], k, r)
                cov, sz = union_stats(sel, blocks, foot)
                cov_all += cov
                size_all += sz
                cov_new += [cov[i] for i in newly[loc]]
                _, szc = union_stats(nms_select(cap["clean_scores"], k, r), blocks, foot)
                size_clean += szc
            rec = {
                "cover_all_pct": 100.0 * sum(cov_all) / len(cov_all),
                "cover_newly_pct": (100.0 * sum(cov_new) / len(cov_new)) if cov_new else None,
                "cover_newly": f"{sum(cov_new)}/{len(cov_new)}",
                "union_tokens_mean": sum(size_all) / len(size_all),
                "clean_union_tokens_mean": sum(size_clean) / len(size_clean),
            }
            grid[f"k={k},r={r}"] = rec
            print(f"{k:>2} {r:>2} | {rec['cover_all_pct']:9.1f}% "
                  f"{rec['cover_newly_pct']:11.1f}% {rec['union_tokens_mean']:13.1f} "
                  f"{rec['clean_union_tokens_mean']:13.1f}")

    # one centred block of each side, the control for "coverage tracks masked area"
    print()
    single: dict[str, dict] = {}
    print(f"{'side':>4} | {'cover all':>10} {'cover newly':>12} {'union tokens':>13}")
    print("-" * 46)
    for side in sides:
        sblocks = anchor_blocks(side, m["window_rule"])
        cov_all, cov_new, size_all = [], [], []
        for loc, byw in cap["by"].items():
            top, left = (int(v) for v in loc.split(","))
            foot = sum(1 << t for t in metrics.patch_token_indices(
                top, left, m["size"], include_cls_offset=False))
            sel = nms_select(byw[str(float(m["weights"][1]))]["scores_own"], 1, 1)
            cov, sz = union_stats(sel, sblocks, foot)
            cov_all += cov
            size_all += sz
            cov_new += [cov[i] for i in newly[loc]]
        single[str(side)] = {
            "cover_all_pct": 100.0 * sum(cov_all) / len(cov_all),
            "cover_newly": f"{sum(cov_new)}/{len(cov_new)}",
            "union_tokens_mean": sum(size_all) / len(size_all),
        }
        print(f"{side:>4} | {single[str(side)]['cover_all_pct']:9.1f}% "
              f"{single[str(side)]['cover_newly']:>12} "
              f"{single[str(side)]['union_tokens_mean']:13.1f}")

    # score decomposition on the footprint and on the winning anchor, both weights
    print()
    hdr = ("location", "w", "foot clean", "foot fixed-ref", "foot own", "won clean",
           "won own", "foot<won")
    print(f"{hdr[0]:>9} {hdr[1]:>4} | {hdr[2]:>10} {hdr[3]:>14} {hdr[4]:>9} "
          f"| {hdr[5]:>9} {hdr[6]:>8} {hdr[7]:>9}")
    print("-" * 78)
    decomp: dict = {}
    for loc, byw in cap["by"].items():
        top, left = (int(v) for v in loc.split(","))
        foot = metrics.patch_token_indices(top, left, m["size"], include_cls_offset=False)
        for w in m["weights"]:
            g = byw[str(float(w))]
            fc = cap["clean_scores"][:, foot].mean(1)
            fr = g["scores_ref"][:, foot].mean(1)
            fo = g["scores_own"][:, foot].mean(1)
            won = g["chosen"]
            wc = cap["clean_scores"].gather(1, won.unsqueeze(1)).squeeze(1)
            wo = g["scores_own"].gather(1, won.unsqueeze(1)).squeeze(1)
            d = {"foot_clean": float(fc.mean()), "foot_fixed_ref": float(fr.mean()),
                 "foot_own": float(fo.mean()), "won_clean": float(wc.mean()),
                 "won_own": float(wo.mean()),
                 "foot_max_below_won_pct": 100.0 * float(
                     (g["scores_own"][:, foot].max(1).values < wo).float().mean())}
            decomp.setdefault(loc, {})[str(w)] = d
            print(f"{loc:>9} {w:>4} | {d['foot_clean']:10.3f} {d['foot_fixed_ref']:14.3f} "
                  f"{d['foot_own']:9.3f} | {d['won_clean']:9.3f} {d['won_own']:8.3f} "
                  f"{d['foot_max_below_won_pct']:8.1f}%")
    print("\nfoot fixed-ref isolates the change in token values at the clean reference; "
          "foot own adds the reference's contribution. won clean is what the anchor that "
          "eventually wins already scored on the unattacked image. foot<won is the share "
          "of images whose best footprint token scores below the winning anchor.")

    results.save(args.out, dict(
        purpose="offline top-k/NMS screening and score decomposition on the saved capture",
        source=f"{args.capture}.pt",
        nms="greedy by score, ties on the lower token index; suppress anchors within "
            "Chebyshev distance < radius; radius 1 is plain top-k, radius 3 separates "
            "3x3 blocks in the interior but border blocks shift inward and can overlap",
        caveat="union coverage is geometric screening on existing attacks, not recovered "
               "accuracy and not robustness against an attacker adapted to the rule",
        settings=dict(k=ks, radius=radii, single_sides=sides, **{k: m[k] for k in
                      ("size", "steps", "weights", "window", "window_rule",
                       "freeze_mask_after", "score_frame", "backward")}),
        grid=grid, single_block=single, decomposition=decomp),
        n_eval_images=n, attack_steps=m["steps"])


if __name__ == "__main__":
    main()
