"""Screen input-space statistics as RSA's selection rule, offline.

The score-map rules of rescore_screen.py and the neighbour statistics of
neighbour_screen.py both stop near 20% coverage under the selection-aware
attack. Both read block 0's value vectors, directly or through the scalar score
they summarise. This screen changes the grounding: it ranks tokens by a property
of the image pixels in the corresponding 16x16 tile, which is evidence neither
earlier family has.

Three rules, predeclared with their definitions before the run:

  tv      within-tile total variation: the mean absolute horizontal and vertical
          pixel difference, taken over pairs lying inside the tile
  blur    blur residual: the mean absolute difference from one fixed blur,
          computed on the whole image and then averaged into tiles
  colour  the tile mean colour's distance from the image mean colour

The blur is fixed at a 5x5 Gaussian, sigma 1.0, and is not tuned. No combination
of rules is screened. Statistics are computed on reconstructed attacked images in
the model's normalised input scale, the same scale for every rule and every image.

Selection is the argmax and the mask geometry is unchanged: the same centred 3x3
block on whatever anchor wins.

Coverage here is geometric, measured on attacks optimised against the argmax, so
it is a transfer quantity. It is not recovered accuracy, and no rule here has
been attacked adaptively.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import config, data, metrics, results  # noqa: E402

G = config.GRID
TILE = 16
BLUR_SIZE, BLUR_SIGMA = 5, 1.0   # fixed before the run; not tuned afterwards
GATE_W0 = 90.0
GATE_W05 = 35.0


def tiles(x: torch.Tensor) -> torch.Tensor:
    """(n, C, 224, 224) -> (n, C, 196, 16, 16), tiles in row-major token order."""
    n, c = x.shape[0], x.shape[1]
    t = x.view(n, c, G, TILE, G, TILE).permute(0, 1, 2, 4, 3, 5)
    return t.reshape(n, c, G * G, TILE, TILE)


def rule_tv(x: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
    """Mean absolute horizontal and vertical pixel difference within each tile."""
    t = tiles(x)
    dh = (t[..., :, 1:] - t[..., :, :-1]).abs().mean((-2, -1))
    dv = (t[..., 1:, :] - t[..., :-1, :]).abs().mean((-2, -1))
    return (0.5 * (dh + dv)).mean(1)


def rule_blur(x: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
    """Whole-image blur residual, averaged into tiles."""
    return tiles(resid).mean((-2, -1)).mean(1)


def rule_colour(x: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
    """Distance of the tile mean colour from the image mean colour."""
    tile_mean = tiles(x).mean((-2, -1))
    image_mean = x.mean((-2, -1), keepdim=False).unsqueeze(-1)
    return (tile_mean - image_mean).pow(2).sum(1).sqrt()


RULES = {"tv": rule_tv, "blur": rule_blur, "colour": rule_colour}


def blur_residual(x: torch.Tensor) -> torch.Tensor:
    """|x - gaussian_blur(x)| on the whole image, before any tile aggregation."""
    g = torch.arange(BLUR_SIZE, dtype=torch.float32, device=x.device) - (BLUR_SIZE - 1) / 2
    k1 = torch.exp(-g.pow(2) / (2 * BLUR_SIGMA ** 2))
    k1 = k1 / k1.sum()
    k = (k1[:, None] * k1[None, :]).expand(x.shape[1], 1, BLUR_SIZE, BLUR_SIZE)
    pad = BLUR_SIZE // 2
    sm = F.conv2d(F.pad(x, (pad,) * 4, mode="reflect"), k, groups=x.shape[1])
    return (x - sm).abs()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="selection_capture_dev")
    ap.add_argument("--out", default="inputspace_screen")
    args = ap.parse_args()

    device = config.get_device()
    print(config.describe())

    cap = torch.load(config.RESULTS_ROOT / f"{args.capture}.pt", weights_only=True)
    m = cap["meta"]
    size, weights, locs = m["size"], m["weights"], list(cap["by"])
    n = len(cap["index"])

    ds, wnids, class_indices = data.build_dataset()
    sub = data.dev_subset(ds, n)
    assert list(sub.indices) == cap["index"].tolist(), \
        "the development draw no longer matches the capture; rebuild the capture"
    xs = [x for x, _ in data.make_loader(sub)]
    x_all = torch.cat(xs).to(device)
    print(f"\n{args.capture}: {n} images, {len(locs)} locations, weights {weights}")
    print(f"rules: max, monotone_null, {', '.join(RULES)}  "
          f"(blur {BLUR_SIZE}x{BLUR_SIZE} sigma {BLUR_SIGMA}, fixed)\n")

    covers = {loc: metrics.centred_coverage(*(int(v) for v in loc.split(",")),
                                            size, m["window"]).to(device) for loc in locs}

    # Reconstruct once per (location, weight); every rule reads the same images.
    img = {"clean": x_all}
    score = {"clean": cap["clean_scores"].to(device)}
    for loc in locs:
        top, left = (int(v) for v in loc.split(","))
        for w in weights:
            adv = x_all.clone()
            adv[:, :, top:top + size, left:left + size] = (
                cap["by"][loc][str(float(w))]["patch"].to(device))
            img[(loc, w)] = adv
            score[(loc, w)] = cap["by"][loc][str(float(w))]["scores_own"].to(device)
    resid = {k: blur_residual(v) for k, v in img.items()}
    base_clean = score["clean"].argmax(1)

    fns = {"max": lambda k: score[k],
           "monotone_null": lambda k: 3.0 * score[k] + 7.0}
    fns.update({name: (lambda k, f=f: f(img[k], resid[k])) for name, f in RULES.items()})

    rec: dict[str, dict] = {}
    for name, fn in fns.items():
        row: dict = {"by_location": {}}
        for w in weights:
            hit = 0
            for loc in locs:
                chosen = fn((loc, w)).argmax(1)
                c = int(covers[loc][chosen].sum())
                row["by_location"].setdefault(loc, {})[f"w{w}"] = 100.0 * c / n
                hit += c
            row[f"cover_w{w}"] = 100.0 * hit / (n * len(locs))
        clean_sel = fn("clean").argmax(1)
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
    if not passed:
        print("stop here: this was a predefined screen, and expanding it after seeing the "
              "numbers is what the gate exists to prevent. Reassess the design instead.")

    results.save(args.out, dict(
        purpose="offline screen of input-space statistics as RSA's selection rule; "
                "mask geometry unchanged, only the ranking differs",
        source=f"{args.capture}.pt plus images reconstructed from the saved patches",
        caveat="coverage is geometric and measured on attacks optimised against the "
               "argmax, so it is a transfer quantity; it is not recovered accuracy and "
               "no rule here has been attacked adaptively",
        definitions=dict(
            tv="mean absolute horizontal and vertical pixel difference within the tile",
            blur=f"mean absolute difference from a fixed {BLUR_SIZE}x{BLUR_SIZE} Gaussian "
                 f"blur, sigma {BLUR_SIGMA}, computed on the whole image then averaged "
                 f"into tiles",
            colour="tile mean colour's distance from the image mean colour",
            scale="the model's normalised input scale, identical across rules and images"),
        gate=dict(cover_w0_min=GATE_W0, cover_w05_min=GATE_W05,
                  note="discretionary resource-allocation rule fixed before the run; "
                       "passing it justifies local classification replay only, and only a "
                       "useful classification gain at an acceptable clean cost earns a pilot"),
        settings={k: m[k] for k in ("size", "steps", "weights", "window", "window_rule",
                                    "score_frame", "freeze_mask_after")},
        rules=rec, passed=passed), n_eval_images=n, attack_steps=m["steps"])


if __name__ == "__main__":
    main()
