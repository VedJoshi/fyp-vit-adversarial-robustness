"""Probe: is leak-free masking invariant to patch pixels, and how many images certify?

Part 1 holds the exact-extent mask across every block and compares logits under
different patch contents, once per renormalisation. Part 2 applies every pair of
windows from a covering set and counts the images correct under all pairs, which
is PatchCleanser's two-mask condition.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fyp import attacks, config, data, models, results, rsa  # noqa: E402

G, PS, IMG = config.GRID, config.PATCH_SIZE, config.IMAGE_SIZE


def window_positions(w: int, s: int) -> list[int]:
    """1-D window origins at stride s, plus the flush-right origin."""
    return sorted(set(range(0, G - w + 1, s)) | {G - w})


def assert_covering(pos: list[int], w: int, size: int) -> None:
    """Every pixel offset of a size-px patch must fall inside one window, per axis."""
    for t in range(IMG - size + 1):
        lo, hi = t // PS, (t + size - 1) // PS
        assert any(p <= lo and hi < p + w for p in pos), \
            f"a {size}px patch at offset {t} is not inside any {w}-token window of {pos}"


def window_masks(w: int, s: int, size: int) -> torch.Tensor:
    """(M, G, G) bool, one covering window each."""
    pos = window_positions(w, s)
    assert_covering(pos, w, size)
    m = torch.zeros(len(pos) ** 2, G, G, dtype=torch.bool)
    for i, (r, c) in enumerate((r, c) for r in pos for c in pos):
        m[i, r:r + w, c:c + w] = True
    return m


def double_masking(pred: torch.Tensor) -> tuple[int, int]:
    """PatchCleanser inference from the (M, M) pair predictions; (label, forward passes)."""
    one = pred.diagonal()
    maj = int(one.mode().values)
    passes = len(one)
    for i in (one != maj).nonzero().flatten().tolist():
        passes += len(one)
        if bool((pred[i] == one[i]).all()):
            return int(one[i]), passes
    return maj, passes


@torch.no_grad()
def invariance(clf, x, size, locations, device) -> dict:
    out = {}
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = [torch.rand(x.shape, generator=gen).to(device) for _ in range(2)]
    contents = {"zeros": torch.zeros_like(x), "ones": torch.ones_like(x),
                "noise0": noise[0], "noise1": noise[1]}
    for renorm in rsa.RENORMALISATIONS:
        cfg = rsa.RSAConfig(patch_px=size, window=1, renormalise=renorm)
        rows = {}
        with rsa.enable(clf, cfg) as handle:
            for top, left in locations:
                handle.set_oracle(top, left, size, mode="extent", device=device)
                pm = attacks.patch_mask(x.shape, top, left, size, device)
                ref = clf(x)
                diff, flips = 0.0, torch.zeros(len(x), dtype=torch.bool, device=device)
                for delta in contents.values():
                    got = clf(attacks.apply_patch(x, delta, pm))
                    diff = max(diff, float((got - ref).abs().max()))
                    flips |= got.argmax(1) != ref.argmax(1)
                rows[f"{top},{left}"] = {"max_abs_logit_diff": diff,
                                         "bit_identical": diff == 0.0,
                                         "prediction_changes": int(flips.sum())}
        worst = max(r["max_abs_logit_diff"] for r in rows.values())
        changed = sum(r["prediction_changes"] for r in rows.values())
        print(f"  renormalise={renorm:<8} max |logit diff| {worst:.3e}   "
              f"prediction changes {changed} of {len(x) * len(locations)}")
        out[renorm] = rows
    return out


@torch.no_grad()
def pair_predictions(clf, handle, x1, masks, fill, mode, chunk) -> torch.Tensor:
    """(M, M) predictions for one image under every union of two masks."""
    M = len(masks)
    iu = torch.triu_indices(M, M)
    union = masks[iu[0]] | masks[iu[1]]                              # (P, G, G)
    preds = []
    for s in range(0, len(union), chunk):
        u = union[s:s + chunk]
        xb = x1.expand(len(u), -1, -1, -1)
        if mode == "attn":
            cls_col = torch.zeros(len(u), 1, dtype=torch.bool, device=u.device)
            handle.set_mask(torch.cat([cls_col, u.flatten(1)], dim=1))
        else:
            px = u.repeat_interleave(PS, 1).repeat_interleave(PS, 2)[:, None]
            xb = torch.where(px, fill, xb)
        preds.append(clf(xb).argmax(1))
    flat = torch.cat(preds)
    pred = torch.empty(M, M, dtype=torch.long, device=flat.device)
    pred[iu[0], iu[1]] = flat
    pred[iu[1], iu[0]] = flat
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=config.SCALE.n_eval_images)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--sets", default="4,2 5,3 6,4 8,6",
                    help="space-separated window,stride in tokens")
    ap.add_argument("--modes", default="attn pixel",
                    help="attn masks keys with -inf in every block; pixel fills the "
                         "window with the dataset mean colour")
    ap.add_argument("--invariance-images", type=int, default=64)
    ap.add_argument("--locations", default="60,60 0,0 100,20 13,29 93,141 204,204")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default="probe_certify")
    args = ap.parse_args()

    config.seed_everything()
    device = config.get_device()
    print(config.describe())
    ds, _, class_indices = data.build_dataset()
    clf, info = models.load_model(class_indices=class_indices, device=device)
    sub = data.eval_subset(ds, args.n_images)
    xs, ys = zip(*[(x.to(device), y.to(device)) for x, y in data.make_loader(sub)])
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    n = len(x_all)
    with torch.no_grad():
        clean_ok = torch.cat([clf(x_all[i:i + 256]).argmax(1) == y_all[i:i + 256]
                              for i in range(0, n, 256)])
    print(f"\n{n} images, undefended clean accuracy {100 * clean_ok.float().mean():.2f}%")

    payload = dict(purpose="leak-free invariance test and two-mask certified count",
                   size=args.size, index=list(sub.indices),
                   indexing="index[i] is the dataset index of position i",
                   undefended_clean_correct=clean_ok.nonzero().flatten().tolist())

    if args.invariance_images:
        locations = [tuple(int(v) for v in loc.split(",")) for loc in args.locations.split()]
        print(f"\ninvariance to patch contents under an exact-extent mask, "
              f"{args.invariance_images} images, {len(locations)} locations")
        payload["invariance"] = invariance(clf, x_all[:args.invariance_images], args.size,
                                           locations, device)

    fill = torch.tensor(info["mean"], device=device).view(1, 3, 1, 1)
    cfg = rsa.RSAConfig(patch_px=args.size, window=1, renormalise="softmax")
    payload["sets"] = {}
    for spec in args.sets.split():
        w, s = (int(v) for v in spec.split(","))
        masks = window_masks(w, s, args.size).to(device)
        M = len(masks)
        for mode in args.modes.split():
            t0 = time.time()
            certified, one_ok, infer_ok, passes = [], [], [], 0
            with rsa.enable(clf, cfg) as handle:
                if mode != "attn":
                    handle.remove()
                for i in range(n):
                    pred = pair_predictions(clf, handle, x_all[i:i + 1], masks, fill,
                                            mode, args.chunk)
                    y = int(y_all[i])
                    if bool((pred == y).all()):
                        certified.append(i)
                    if bool((pred.diagonal() == y).all()):
                        one_ok.append(i)
                    label, p = double_masking(pred)
                    passes += p
                    if label == y:
                        infer_ok.append(i)
            rec = dict(window=w, stride=s, n_masks=M, n_pairs=M * (M + 1) // 2,
                       masked_fraction=w * w / G ** 2,
                       certified=certified, one_mask_correct=one_ok,
                       defended_clean_correct=infer_ok,
                       mean_inference_passes=passes / n,
                       seconds=round(time.time() - t0, 1))
            payload["sets"][f"{w},{s}/{mode}"] = rec
            print(f"  window {w} stride {s} {mode:<5} {M:>3} masks  "
                  f"certified {100 * len(certified) / n:6.2f}%  "
                  f"defended clean {100 * len(infer_ok) / n:6.2f}%  "
                  f"all one-mask correct {100 * len(one_ok) / n:6.2f}%  "
                  f"passes/image {passes / n:7.1f}  {rec['seconds']:.0f}s")
            results.save(args.out, payload, n_eval_images=n)


def self_check():
    for w, s in ((3, 1), (4, 2), (5, 3), (6, 4), (8, 6)):
        assert window_masks(w, s, 20).any(0).all()
    try:
        window_masks(3, 2, 20)
    except AssertionError:
        pass
    else:
        raise AssertionError("a 3-token window at stride 2 cannot cover a 20px patch")
    # Unanimous first round; a minority mask confirmed by its row; one that is not.
    assert double_masking(torch.tensor([[1, 1], [1, 1]])) == (1, 2)
    assert double_masking(torch.tensor([[1, 2, 1], [2, 2, 2], [1, 2, 1]])) == (2, 6)
    assert double_masking(torch.tensor([[1, 1, 1], [1, 2, 1], [1, 1, 1]])) == (1, 6)


if __name__ == "__main__":
    self_check()
    main()
