"""Measurements 2 and 3 at `FULL`, driven by Patch-Fool.

Both measurements previously rested on a 50-step patch-PGD at `DEV`, which was the
weakest of their three caveats: an attack indifferent to attention cannot say much
about a defense that reads attention. Patch-Fool is attention-aware, so this is the
re-run those numbers were waiting on.

    Measurement 2  How much of the value-space deviation RSA charges for actually
                   reduces the margin. `aligned_fraction` = ||delta_par||^2/||delta||^2
                   against a random-direction baseline of 1/head_dim. A low number is
                   slack: RSA is paying for deviation that does no first-order harm.

    Measurement 3  Whether the mechanism Liu et al. describe is present - adversarial
                   value vectors ~50% *smaller* in norm than benign ones - and whether
                   RSA's argmax window lands on the patch, against its false-positive
                   rate on clean images.

One difference from the notebook versions, and it runs through everything below:
**Patch-Fool chooses a different token per image**, so the patch tokens are a per-image
gather rather than one shared index list. The patch is also a single grid-aligned 16 px
token cell rather than the notebook's 32 px square at (96,96), which is a smaller and
better-placed patch, not a comparable one.

Usage::

    python scripts/mechanism_full.py                    # 256 images, ~15 min
    python scripts/mechanism_full.py --n-images 64      # quicker
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import islice
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fyp import attacks, config, data, hooks, metrics, models, results  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-images", type=int, default=config.SCALE.n_mechanism_images)
    p.add_argument("--steps", type=int, default=250, help="Patch-Fool iterations")
    p.add_argument("--layer", type=int, default=6, help="layer for measurement 3")
    p.add_argument("--score-frame", default=metrics.DEFAULT_SCORE_FRAME,
                   choices=metrics.SCORE_FRAMES,
                   help="which mean the deviation is centred on; must match the frame "
                        "RSA's score is ranked in for measurement 2 to be about RSA")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def gather_tokens(t: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    """Pick one token per image out of `(B, H, N, D)` -> `(B, H, 1, D)`.

    `tok` is `(B,)` indices into the full 197-token sequence, CLS included.
    """
    B, H, _, D = t.shape
    return t.gather(2, tok.view(B, 1, 1, 1).expand(B, H, 1, D))


def main():
    args = parse_args()
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    ds, wnids, class_indices = data.build_dataset()
    clf, info = models.load_model(class_indices=class_indices, device=device)
    depth, heads, head_dim = info["depth"], info["num_heads"], info["head_dim"]

    n_batches = max(args.n_images // config.SCALE.batch_size, 1)
    n_total = n_batches * config.SCALE.batch_size
    window = metrics.rsa_window_size(config.PATCH_SIZE)
    print(f"\n{n_batches} batches, {n_total} images, Patch-Fool at {args.steps} steps, "
          f"16 px token patch, RSA window {window}x{window}\n")

    # Accumulators. Sums and counts, so the batch mean is not averaged twice.
    aligned_sum = [0.0] * depth
    norm_sum, par_sum, perp_sum = [0.0] * depth, [0.0] * depth, [0.0] * depth
    aligned_n = 0
    rand_sum, rand_n = 0.0, 0
    vn_clean = vn_adv = vn_other = 0.0
    vn_n = 0
    hits_adv = hits_clean = 0
    n_clean_correct = n_robust = n_seen = 0
    tokens_seen: list[int] = []

    t_start = time.time()
    loader = data.make_loader(data.eval_subset(ds, n_total))

    for bi, (x, y) in enumerate(islice(loader, n_batches)):
        x, y = x.to(device), y.to(device)
        B = x.shape[0]

        clean = attacks.accuracy(clf, x, y)
        x_adv, sel = attacks.patch_fool(clf, x, y, steps=args.steps)
        robust = attacks.robust_correct(clf, x_adv, y, clean)

        n_seen += B
        n_clean_correct += int(clean.sum())
        n_robust += int(robust.sum())
        tok = sel[:, 0] + 1                       # into the 197-token sequence
        tokens_seen += sel[:, 0].tolist()

        # -- Measurement 2: is the deviation aimed where it does harm? -------
        for L in range(depth):
            with hooks.AttentionCapture(clf, layers=[L], store=("v",), detach=False) as cap:
                out = clf(x_adv)
                metrics.margin(out, y).sum().backward()
            rec = cap[L]
            v = rec.v.detach()
            u = -rec.grad("v")                    # not rec.v.grad; see LayerAttention.grad
            # The same mean the anomaly score ranks against, not the per-head mean.
            # `delta` here stands in for "what RSA penalises", so it has to be centred
            # in the score's frame; the two differ by the per-head offset and change
            # both the norm and the direction. See `metrics.score_frame_mean`.
            mu = metrics.score_frame_mean(v, frame=args.score_frame)

            d = metrics.direction_decomposition(gather_tokens(v, tok) - mu,
                                                gather_tokens(u, tok))
            aligned_sum[L] += float(d["aligned_fraction"].sum())
            norm_sum[L] += float(d["norm"].sum())
            par_sum[L] += float(d["parallel_norm"].sum())
            perp_sum[L] += float(d["perp_norm"].sum())
        aligned_n += B * heads

        # -- Measurement 3 and the random-direction control ------------------
        with hooks.AttentionCapture(clf, layers=[args.layer], store=("v",)) as cap:
            with torch.no_grad():
                clf(x_adv)
            v_ad = cap[args.layer].v
        with hooks.AttentionCapture(clf, layers=[args.layer], store=("v",)) as cap:
            with torch.no_grad():
                clf(x)
            v_cl = cap[args.layer].v

        mu_ad = metrics.score_frame_mean(v_ad, frame=args.score_frame)
        delta = gather_tokens(v_ad, tok) - mu_ad
        rand = metrics.direction_decomposition(delta, torch.randn_like(delta))
        rand_sum += float(rand["aligned_fraction"].sum())
        rand_n += B * heads

        # Value norms: the attacked token, the same token clean, and everything else.
        onehot = torch.zeros(B, 1, config.N_IMAGE_TOKENS, device=device)
        onehot.scatter_(2, (tok - 1).view(B, 1, 1), 1.0)
        na = v_ad[:, :, 1:, :].norm(dim=-1)
        nc = v_cl[:, :, 1:, :].norm(dim=-1)
        n_patch_el = onehot.expand_as(na).sum()
        vn_adv += float((na * onehot).sum() / n_patch_el) * B
        vn_clean += float((nc * onehot).sum() / n_patch_el) * B
        vn_other += float((na * (1 - onehot)).sum() / (1 - onehot).expand_as(na).sum()) * B
        vn_n += B

        # Does RSA's argmax window land on the token Patch-Fool attacked?
        for v_here, name in ((v_ad, "adv"), (v_cl, "clean")):
            am = metrics.rsa_argmax_window(
                metrics.rsa_window_scores(metrics.rsa_token_scores(v_here), window))
            pr, pc = (tok - 1) // config.GRID, (tok - 1) % config.GRID
            hit = ((am[:, 0] <= pr) & (pr <= am[:, 0] + window - 1) &
                   (am[:, 1] <= pc) & (pc <= am[:, 1] + window - 1))
            if name == "adv":
                hits_adv += int(hit.sum())
            else:
                hits_clean += int(hit.sum())

        print(f"  batch {bi + 1}/{n_batches}  clean {100.0 * clean.float().mean():5.2f}%  "
              f"robust {100.0 * robust.float().mean():5.2f}%  "
              f"{(time.time() - t_start) / 60:.1f} min")

    # -- report ------------------------------------------------------------
    aligned = [s / aligned_n for s in aligned_sum]
    baseline = 1.0 / head_dim
    mean_aligned = sum(aligned) / len(aligned)
    rand_frac = rand_sum / rand_n

    print(f"\nattack: clean {100.0 * n_clean_correct / n_seen:.2f}%, "
          f"robust {100.0 * n_robust / n_seen:.2f}%, "
          f"{len(set(tokens_seen))} distinct tokens over {n_seen} images")

    print(f"\nMeasurement 2 - value-space direction, {n_seen} images")
    print(f"{'layer':>5} {'||delta||':>10} {'||par||':>9} {'||perp||':>10} {'aligned frac':>13}")
    for L in range(depth):
        print(f"{L:5d} {norm_sum[L] / aligned_n:10.3f} {par_sum[L] / aligned_n:9.3f} "
              f"{perp_sum[L] / aligned_n:10.3f} {aligned[L]:13.4f}")
    print(f"\n  mean aligned fraction {mean_aligned:.4f}  vs random baseline "
          f"{baseline:.4f}  ->  {mean_aligned / baseline:.1f}x chance")
    ok = abs(rand_frac - baseline) < 0.01
    print(f"  random-direction control {rand_frac:.4f} against {baseline:.4f}  "
          f"{'OK' if ok else 'CHECK THIS'}")
    print(f"  {100.0 * (1 - mean_aligned):.1f}% of the squared deviation energy RSA "
          f"ranks on ({args.score_frame} frame) lies orthogonal to the local margin "
          f"gradient at this layer")
    print("  that is a first-order, single-layer, 16 px grid-aligned measurement; it "
          "does not say the deviation is harmless, and does not transfer to RSA's "
          "10-50 px arbitrarily-placed squares")

    print(f"\nMeasurement 3 - layer {args.layer}, {n_seen} images")
    c, a, o = vn_clean / vn_n, vn_adv / vn_n, vn_other / vn_n
    print(f"  mean ||v|| at the attacked token, clean    {c:.3f}")
    print(f"  mean ||v|| at the attacked token, attacked {a:.3f}   ({100 * (a / c - 1):+.1f}% vs clean)")
    print(f"  mean ||v|| at every other token, attacked  {o:.3f}   ({100 * (a / o - 1):+.1f}% vs the rest)")
    print("  Liu et al. report adversarial value vectors ~50% SMALLER than benign")
    print(f"  RSA window hits the patch on {100.0 * hits_adv / n_seen:.1f}% of attacked "
          f"images, against {100.0 * hits_clean / n_seen:.1f}% on clean")

    minutes = round((time.time() - t_start) / 60, 2)
    print(f"\n{minutes} min")

    payload = {
        "attack": "patch_fool",
        "steps": args.steps,
        "patch_px": config.PATCH_SIZE,
        "patch_is_grid_aligned_token": True,
        "n_images": n_seen,
        "clean_acc": 100.0 * n_clean_correct / n_seen,
        "robust_acc": 100.0 * n_robust / n_seen,
        "distinct_tokens": len(set(tokens_seen)),
        "measurement_2": {
            "score_frame": args.score_frame,
            "aligned_fraction_by_layer": aligned,
            "mean_aligned_fraction": mean_aligned,
            "random_baseline": baseline,
            "random_control": rand_frac,
            "random_control_ok": ok,
            "times_chance": mean_aligned / baseline,
            "mean_norm_by_layer": [s / aligned_n for s in norm_sum],
            "mean_parallel_by_layer": [s / aligned_n for s in par_sum],
            "mean_perp_by_layer": [s / aligned_n for s in perp_sum],
        },
        "measurement_3": {
            "layer": args.layer,
            "window": window,
            "v_norm_patch_clean": c,
            "v_norm_patch_attacked": a,
            "v_norm_other_attacked": o,
            "pct_vs_clean": 100 * (a / c - 1),
            "pct_vs_other": 100 * (a / o - 1),
            "rsa_hit_rate_attacked": 100.0 * hits_adv / n_seen,
            "rsa_hit_rate_clean": 100.0 * hits_clean / n_seen,
        },
        "minutes": minutes,
        "model_info": info,
        "invocation": " ".join(sys.argv),
    }
    if not args.no_save:
        results.save("m6_mechanism_patch_fool", payload,
                     n_eval_images=n_seen, attack_steps=args.steps)


if __name__ == "__main__":
    main()
