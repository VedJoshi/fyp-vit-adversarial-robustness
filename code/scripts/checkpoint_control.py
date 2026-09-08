"""Step 6h: does RSA's own checkpoint change the defense's value-space geometry?

    python code/scripts/checkpoint_control.py --model ours
    python code/scripts/checkpoint_control.py --model rsa --checkpoint data/vitsmall_100.pt

Two stages, both cheap, run under identical code on either model so the outputs are
directly comparable.

**Stage 1, the clean-accuracy gate.** Loads the model, reports state-dict key
agreement, and measures clean accuracy over the evaluation sample. RSA report 93.16%
undefended. A model that misses that by more than a few points is loaded wrongly, and
nothing after it means anything.

**Stage 2, the value-space geometry.** RSA ranks tokens by `a_i = ||v_i - mu||_2` and
masks the argmax window, so the checkpoint can only matter through that geometry. At
20px, at one location where the patch owns a token cell outright and one where it
straddles, against a 100-step PatchAutoPGD patch, this records per layer: the median
rank of the best patch token among the 196 image tokens, the median rank of the best
patch-containing window among all windows, and the rate at which the selected window
fully contains the patch.

If their checkpoint gives the same ranking behaviour as this project's, the checkpoint
cannot explain the Gate A gap, and the five-size sweep under their weights is not worth
its GPU-hours. If it differs, the sweep is the next job.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from fyp import attacks, config, data, metrics, models, results, rsa  # noqa: E402

SIZE = 20
#: A 20px patch owns a token cell outright at offset 0 and straddles two at offset 20
#: (§5.4). Both are locations RSA's own stride-20 grid visits.
LOCATIONS = [(0, 0), (20, 20)]
RSA_CLEAN_REFERENCE = 93.16


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=("ours", "rsa"), default="ours",
                   help="'ours' is timm DeiT-S with a sliced head; 'rsa' is their checkpoint")
    p.add_argument("--checkpoint", help="path to vitsmall_100.pt; required for --model rsa")
    p.add_argument("--n-images", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--tag", help="names the output record; defaults to the model")
    p.add_argument("--allow-key-mismatch", action="store_true",
                   help="load their checkpoint even if keys do not match, and record which")
    return p.parse_args()


def geometry(clf, handle, adv, y, clean_ok, top, left, window):
    """Returns per-layer token rank, window rank and full-mask rate for one batch.

    Ranks are medians over images, 1 being the most anomalous. The patch-containing
    window is the best over every window that fully covers the patch's cell extent.
    """
    captured = {}
    hooks = [
        m.register_forward_pre_hook(
            lambda mod, inp, i=i: captured.__setitem__(i, inp[0].detach())
        )
        for i, m in zip(handle.layers, handle.modules)
    ]
    with torch.no_grad():
        logits = clf(adv)
    for h in hooks:
        h.remove()

    ok = logits.argmax(1) == y
    robust = 100 * (ok & clean_ok).float().mean().item()

    grid = config.GRID
    patch_tokens = metrics.patch_token_indices(top, left, SIZE)
    patch_cells = [t - 1 for t in patch_tokens]
    r0, c0 = top // config.PATCH_SIZE, left // config.PATCH_SIZE
    r1 = (top + SIZE - 1) // config.PATCH_SIZE
    c1 = (left + SIZE - 1) // config.PATCH_SIZE
    covering = [
        (r, c)
        for r in range(grid - window + 1)
        for c in range(grid - window + 1)
        if r <= r0 and r + window - 1 >= r1 and c <= c0 and c + window - 1 >= c1
    ]

    token_rank, window_rank, full_mask = [], [], []
    for i, m in zip(handle.layers, handle.modules):
        xin = captured[i]
        b, n, _ = xin.shape
        with torch.no_grad():
            qkv = m.qkv(xin).reshape(b, n, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
            tok = metrics.rsa_token_scores(qkv[2])
            win = metrics.rsa_window_scores(tok, window)
        best_token = tok[:, patch_cells].amax(1)
        token_rank.append((tok > best_token[:, None]).sum(1).float().median().item() + 1)
        best_window = torch.stack([win[:, r, c] for (r, c) in covering], 1).amax(1)
        window_rank.append((win.flatten(1) > best_window[:, None]).sum(1).float().median().item() + 1)
        full_mask.append(100 * m.last_mask[:, patch_tokens].all(1).float().mean().item())

    return {
        "robust_acc_pct": robust,
        "median_token_rank_by_layer": token_rank,
        "median_window_rank_by_layer": window_rank,
        "full_mask_pct_by_layer": full_mask,
    }


def main():
    args = parse_args()
    tag = args.tag or args.model
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    # ---- stage 1: load and gate ------------------------------------------------
    if args.model == "rsa":
        if not args.checkpoint:
            raise SystemExit("--model rsa requires --checkpoint")
        wnids_expected = data.rsa_class_wnids()
        clf, info = models.load_rsa_checkpoint(
            args.checkpoint, device=device, strict=not args.allow_key_mismatch
        )
        print(f"\nloaded {args.checkpoint}")
        print(f"  missing keys   : {len(info['missing_keys'])} {info['missing_keys'][:4]}")
        print(f"  unexpected keys: {len(info['unexpected_keys'])} {info['unexpected_keys'][:4]}")
    else:
        wnids_expected = data.imagenet100_wnids()
        clf, info = None, None

    # Fatal for their checkpoint, a warning for ours. Their head is ordered by their
    # seed-0 wnid list, so a different directory silently scores every image against the
    # wrong class while clean accuracy stays plausible and the 5-point gate still passes.
    # Ours is head-sliced from the directory itself, so a mismatch there is a comparison
    # problem rather than a labelling one.
    ds, wnids, class_indices = data.build_dataset(
        expect_wnids=wnids_expected if args.model == "rsa" else None)
    if wnids != sorted(wnids_expected):
        print(f"\n  WARNING: the data on disk is not the class set this model expects.")
        print(f"  on disk : {wnids[:3]} ... ({len(wnids)} classes)")
        print(f"  expected: {sorted(wnids_expected)[:3]} ... ({len(wnids_expected)} classes)")
        print(f"  overlap : {len(set(wnids) & set(wnids_expected))} of {len(wnids_expected)}")
        print("  Rebuild with scripts/build_imagenet100.py before trusting anything below.")

    if args.model == "ours":
        clf, info = models.load_model(class_indices=class_indices, device=device)

    # Drawn at `--n-images`, not sliced from a 512-draw: sorted indices make a prefix
    # the lowest order statistics of the sample. See `data.eval_subset`.
    sub = data.eval_subset(ds, args.n_images)
    loader = data.make_loader(sub, batch_size=args.n_images)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    clean_ok = attacks.accuracy(clf, x, y)
    clean_pct = 100 * clean_ok.float().mean().item()
    print(f"\nstage 1: clean accuracy {clean_pct:.2f}% on {len(y)} images "
          f"(RSA report {RSA_CLEAN_REFERENCE} undefended)")
    if abs(clean_pct - RSA_CLEAN_REFERENCE) > 5:
        print("  GATE FAILED by more than 5 points. Stop and check the load path.")

    # ---- stage 2: value-space geometry ----------------------------------------
    record = {
        "purpose": "step 6h: whether RSA's checkpoint changes the value-space geometry",
        "tag": tag,
        "settings": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "size": SIZE,
            "locations": LOCATIONS,
            "n_images": len(y),
            "steps": args.steps,
            "window_rule": rsa.DEFAULT_WINDOW_RULE,
            "score_frame": rsa.DEFAULT_SCORE_FRAME,
            "renormalise": rsa.DEFAULT_RENORMALISATION,
        },
        "clean_acc_pct": clean_pct,
        "clean_acc_reference": RSA_CLEAN_REFERENCE,
        "wnids_on_disk_match_expected": wnids == sorted(wnids_expected),
        "per_location": {},
        "model_info": info,
    }

    cfg = rsa.RSAConfig.for_patch(SIZE)
    for top, left in LOCATIONS:
        alignment = "owns a cell" if left % config.PATCH_SIZE == 0 else "straddles"
        print(f"\nstage 2: ({top},{left}), {alignment}, window {cfg.window}x{cfg.window}")
        with rsa.enable(clf, cfg) as handle:
            t0 = time.time()
            adv = attacks.patch_autopgd(clf, x, y, top=top, left=left,
                                        size=SIZE, steps=args.steps)
            g = geometry(clf, handle, adv, y, clean_ok, top, left, cfg.window)
        print(f"  ({time.time() - t0:.0f}s) robust {g['robust_acc_pct']:.2f}%")
        print("  token rank  : " + " ".join(f"{r:4.0f}" for r in g["median_token_rank_by_layer"]))
        print("  window rank : " + " ".join(f"{r:4.0f}" for r in g["median_window_rank_by_layer"]))
        print("  full mask % : " + " ".join(f"{f:4.0f}" for f in g["full_mask_pct_by_layer"]))
        record["per_location"][f"{top},{left}"] = dict(g, alignment=alignment)

    results.save(f"checkpoint_control_{tag}", record,
                 n_eval_images=int(len(y)), attack_steps=args.steps)
    print("\nCompare the two runs' window-rank rows. If they agree, the checkpoint")
    print("cannot explain the Gate A gap and the five-size sweep is not worth running.")


if __name__ == "__main__":
    main()
