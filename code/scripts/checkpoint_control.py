"""Compare checkpoint clean accuracy and per-layer patch-selection geometry."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from fyp import attacks, config, data, metrics, models, results, rsa  # noqa: E402

SIZE = 20
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


# These rules do not rank one square sliding-window candidate set.
NO_WINDOW_RANK = ("multi", "token", "topk", "centred")


def centred_contains(scores, side, grid, r0, r1, c0, c1):
    """Whether the actual centred rule covers inclusive token bounds r0:r1, c0:c1."""
    br, bc, _, _ = rsa.RSAAttention._window_corner(None, scores, side, grid, "centred")
    return (br <= r0) & (br + side - 1 >= r1) & (bc <= c0) & (bc + side - 1 >= c1)


def geometry(clf, handle, adv, y, clean_ok, top, left, window, size=SIZE,
             frame=metrics.DEFAULT_SCORE_FRAME, padded=False, pool="mean",
             window_rank=True, centred_sides=()):
    """Per-image ranks, peak membership and mask coverage; one (B,) tensor per layer.

    frame, padded and pool must match the defense. Disable window_rank for
    rules in NO_WINDOW_RANK. Rank counts strictly higher scores, so rank 1
    does not imply peak membership under ties.

    centred_sides adds counterfactual reselection coverage; after a frozen block
    this can differ from the actual mask. Callers pool batches before taking medians.
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
    patch_tokens = metrics.patch_token_indices(top, left, size)
    patch_cells = [t - 1 for t in patch_tokens]
    r0, c0 = top // config.PATCH_SIZE, left // config.PATCH_SIZE
    r1 = (top + size - 1) // config.PATCH_SIZE
    c1 = (left + size - 1) // config.PATCH_SIZE
    covering = [
        (r, c)
        for r in range(grid - window + 1)
        for c in range(grid - window + 1)
        if r <= r0 and r + window - 1 >= r1 and c <= c0 and c + window - 1 >= c1
    ]

    token_rank, win_rank, full_mask, overlap, ties = [], [], [], [], []
    argmax_hit, peak_ties = [], []
    centred_hit = {s: [] for s in centred_sides}
    for i, m in zip(handle.layers, handle.modules):
        xin = captured[i]
        b, n, _ = xin.shape
        with torch.no_grad():
            qkv = m.qkv(xin).reshape(b, n, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
            tok = metrics.rsa_token_scores(qkv[2], frame=frame)
            win = (metrics.rsa_window_scores(tok, window, padded=padded, pool=pool)
                   if window_rank else None)
        best_token = tok[:, patch_cells].amax(1)
        token_rank.append((tok > best_token[:, None]).sum(1).float() + 1)
        cells = torch.as_tensor(patch_cells, device=tok.device)
        argmax_hit.append((tok.argmax(1)[:, None] == cells[None, :]).any(1))
        peak_ties.append((tok == tok.amax(1, keepdim=True)).sum(1).float())
        for s in centred_sides:
            centred_hit[s].append(centred_contains(tok, s, grid, r0, r1, c0, c1))
        if window_rank:
            flat = win.flatten(1)
            best_window = torch.stack([win[:, r, c] for (r, c) in covering], 1).amax(1)
            win_rank.append((flat > best_window[:, None]).sum(1).float() + 1)
            ties.append(metrics.rsa_tied_windows(win).float())
        hit = m.last_mask[:, patch_tokens]
        full_mask.append(hit.all(1))
        overlap.append(hit.float().mean(1))

    out = {
        "robust_acc_pct": robust,
        "token_rank_by_layer": token_rank,
        "argmax_in_patch_by_layer": argmax_hit,
        "tied_peak_tokens_by_layer": peak_ties,
        "full_mask_by_layer": full_mask,
        "overlap_by_layer": overlap,
        "n_patch_tokens": len(patch_tokens),
    }
    if window_rank:
        out["window_rank_by_layer"] = win_rank
        out["tied_windows_by_layer"] = ties
    for s in centred_sides:
        out[f"centred_contain_{s}_by_layer"] = centred_hit[s]
    return out


def main():
    args = parse_args()
    tag = args.tag or args.model
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

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

    # The released checkpoint requires its original class order; ours slices the head.
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
            g = {
                "robust_acc_pct": g["robust_acc_pct"],
                "median_token_rank_by_layer": [t.median().item() for t in g["token_rank_by_layer"]],
                "median_window_rank_by_layer": [t.median().item() for t in g["window_rank_by_layer"]],
                "full_mask_pct_by_layer": [100 * t.float().mean().item() for t in g["full_mask_by_layer"]],
                "mean_overlap_pct_by_layer": [100 * t.mean().item() for t in g["overlap_by_layer"]],
                "n_patch_tokens": g["n_patch_tokens"],
            }
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
