"""Gate A - reproduce the `ViT-small + RSA` row of RSA's Table 1.

For each patch size, RSA is configured for that size (their threat model takes it as
known), and `attacks.patch_autopgd` is driven over `attacks.patch_locations` against
the defended model. An image counts as robust only if it survives every location.

The location loop is written out here rather than calling
`attacks.worst_location_attack`, which carries the same rule -- an image is robust only
if no location flipped it -- but holds its state in tensors and returns only at the end.
This version records each location as it finishes, so a sweep measured in hours resumes
after an interruption instead of restarting.

Usage::

    python scripts/gate_a.py --loc-stride 4 --n-images 128 --tag pilot   # ~4 h
    python scripts/gate_a.py --tag full                                  # all five sizes

The default is RSA's five patch sizes: 523 location-sweeps of 16 batches each. Measured
with RSA active at 20.7 s per 100-step attack batch, that is 48.3 hours -- 11.2 h at 10px
and 20px, 9.2 h at 30px and 40px, 7.5 h at 50px. Record: `results/m5_rsa_sanity.json`.

`--loc-stride` keeps every n-th location, which weakens the worst-case search and so
raises the reported robust accuracy: a pilot below the target is decisive, a pilot at
or above it is not. `--n-images` subsamples the front of the seeded evaluation sample.

Progress is appended to `results/gate_a_<tag>.progress.jsonl` after every location and
replayed on restart, so an interrupted sweep resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fyp import attacks, config, data, models, results, rsa

#: RSA Table 1, row `ViT-small + RSA`, and row `ViT-small` for the undefended model.
TABLE1_RSA = {10: 83.20, 20: 72.46, 30: 68.55, 40: 56.25, 50: 43.16}
TABLE1_UNDEFENDED = {10: 45.70, 20: 0.00, 30: 0.00, 40: 0.00, 50: 0.00}
#: RSA Table 2, row `ViT-small + RSA`: clean accuracy by assumed patch size.
TABLE2_RSA = {0: 93.16, 10: 92.58, 20: 90.43, 30: 90.43, 40: 88.67, 50: 83.98}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="10,20,30,40,50",
                   help="comma-separated patch sizes in pixels; the default is RSA's five")
    p.add_argument("--renorm", default=rsa.DEFAULT_RENORMALISATION,
                   choices=rsa.RENORMALISATIONS)
    p.add_argument("--window-rule", default=rsa.DEFAULT_WINDOW_RULE,
                   choices=rsa.WINDOW_RULES,
                   help="which reading of the window rule to run under")
    p.add_argument("--score-frame", default=rsa.DEFAULT_SCORE_FRAME,
                   choices=rsa.SCORE_FRAMES,
                   help="which mean the anomaly score is taken against")
    p.add_argument("--n-images", type=int, default=config.SCALE.n_eval_images)
    p.add_argument("--steps", type=int, default=config.SCALE.attack_steps)
    p.add_argument("--loc-stride", type=int, default=1, help="keep every n-th location")
    p.add_argument("--no-defense", action="store_true",
                   help="force the window to 0, which is ordinary attention: the "
                        "undefended baseline measured under this exact protocol")
    p.add_argument("--checkpoint",
                   help="RSA's released ImageNet-100 ViT-small. Loaded whole, with no "
                        "head slicing and no class subset: the checkpoint is already "
                        "100-way. Step 6h")
    p.add_argument("--tag", default="full", help="names the output files")
    p.add_argument("--loss", default="dlr", help="patch_autopgd loss: dlr or ce")
    p.add_argument("--init", default="uniform", choices=("uniform", "boundary"),
                   help="patch initialisation; RSA does not state which it used")
    return p.parse_args()


def load_progress(path: Path) -> dict[tuple[int, int, int], list[int]]:
    """Replay the per-location log: {(size, top, left): indices misclassified}."""
    done: dict[tuple[int, int, int], list[int]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done[(r["size"], r["top"], r["left"])] = r["wrong"]
    return done


def main():
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    defense_note = " (undefended baseline)" if args.no_defense else ""
    rsa_state = "RSA off" if args.no_defense else "RSA active"

    config.seed_everything()
    device = config.get_device()
    print(config.describe())
    print(f"\ngate A{defense_note}: sizes {sizes}, window rule {args.window_rule!r}, "
          f"renormalise {args.renorm!r}, score frame {args.score_frame!r}, {args.steps}-step "
          f"patch_autopgd({args.loss}), {args.n_images} images, location stride {args.loc_stride}\n")

    ds, wnids, class_indices = data.build_dataset()
    sub = data.eval_subset(ds)
    loader = data.make_loader(sub)

    # Which 100 classes are actually on disk, read from the data rather than inferred
    # from the flags: the model and the class subset are independent variables of step
    # 6h, and a record that names the wrong one is worse than one that names neither.
    if wnids == sorted(data.rsa_class_wnids()):
        class_draw = "rsa_seed0"
    elif wnids == sorted(data.imagenet100_wnids()):
        class_draw = "local_seed1234"
    else:
        class_draw = "unrecognised"
    print(f"class draw: {class_draw} ({len(wnids)} classes)")

    if args.checkpoint:
        expected = sorted(data.rsa_class_wnids())
        if wnids != expected:
            raise SystemExit(
                f"--checkpoint expects RSA's seed-0 class draw and the data on disk is a "
                f"different set ({len(set(wnids) & set(expected))} of {len(expected)} "
                f"classes overlap).\nRebuild with scripts/build_imagenet100.py against "
                f"data.rsa_class_wnids() first; the checkpoint's head is ordered by that "
                f"list and a mismatch silently scores against the wrong classes."
            )
        clf, info = models.load_rsa_checkpoint(args.checkpoint, device=device)
        print(f"loaded {args.checkpoint}: {info['num_classes']}-way, "
              f"{len(info['missing_keys'])} missing / {len(info['unexpected_keys'])} "
              f"unexpected keys")
    else:
        clf, info = models.load_model(class_indices=class_indices, device=device)

    batches = []
    taken = 0
    for x, y in loader:
        if taken >= args.n_images:
            break
        keep = min(len(y), args.n_images - taken)
        batches.append((x[:keep].to(device), y[:keep].to(device)))
        taken += keep
    n = taken
    print(f"{n} images in {len(batches)} batches")

    progress_path = config.RESULTS_ROOT / f"gate_a_{args.tag}.progress.jsonl"
    done = load_progress(progress_path)
    if done:
        print(f"resuming: {len(done)} locations already recorded in {progress_path.name}")

    per_size = {}
    t_start = time.time()

    for size in sizes:
        cfg = rsa.RSAConfig.for_patch(size, renormalise=args.renorm,
                                     window_rule=args.window_rule,
                                     score_frame=args.score_frame)
        if args.no_defense:
            # The patch size still drives the attack; only the defense goes away.
            # `window = 0` is RSAConfig's documented no-masking case, so the wrapper
            # stays installed and the forward path matches the defended sweeps.
            cfg = replace(cfg, window=0)
        locations = attacks.patch_locations(size)[:: args.loc_stride]
        print(f"\n--- {size}px patch, window {cfg.window}x{cfg.window}, "
              f"{len(locations)} of {len(attacks.patch_locations(size))} locations ---")

        with rsa.enable(clf, cfg):
            clean_ok = torch.cat([attacks.accuracy(clf, x, y) for x, y in batches])
            clean_acc = 100.0 * float(clean_ok.float().mean())
            print(f"clean accuracy, {rsa_state}: {clean_acc:.2f}%   "
                  f"(RSA Table 2: {TABLE2_RSA.get(size, float('nan')):.2f})")

            # Seeded from the clean predictions, not from zeros. An image RSA already
            # misclassifies is not robust: the patch may hold the original pixels, so
            # the clean image is a feasible attack point. Starting at zeros counts
            # those images robust and inflates the number Gate A is compared against.
            ever_wrong = ~clean_ok.to(device)
            for key, wrong in done.items():
                if key[0] == size:
                    ever_wrong[torch.as_tensor(wrong, device=device, dtype=torch.long)] = True

            for li, (top, left) in enumerate(locations, 1):
                key = (size, top, left)
                if key in done:
                    continue
                t0 = time.time()
                wrong_idx = []
                base = 0
                for x, y in batches:
                    adv = attacks.patch_autopgd(clf, x, y, top=top, left=left, size=size,
                                                steps=args.steps, loss=args.loss,
                                                init=args.init)
                    bad = ~attacks.accuracy(clf, adv, y)
                    wrong_idx += (base + bad.nonzero(as_tuple=True)[0]).tolist()
                    base += len(y)

                done[key] = wrong_idx
                if wrong_idx:
                    ever_wrong[torch.as_tensor(wrong_idx, device=device, dtype=torch.long)] = True
                with progress_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"size": size, "top": top, "left": left,
                                         "wrong": wrong_idx}) + "\n")

                ra = 100.0 * float((~ever_wrong).float().mean())
                dt = time.time() - t0
                left_s = dt * (len(locations) - li)
                print(f"  [{li:>3}/{len(locations)}] ({top:>3},{left:>3})  "
                      f"{len(wrong_idx):>3} wrong here, worst-case RA so far {ra:6.2f}%  "
                      f"{dt:5.1f}s  eta {left_s / 3600:.1f} h")

            robust_acc = 100.0 * float((~ever_wrong).float().mean())

        reference_row = "ViT-small" if args.no_defense else "ViT-small + RSA"
        target = (TABLE1_UNDEFENDED if args.no_defense else TABLE1_RSA).get(size)
        per_size[size] = {
            "window": cfg.window,
            "defense": "none" if args.no_defense else "rsa",
            "renormalise": args.renorm,
            "window_rule": args.window_rule,
            "score_frame": args.score_frame,
            "n_locations": len(locations),
            "n_locations_full": len(attacks.patch_locations(size)),
            "clean_acc_pct": clean_acc,
            "clean_acc_reference_table2": TABLE2_RSA.get(size),
            "robust_acc_pct": robust_acc,
            "robust_acc_counts_clean_errors_as_not_robust": True,
            "robust_acc_reference_table1": target,
            "reference_row": reference_row,
            "undefended_reference_table1": TABLE1_UNDEFENDED.get(size),
            "delta_vs_reference": None if target is None else robust_acc - target,
        }
        if target is None:
            print(f"\n{size}px: robust accuracy {robust_acc:.2f}%   "
                  f"(no RSA Table 1 reference for this size)")
        else:
            print(f"\n{size}px: robust accuracy {robust_acc:.2f}%   "
                  f"RSA Table 1 ({reference_row}): {target:.2f}%   "
                  f"delta {robust_acc - target:+.2f}")

    elapsed = time.time() - t_start
    print(f"\ntotal {elapsed / 3600:.2f} h")

    payload = {
        "purpose": "Gate A - reproduce the ViT-small + RSA row of RSA Table 1",
        "tag": args.tag,
        "settings": {
            "sizes": sizes, "defense": "none" if args.no_defense else "rsa",
            "renormalise": args.renorm,
            "window_rule": args.window_rule, "score_frame": args.score_frame,
            "n_images": n,
            "attack": f"patch_autopgd({args.loss})", "steps": args.steps,
            "init": args.init,
            "loc_stride": args.loc_stride, "location_grid": "range(0, 224 - size + 1, 20)",
            "checkpoint": args.checkpoint,
            "class_draw": class_draw,
        },
        "per_size": per_size,
        "hours": round(elapsed / 3600, 3),
        "partial": args.loc_stride > 1 or n < config.SCALE.n_eval_images,
        "model_info": info,
    }
    results.save(f"gate_a_{args.tag}", payload)


if __name__ == "__main__":
    main()
