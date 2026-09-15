"""Evaluate RSA over patch sizes and locations, with resumable progress.

An image survives only if clean classification and every attacked location
are correct. Each arm receives a separately optimised attack.

--arm oracle_window supplies a maximum-coverage candidate window;
oracle_extent supplies exactly the patch tokens. Both retain mean replacement
and measure clean accuracy per location.

--released selects the released derivative and attack settings; checkpoint
and class draw remain separate inputs. Resume requires a matching fingerprint.

    python scripts/gate_a.py --sizes 20 --n-images 128 --loc-stride 3 --tag pilot
    python scripts/gate_a.py --tag full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fyp import attacks, config, data, models, results, rsa
from checkpoint_control import NO_WINDOW_RANK, geometry

TABLE1_RSA = {10: 83.20, 20: 72.46, 30: 68.55, 40: 56.25, 50: 43.16}
TABLE1_UNDEFENDED = {10: 45.70, 20: 0.00, 30: 0.00, 40: 0.00, 50: 0.00}
TABLE2_RSA = {0: 93.16, 10: 92.58, 20: 90.43, 30: 90.43, 40: 88.67, 50: 83.98}

ORACLE_ARMS = {"oracle_window": "window", "oracle_extent": "extent"}


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
    p.add_argument("--record-geometry", action="store_true",
                   help="also record where RSA's selection lands on the adversarial "
                        "patch: per-layer token rank, patch-window rank and "
                        "fully-masked rate, reduced over the location's images. "
                        "Costs one extra forward pass per batch")
    p.add_argument("--freeze-mask-after", type=int, default=None, metavar="K",
                   help="blocks after K reuse block K's mask instead of selecting "
                        "their own. Changes the defense, so the arm is attacked in its "
                        "own right; the default reselects in every block, as RSA "
                        "specifies")
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
    p.add_argument("--backward", default=rsa.DEFAULT_BACKWARD, choices=rsa.BACKWARDS,
                   help="which derivative the masking step carries. 'released' is the "
                        "repeated-index scatter the authors' code differentiates; "
                        "'intended' is the natural one. Forward-identical, and the "
                        "attack follows a different gradient. See rsa.BACKWARDS")
    p.add_argument("--schedule", default="counter", choices=("counter", "released"),
                   help="AutoPGD checkpoint spelling; 'released' is the explicit "
                        "[0, 22, 41, ...] list, one iteration later than 'counter'")
    p.add_argument("--batch-size", type=int, default=None,
                   help="images per attack batch; the authors use one 512-image batch")
    p.add_argument("--attack-survivors-only", action="store_true",
                   help="at each location attack only images not already broken, as "
                        "the authors' loop does. The union is unchanged; per-location "
                        "wrong lists stop being per-location robust accuracy")
    p.add_argument("--arm", default="rsa", choices=("rsa", *ORACLE_ARMS),
                   help="which half of the defense is under test. 'rsa' is the defense "
                        "as specified. 'oracle_window' replaces the window argmax with "
                        "the candidate window of --window-rule covering the most patch "
                        "tokens, holding the extent at window^2. 'oracle_extent' masks "
                        "exactly the tokens the patch intersects")
    p.add_argument("--released", action="store_true",
                   help="shorthand for the authors' released reading throughout: "
                        "--backward released --window-rule released --renorm uniform "
                        "--score-frame global --schedule released "
                        "--attack-survivors-only. Still needs their checkpoint and "
                        "their seed-0 classes to be an exact-released control")
    args = p.parse_args()
    if args.released:
        args.backward = "released"
        args.window_rule = "released"
        args.renorm = "uniform"
        args.score_frame = "global"
        args.schedule = "released"
        args.attack_survivors_only = True
    if args.record_geometry and args.window_rule in NO_WINDOW_RANK:
        print(f"note: --window-rule {args.window_rule!r} has no square sliding candidate "
              f"set, so geometry records token rank, full-mask rate and mask overlap "
              f"and drops window rank and ties")
    if args.arm != "rsa":
        if args.no_defense:
            p.error("--no-defense turns off the masking the oracle arms redirect; the "
                    "two cannot be combined")
        if args.attack_survivors_only:
            p.error(
                "the arms pair only when all three attack the same images at each "
                "location, and --attack-survivors-only makes the attacked set depend on "
                "the arm's own clean errors. Spell the released reading out instead: "
                "--backward released --window-rule released --renorm uniform "
                "--score-frame global --schedule released")
        if args.arm == "oracle_window" and args.window_rule == "topk":
            p.error("--window-rule topk masks single tokens and ranges over no window "
                    "set, so there is no candidate window for oracle_window to pick")
    return args


def paired_seed(size: int, top: int, left: int, offset: int) -> int:
    """Seed by size, location and batch offset so arms and resumed runs share initial patches."""
    digest = hashlib.sha1(
        f"{config.SEED}:{size}:{top}:{left}:{offset}".encode()).hexdigest()
    return int(digest[:8], 16)


def reduce_geometry(batches: list[dict]) -> dict:
    """Pool per-image geometry across batches before taking medians or percentages."""
    n_layers = len(batches[0]["token_rank_by_layer"])
    out: dict[str, list[float]] = {}
    for src, dst, reduce_fn in (
        ("token_rank_by_layer", "median_token_rank_by_layer",
         lambda t: t.median().item()),
        ("argmax_in_patch_by_layer", "argmax_in_patch_pct_by_layer",
         lambda t: 100.0 * t.float().mean().item()),
        ("tied_peak_tokens_by_layer", "median_tied_peak_tokens_by_layer",
         lambda t: t.median().item()),
        ("tied_peak_tokens_by_layer", "tied_peak_pct_by_layer",
         lambda t: 100.0 * (t > 1).float().mean().item()),
        ("window_rank_by_layer", "median_window_rank_by_layer",
         lambda t: t.median().item()),
        ("tied_windows_by_layer", "median_tied_windows_by_layer",
         lambda t: t.median().item()),
        ("full_mask_by_layer", "full_mask_pct_by_layer",
         lambda t: 100.0 * t.float().mean().item()),
        ("overlap_by_layer", "mean_overlap_pct_by_layer",
         lambda t: 100.0 * t.mean().item()),
    ):
        if src not in batches[0]:
            continue
        out[dst] = [
            reduce_fn(torch.cat([b[src][i] for b in batches]))
            for i in range(n_layers)
        ]
    npt = batches[0]["n_patch_tokens"]
    out["overlap_hist_by_layer"] = [
        torch.bincount(
            (torch.cat([b["overlap_by_layer"][i] for b in batches]) * npt).round().long(),
            minlength=npt + 1,
        ).tolist()
        for i in range(n_layers)
    ]
    out["n_images"] = int(sum(b["token_rank_by_layer"][0].numel() for b in batches))
    out["n_patch_tokens"] = batches[0]["n_patch_tokens"]
    return out


def protocol_fingerprint(args, cfg, n_images: int, batch_sizes: list[int],
                         class_draw: str, wnids: list[str]) -> dict:
    """Settings that must match before resuming a sweep and unioning its failures."""
    return {
        "defense": "none" if args.no_defense else args.arm,
        "window": cfg.window,
        "window_rule": cfg.window_rule,
        "renormalise": cfg.renormalise,
        "score_frame": cfg.score_frame,
        "backward": cfg.backward,
        "checkpoint": args.checkpoint,
        "class_draw": class_draw,
        "wnids_sha1": hashlib.sha1(",".join(wnids).encode()).hexdigest(),
        "n_images": n_images,
        "sample_seed": config.SEED,
        "attack": f"patch_autopgd({args.loss})",
        "steps": args.steps,
        "init": args.init,
        "init_rng": "paired_seed(seed, size, top, left, batch offset)",
        "schedule": args.schedule,
        "attack_survivors_only": args.attack_survivors_only,
        "batch_sizes": batch_sizes,
        "loc_stride": args.loc_stride,
        "record_geometry": args.record_geometry,
        # Omit unused freeze settings for compatibility with older fingerprints.
        **({"freeze_mask_after": args.freeze_mask_after}
           if args.freeze_mask_after is not None else {}),
    }


def load_progress(path: Path):
    """Replay progress into (done, clean_wrong, loc_clean, protocols).

    done and loc_clean use (size, top, left) keys; clean_wrong and protocols use
    size keys. Oracle clean errors are location-specific. Legacy logs may lack them.
    """
    done: dict[tuple[int, int, int], list[int]] = {}
    clean_wrong: dict[int, list[int]] = {}
    loc_clean: dict[tuple[int, int, int], list[int]] = {}
    protocols: dict[int, dict] = {}
    if not path.exists():
        return done, clean_wrong, loc_clean, protocols
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record") == "protocol":
            clean_wrong[r["size"]] = r["clean_wrong"]
            protocols[r["size"]] = r["fingerprint"]
        else:
            key = (r["size"], r["top"], r["left"])
            done[key] = r["wrong"]
            if "clean_wrong" in r:
                loc_clean[key] = r["clean_wrong"]
    return done, clean_wrong, loc_clean, protocols


def main():
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    defense_note = " (undefended baseline)" if args.no_defense else ""
    rsa_state = "RSA off" if args.no_defense else "RSA active"
    oracle = ORACLE_ARMS.get(args.arm)

    config.seed_everything()
    device = config.get_device()
    print(config.describe())
    print(f"\ngate A{defense_note}: arm {args.arm!r}, sizes {sizes}, window rule {args.window_rule!r}, "
          f"renormalise {args.renorm!r}, score frame {args.score_frame!r}, {args.steps}-step "
          f"patch_autopgd({args.loss}), {args.n_images} images, location stride {args.loc_stride}\n")

    ds, wnids, class_indices = data.build_dataset()
    sub = data.eval_subset(ds, args.n_images)
    loader = data.make_loader(sub)

    if wnids == sorted(data.rsa_class_wnids()):
        class_draw = "rsa_seed0"
    elif wnids == sorted(data.imagenet100_wnids()):
        class_draw = "local_seed1234"
    else:
        class_draw = "unrecognised"
    print(f"class draw: {class_draw} ({len(wnids)} classes)")
    if class_draw == "unrecognised":
        raise SystemExit(
            f"the 100 wnids on disk match neither RSA's seed-0 draw "
            f"({len(set(wnids) & set(data.rsa_class_wnids()))} overlap) nor this "
            f"project's seed-1234 draw "
            f"({len(set(wnids) & set(data.imagenet100_wnids()))} overlap).\n"
            f"Every shape and order check would still pass and clean accuracy would "
            f"still look plausible, so the run would be uninterpretable rather than "
            f"wrong-looking. Rebuild with scripts/build_imagenet100.py."
        )

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

    xs, ys, taken = [], [], 0
    for x, y in loader:
        if taken >= args.n_images:
            break
        keep = min(len(y), args.n_images - taken)
        xs.append(x[:keep].to(device))
        ys.append(y[:keep].to(device))
        taken += keep
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    n = taken
    del xs, ys

    bs = args.batch_size or config.SCALE.batch_size
    batches = [(x_all[i:i + bs], y_all[i:i + bs]) for i in range(0, n, bs)]
    # Batch offsets affect patch seeds and must be recorded for replay.
    batch_sizes = [len(y) for _, y in batches]
    print(f"{n} images in {len(batches)} batches of at most {bs}")
    progress_path = config.RESULTS_ROOT / f"gate_a_{args.tag}.progress.jsonl"
    done, logged_clean_wrong, loc_clean, logged_protocols = load_progress(progress_path)
    if done:
        print(f"resuming: {len(done)} locations already recorded in {progress_path.name}")
        stale = [s for s in sizes if any(k[0] == s for k in done) and s not in logged_protocols]
        if stale:
            raise SystemExit(
                f"{progress_path.name} has locations for sizes {stale} written before "
                f"protocol headers existed (before 8 September 2026). Their defense, "
                f"window rule, score frame, backward, checkpoint and class set are "
                f"unrecoverable, so resuming would union two protocols into one robust "
                f"accuracy. Start a new --tag, or delete the file to rerun from scratch."
            )

    per_size = {}
    t_start = time.time()

    for size in sizes:
        cfg = rsa.RSAConfig.for_patch(size, renormalise=args.renorm,
                                     window_rule=args.window_rule,
                                     score_frame=args.score_frame,
                                     backward=args.backward)
        if args.freeze_mask_after is not None:
            cfg = replace(cfg, freeze_mask_after=args.freeze_mask_after)
        if args.no_defense:
            cfg = replace(cfg, window=0)
        locations = attacks.patch_locations(size)[:: args.loc_stride]
        print(f"\n--- {size}px patch, window {cfg.window}x{cfg.window}, "
              f"{len(locations)} of {len(attacks.patch_locations(size))} locations ---")

        with rsa.enable(clf, cfg) as handle:
            def clean_correct():
                return torch.cat([attacks.accuracy(clf, x, y) for x, y in batches])

            if oracle is None:
                clean_ok = clean_correct()
                clean_acc = 100.0 * float(clean_ok.float().mean())
                clean_wrong = (~clean_ok).nonzero(as_tuple=True)[0].tolist()
                print(f"clean accuracy, {rsa_state}: {clean_acc:.2f}%   "
                      f"(RSA Table 2: {TABLE2_RSA.get(size, float('nan')):.2f})")
                ever_wrong = ~clean_ok.to(device)
            else:
                clean_acc, clean_wrong = None, []
                ever_wrong = torch.zeros(n, dtype=torch.bool, device=device)
                print(f"clean accuracy, arm {args.arm!r}: measured at each location, "
                      f"because the masked tokens move with the patch")

            fingerprint = protocol_fingerprint(args, cfg, n, batch_sizes,
                                               class_draw, wnids)
            if size in logged_protocols and logged_protocols[size] != fingerprint:
                differs = sorted(
                    k for k in set(fingerprint) | set(logged_protocols[size])
                    if fingerprint.get(k) != logged_protocols[size].get(k)
                )
                raise SystemExit(
                    f"{progress_path.name} recorded {size}px under a different "
                    f"protocol; these fields differ: {differs}.\n"
                    f"  logged : { {k: logged_protocols[size].get(k) for k in differs} }\n"
                    f"  current: { {k: fingerprint.get(k) for k in differs} }\n"
                    f"Resuming would union two protocols into one robust accuracy. "
                    f"Use a new --tag."
                )
            if size not in logged_protocols:
                with progress_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"record": "protocol", "size": size,
                                         "clean_wrong": clean_wrong,
                                         "clean_acc_pct": clean_acc,
                                         "fingerprint": fingerprint}) + "\n")
                logged_protocols[size] = fingerprint
                logged_clean_wrong[size] = clean_wrong

            for key, wrong in done.items():
                if key[0] == size:
                    ever_wrong[torch.as_tensor(wrong, device=device, dtype=torch.long)] = True

            for li, (top, left) in enumerate(locations, 1):
                key = (size, top, left)
                if key in done:
                    continue
                t0 = time.time()
                geom_batches: list[dict] = []
                loc_clean_wrong: list[int] = []
                if oracle is not None:
                    handle.set_oracle(top, left, size, mode=oracle, device=device)
                    loc_clean_ok = clean_correct()
                    loc_clean_wrong = (~loc_clean_ok).nonzero(as_tuple=True)[0].tolist()
                    ever_wrong |= ~loc_clean_ok.to(device)

                sel = ((~ever_wrong).nonzero(as_tuple=True)[0] if args.attack_survivors_only
                       else torch.arange(n, device=device))
                wrong_idx = []
                for s in range(0, len(sel), bs):
                    take = sel[s:s + bs]
                    # Seed independently of completed locations so resume preserves initialization.
                    config.seed_everything(paired_seed(size, top, left, s))
                    adv = attacks.patch_autopgd(clf, x_all[take], y_all[take],
                                                top=top, left=left, size=size,
                                                steps=args.steps, loss=args.loss,
                                                init=args.init, schedule=args.schedule)
                    bad = ~attacks.accuracy(clf, adv, y_all[take])
                    wrong_idx += take[bad].tolist()
                    if args.record_geometry:
                        geom_batches.append(
                            geometry(clf, handle, adv, y_all[take],
                                     torch.ones_like(take, dtype=torch.bool),
                                     top, left, cfg.window, size,
                                     frame=args.score_frame,
                                     padded=(args.window_rule == "padded"),
                                     pool=rsa.WINDOW_POOL[args.window_rule],
                                     window_rank=args.window_rule not in NO_WINDOW_RANK))

                lost = sorted(set(wrong_idx) | set(loc_clean_wrong))
                done[key] = lost
                loc_clean[key] = loc_clean_wrong
                if lost:
                    ever_wrong[torch.as_tensor(lost, device=device, dtype=torch.long)] = True
                loc_record = {"record": "location", "size": size,
                              "top": top, "left": left,
                              "n_attacked": len(sel),
                              "clean_wrong": loc_clean_wrong,
                              "wrong": lost}
                if geom_batches:
                    loc_record["geometry"] = reduce_geometry(geom_batches)
                with progress_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(loc_record) + "\n")

                ra = 100.0 * float((~ever_wrong).float().mean())
                dt = time.time() - t0
                left_s = dt * (len(locations) - li)
                clean_note = ("" if oracle is None
                              else f" ({len(loc_clean_wrong):>3} of them wrong clean)")
                print(f"  [{li:>3}/{len(locations)}] ({top:>3},{left:>3})  "
                      f"{len(lost):>3} wrong here{clean_note}, "
                      f"worst-case RA so far {ra:6.2f}%  "
                      f"{dt:5.1f}s  eta {left_s / 3600:.1f} h")

            robust_acc = 100.0 * float((~ever_wrong).float().mean())

        if oracle is None:
            clean_union = logged_clean_wrong.get(size, clean_wrong)
            clean_acc_mean = clean_acc_min = clean_acc
        else:
            per_loc = [loc_clean.get((size, t, l), []) for t, l in locations]
            clean_union = sorted({i for cw in per_loc for i in cw})
            accs = [100.0 * (n - len(cw)) / n for cw in per_loc]
            clean_acc_mean, clean_acc_min = sum(accs) / len(accs), min(accs)
            print(f"clean accuracy over {len(locations)} locations: "
                  f"mean {clean_acc_mean:.2f}%, worst {clean_acc_min:.2f}%, "
                  f"{len(clean_union)} of {n} images wrong at some location")

        reference_row = "ViT-small" if args.no_defense else "ViT-small + RSA"
        target = (TABLE1_UNDEFENDED if args.no_defense else TABLE1_RSA).get(size)
        per_size[size] = {
            "window": cfg.window,
            "defense": "none" if args.no_defense else args.arm,
            "arm": args.arm,
            "renormalise": args.renorm,
            "window_rule": args.window_rule,
            "score_frame": args.score_frame,
            "backward": cfg.backward,
            "protocol_fingerprint": logged_protocols[size],
            "n_clean_errors": len(clean_union),
            "n_locations": len(locations),
            "n_locations_full": len(attacks.patch_locations(size)),
            "clean_acc_pct": clean_acc_mean,
            "clean_acc_min_pct": clean_acc_min,
            "clean_measured_per_location": oracle is not None,
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
        "purpose": (
            "Gate A - reproduce the ViT-small + RSA row of RSA Table 1"
            if args.arm == "rsa" else
            f"Gate A, arm {args.arm}: RSA's token scoring and window argmax replaced by "
            f"the patch location. A near-zero result establishes that perfect "
            f"localisation does not rescue RSA's mean-replacement rule; it says nothing "
            f"about a different replacement rule, aggregation, architecture or training"
        ),
        "tag": args.tag,
        "settings": {
            "sizes": sizes, "defense": "none" if args.no_defense else args.arm,
            "arm": args.arm,
            "renormalise": args.renorm,
            "window_rule": args.window_rule, "score_frame": args.score_frame,
            "backward": args.backward,
            "n_images": n,
            "attack": f"patch_autopgd({args.loss})", "steps": args.steps,
            "init": args.init, "schedule": args.schedule,
            "batch_sizes": batch_sizes,
            "attack_survivors_only": args.attack_survivors_only,
            "released_reading": args.released,
            "record_geometry": args.record_geometry,
            "loc_stride": args.loc_stride, "location_grid": "range(0, 224 - size + 1, 20)",
            "checkpoint": args.checkpoint,
            "class_draw": class_draw,
        },
        "per_size": per_size,
        "hours": round(elapsed / 3600, 3),
        "partial": args.loc_stride > 1 or n < config.SCALE.n_eval_images,
        "model_info": info,
    }
    results.save(f"gate_a_{args.tag}", payload,
                 n_eval_images=n, attack_steps=args.steps)


if __name__ == "__main__":
    main()
