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
    python scripts/gate_a.py --released --checkpoint vitsmall_100.pt --tag released

The default is RSA's five patch sizes: 523 location-sweeps of 16 batches each. Measured
with RSA active at 20.7 s per 100-step attack batch, that is 48.3 hours -- 11.2 h at 10px
and 20px, 9.2 h at 30px and 40px, 7.5 h at 50px. Record: `results/m5_rsa_sanity.json`.

`--loc-stride` keeps every n-th location, which weakens the worst-case search and so
raises the reported robust accuracy: a pilot below the target is decisive, a pilot at
or above it is not. `--n-images` draws a seeded random sample of that size; it is not a
prefix of the 512-image sample, because `data.eval_subset` sorts its indices and a
prefix of a sorted sample is a prefix of the class directories.

**What this script defaults to is the RSA operator Algorithm 1 describes, not the code
the authors released.** They are forward-identical and differ in four places that a
Table 1 comparison has to control: the derivative of the masking step (`--backward`),
the window-side formula (`--window-rule released`), the AutoPGD checkpoint spelling
(`--schedule`) and the attack loop's batching and still-correct filtering
(`--batch-size`, `--attack-survivors-only`). `--released` sets all of them together.
Two more differences are not flags, because they are inputs: the authors' released
checkpoint (`--checkpoint`) and their seed-0 class draw
(`scripts/build_imagenet100.py` against `data.rsa_class_wnids()`). Until all six agree,
the honest description of a result here is "the intended RSA operator", not "RSA as
released".

Progress is appended to `results/gate_a_<tag>.progress.jsonl` after every location and
replayed on restart, so an interrupted sweep resumes where it stopped. Each size writes
a protocol header first, carrying the clean errors and a fingerprint of everything the
union depends on; a resume whose fingerprint differs is refused rather than silently
pooled.
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
    return args


def protocol_fingerprint(args, cfg, n_images: int, batch_sizes: list[int],
                         class_draw: str, wnids: list[str]) -> dict:
    """Everything that has to match for two runs to be poolable into one union.

    Resume identity used to be `(size, top, left)` alone, so a tag rerun under a
    different defense, window rule, score frame, checkpoint or class set skipped the
    locations the first run had finished, folded their wrong-image sets into
    `ever_wrong`, and then stamped the *current* flags on the record. The result was a
    complete-looking robust accuracy that was a union across two different defenses.
    `main` refuses to resume when this dict differs from the one in the progress file.

    `wnids_sha1` pins the class identities themselves, not just the name of the draw.
    """
    return {
        "defense": "none" if args.no_defense else "rsa",
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
        "schedule": args.schedule,
        "attack_survivors_only": args.attack_survivors_only,
        "batch_sizes": batch_sizes,
        "loc_stride": args.loc_stride,
    }


def load_progress(path: Path):
    """Replay the log.

    Returns `(done, clean_wrong, protocols)`:

    * `done`      `{(size, top, left): indices misclassified after the attack}`
    * `clean_wrong` `{size: indices misclassified on the clean defended input}`
    * `protocols` `{size: fingerprint}` for the protocol headers present

    Location lines carry post-attack errors only. Without the clean errors beside them
    a reader cannot reconstruct robust accuracy, because an image that is wrong clean
    but happens to be *repaired* by the optimised patch is absent from `wrong` and looks
    like a survivor. Runs before 8 September 2026 wrote no protocol or clean-error
    header; they replay as before and `clean_wrong` is empty for them.
    """
    done: dict[tuple[int, int, int], list[int]] = {}
    clean_wrong: dict[int, list[int]] = {}
    protocols: dict[int, dict] = {}
    if not path.exists():
        return done, clean_wrong, protocols
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record") == "protocol":
            clean_wrong[r["size"]] = r["clean_wrong"]
            protocols[r["size"]] = r["fingerprint"]
        else:
            done[(r["size"], r["top"], r["left"])] = r["wrong"]
    return done, clean_wrong, protocols


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
    # Drawn at the size that gets measured. Taking the front of a 512-draw would hand a
    # reduced run the lowest order statistics of a sorted sample, which is a prefix of
    # the class directories rather than a random subsample. See `data.eval_subset`.
    sub = data.eval_subset(ds, args.n_images)
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

    # The batch split is a protocol variable, not an implementation detail: it decides
    # how many patch initialisations come off the RNG stream and in what order, so two
    # runs that differ only here can flip a borderline image. The authors' wrapper puts
    # the whole 512-image sample through in one batch; `--batch-size` names ours.
    bs = args.batch_size or config.SCALE.batch_size
    batches = [(x_all[i:i + bs], y_all[i:i + bs]) for i in range(0, n, bs)]
    batch_sizes = [len(y) for _, y in batches]
    print(f"{n} images in {len(batches)} batches of at most {bs}")
    progress_path = config.RESULTS_ROOT / f"gate_a_{args.tag}.progress.jsonl"
    done, logged_clean_wrong, logged_protocols = load_progress(progress_path)
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
            clean_wrong = (~clean_ok).nonzero(as_tuple=True)[0].tolist()

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
                # Clean errors go in the log beside the attacked errors. The headline
                # union seeds from them here, but a reader of the progress file cannot
                # recover them from the location lines, so a per-location analysis
                # counts a clean error that the optimised patch happens to repair as a
                # survivor. See scripts/pilot_analysis.py.
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
                # Which images this location attacks. `--attack-survivors-only` is the
                # authors' loop: an image already broken at an earlier location cannot
                # become robust again, so the union is identical, but the batches carry
                # different images and consume the RNG differently. It also means the
                # per-location `wrong` lists stop being per-location robust accuracy,
                # which is why the fingerprint records it and pilot_analysis reads it.
                sel = ((~ever_wrong).nonzero(as_tuple=True)[0] if args.attack_survivors_only
                       else torch.arange(n, device=device))
                wrong_idx = []
                for s in range(0, len(sel), bs):
                    take = sel[s:s + bs]
                    adv = attacks.patch_autopgd(clf, x_all[take], y_all[take],
                                                top=top, left=left, size=size,
                                                steps=args.steps, loss=args.loss,
                                                init=args.init, schedule=args.schedule)
                    bad = ~attacks.accuracy(clf, adv, y_all[take])
                    wrong_idx += take[bad].tolist()

                done[key] = wrong_idx
                if wrong_idx:
                    ever_wrong[torch.as_tensor(wrong_idx, device=device, dtype=torch.long)] = True
                with progress_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"record": "location", "size": size,
                                         "top": top, "left": left,
                                         "n_attacked": len(sel),
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
            "backward": cfg.backward,
            # What the locations in this size's union were actually produced under,
            # replayed from the progress file rather than restated from the flags.
            "protocol_fingerprint": logged_protocols[size],
            "n_clean_errors": len(logged_clean_wrong.get(size, clean_wrong)),
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
            "backward": args.backward,
            "n_images": n,
            "attack": f"patch_autopgd({args.loss})", "steps": args.steps,
            "init": args.init, "schedule": args.schedule,
            "batch_sizes": batch_sizes,
            "attack_survivors_only": args.attack_survivors_only,
            "released_reading": args.released,
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
