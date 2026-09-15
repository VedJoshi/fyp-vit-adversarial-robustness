"""Evaluate transferred patches or a direct selection-aware attack against one target.

Swap defense configs while retaining the same model and image. Read the
target's original failures from its progress file; --combine unions all
recorded attacks without another sweep. --source self checks replay identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import attacks, config, data, models, results, rsa  # noqa: E402
from gate_a import paired_seed  # noqa: E402
from selection_attack import evasion_loss  # noqa: E402

# Generation-side overrides on the target config.
SOURCES = {
    "undefended": dict(window=0),
    "mean": dict(window_rule="ceil+1", freeze_mask_after=None),
    "centred": dict(window_rule="centred", freeze_mask_after=None),
    "self": {},
    # Direct attack on the target using the selection auxiliary loss.
    "selection": {},
}

AUX_SOURCES = ("selection",)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=sorted(SOURCES),
                   help="the defense the patch is optimised against; omit with --combine")
    p.add_argument("--aux-weight", type=float, default=0.5,
                   help="weight on the selection-evasion term, for --source selection; "
                        "0.5 was chosen on development images (job 847075_0) and is "
                        "frozen -- changing it makes this a new development run")
    p.add_argument("--target-tag", default="gate_a_diag_centred_frz0",
                   help="progress file of the arm being attacked, for its own failures")
    p.add_argument("--tag", default=None, help="output tag; defaults to transfer_<source>")
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--n-images", type=int, default=512)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--loc-stride", type=int, default=1)
    p.add_argument("--loss", default="dlr")
    p.add_argument("--init", default="uniform")
    p.add_argument("--schedule", default="counter")
    p.add_argument("--window-rule", default="centred", choices=rsa.WINDOW_RULES)
    p.add_argument("--freeze-mask-after", type=int, default=0, metavar="K")
    p.add_argument("--score-frame", default="global", choices=rsa.SCORE_FRAMES)
    p.add_argument("--renorm", default="uniform", choices=rsa.RENORMALISATIONS)
    p.add_argument("--backward", default="intended", choices=rsa.BACKWARDS)
    p.add_argument("--combine", default=None, metavar="TAG,TAG,...",
                   help="attack nothing; re-summarise these recorded transfer runs "
                        "against --target-tag and report the union across all of them")
    p.add_argument("--out", default=None, help="record name for --combine")
    args = p.parse_args()
    if args.combine is None and args.source is None:
        p.error("--source is required unless --combine is given")
    return args


def read_target(tag: str, results_root: Path):
    """`(clean_wrong, {(top, left): wrong}, fingerprint)` from a recorded sweep."""
    path = results_root / f"{tag}.progress.jsonl"
    if not path.exists():
        raise SystemExit(
            f"no {path}. The target's own failures are half of every union this script "
            f"reports, and rerunning its sweep costs another five GPU-hours, so the "
            f"recorded file is required rather than optional.")
    clean_wrong, per_loc, fingerprint = set(), {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record") == "protocol":
            clean_wrong = set(r["clean_wrong"])
            fingerprint = r["fingerprint"]
        else:
            per_loc[(r["top"], r["left"])] = set(r["wrong"]) | set(r.get("clean_wrong", ()))
    if fingerprint is None:
        raise SystemExit(f"{path.name} has no protocol header, so its sample and defense "
                         f"are unrecoverable and a union with it would be meaningless")
    return clean_wrong, per_loc, fingerprint


def check_comparable(mine: dict, theirs: dict) -> None:
    """Require the same image population; target defense and locations are checked separately."""
    must_match = ("n_images", "sample_seed", "wnids_sha1", "class_draw")
    differs = {k: (mine.get(k), theirs.get(k)) for k in must_match
               if mine.get(k) != theirs.get(k)}
    if differs:
        raise SystemExit(
            "this run and the target drew different image populations, so their failures "
            "cannot be unioned:\n" +
            "\n".join(f"  {k}: this run {a!r}, target {b!r}"
                      for k, (a, b) in differs.items()))


def check_target_config(cfg, theirs: dict) -> None:
    """Require the evaluation defense to match the target record."""
    mine = {"window": cfg.window, "window_rule": cfg.window_rule,
            "renormalise": cfg.renormalise, "score_frame": cfg.score_frame,
            "backward": cfg.backward, "freeze_mask_after": cfg.freeze_mask_after}
    differs = {k: (v, theirs.get(k)) for k, v in mine.items() if theirs.get(k) != v}
    if theirs.get("checkpoint") is not None:
        differs["checkpoint"] = (None, theirs["checkpoint"])
    if differs:
        raise SystemExit(
            "the defense this run evaluates under is not the one the target recorded:\n" +
            "\n".join(f"  {k}: this run {a!r}, target record {b!r}"
                      for k, (a, b) in differs.items()) +
            "\nPass flags that match the record, or point --target-tag at the arm you "
            "actually mean to attack.")


_ABSENT = object()

_RECORDED_TARGET = {"target_window": "window",
                    "target_window_rule": "window_rule",
                    "target_renormalise": "renormalise",
                    "target_score_frame": "score_frame",
                    "target_backward": "backward",
                    "target_freeze_mask_after": "freeze_mask_after"}


def check_recorded_target(tag: str, fp: dict, theirs: dict, target_tag: str) -> list[str]:
    """Reject a different target defense; return fields absent from older records."""
    if fp.get("target_tag") not in (None, target_tag):
        raise SystemExit(
            f"{tag} was evaluated against target {fp['target_tag']!r}, not "
            f"{target_tag!r}; its failures belong to a different arm's suite.")
    differs, unchecked = {}, []
    for k, their_key in _RECORDED_TARGET.items():
        mine = fp.get(k, _ABSENT)
        if mine is _ABSENT:
            unchecked.append(their_key)
        elif mine != theirs.get(their_key):
            differs[their_key] = (mine, theirs.get(their_key))
    if differs:
        raise SystemExit(
            f"{tag} evaluated under a different defense from the target record:\n" +
            "\n".join(f"  {k}: that run {a!r}, target record {b!r}"
                       for k, (a, b) in differs.items()))
    return unchecked


def check_self_protocol(mine: dict, theirs: dict, clean_wrong: set,
                        tgt_clean_wrong: set) -> None:
    """Require matching attack settings, batching and clean-error identities for self replay."""
    must_match = ("attack", "steps", "init", "schedule", "batch_sizes", "init_rng",
                  "attack_survivors_only", "loc_stride")
    differs = {k: (mine.get(k), theirs.get(k)) for k in must_match
               if mine.get(k) != theirs.get(k)}
    if clean_wrong != tgt_clean_wrong:
        differs["clean_wrong"] = (
            f"{len(clean_wrong)} images, {len(clean_wrong - tgt_clean_wrong)} not in the record",
            f"{len(tgt_clean_wrong)} images, {len(tgt_clean_wrong - clean_wrong)} not in this run")
    if differs:
        raise SystemExit(
            "--source self is a control and must reproduce the target's protocol "
            "exactly; these differ:\n" +
            "\n".join(f"  {k}: this run {a!r}, target record {b!r}"
                      for k, (a, b) in differs.items()))


def main():
    args = parse_args()
    if args.combine:
        combine(args)
        return
    tag = args.tag or f"transfer_{args.source}"
    size = args.size
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    ds, wnids, class_indices = data.build_dataset()
    sub = data.eval_subset(ds, args.n_images)
    class_draw = ("rsa_seed0" if wnids == sorted(data.rsa_class_wnids())
                  else "local_seed1234" if wnids == sorted(data.imagenet100_wnids())
                  else "unrecognised")
    if class_draw == "unrecognised":
        raise SystemExit("the wnids on disk match neither class draw; see gate_a.py")
    clf, _ = models.load_model(class_indices=class_indices, device=device)

    xs, ys, taken = [], [], 0
    for x, y in data.make_loader(sub):
        if taken >= args.n_images:
            break
        keep = min(len(y), args.n_images - taken)
        xs.append(x[:keep].to(device))
        ys.append(y[:keep].to(device))
        taken += keep
    x_all, y_all, n = torch.cat(xs), torch.cat(ys), taken
    del xs, ys

    target_cfg = rsa.RSAConfig.for_patch(size, renormalise=args.renorm,
                                         window_rule=args.window_rule,
                                         score_frame=args.score_frame,
                                         backward=args.backward)
    target_cfg = replace(target_cfg, freeze_mask_after=args.freeze_mask_after)
    source_cfg = replace(target_cfg, **SOURCES[args.source])
    if args.source in AUX_SOURCES and args.aux_weight <= 0:
        raise SystemExit(f"--source {args.source} needs a positive --aux-weight; at 0 it "
                         f"is the plain attack and `--source self` already measures that.")

    tgt_clean_wrong, tgt_per_loc, tgt_fp = read_target(args.target_tag, config.RESULTS_ROOT)
    locations = [l for l in attacks.patch_locations(size)[:: args.loc_stride]]
    mine = {"n_images": n, "sample_seed": config.SEED, "class_draw": class_draw,
            "wnids_sha1": hashlib.sha1(",".join(wnids).encode()).hexdigest(),
            "loc_stride": args.loc_stride}
    check_comparable(mine, tgt_fp)
    # Reject mismatched settings before the expensive sweep.
    check_target_config(target_cfg, tgt_fp)
    missing = [l for l in locations if l not in tgt_per_loc]
    if missing:
        raise SystemExit(
            f"the target swept no patch at {len(missing)} of these {len(locations)} "
            f"locations, starting {missing[:4]}. Every location in this run must be one "
            f"the target also attacked, or the union is not a union; widen --loc-stride.")

    bs = args.batch_size
    print(f"\ntransfer: patches optimised against {args.source!r} "
          f"({source_cfg.window_rule if source_cfg.window else 'no defense'}, "
          f"freeze={source_cfg.freeze_mask_after}), evaluated on "
          f"{target_cfg.window_rule!r} freeze={target_cfg.freeze_mask_after}")
    print(f"{n} images, {len(locations)} of the target's {len(tgt_per_loc)} locations, "
          f"{args.steps} steps, batch {bs}")
    print(f"target's own sweep: {args.target_tag}\n")

    progress_path = config.RESULTS_ROOT / f"gate_a_{tag}.progress.jsonl"
    per_loc: dict[tuple[int, int], set[int]] = {}
    t_start = time.time()

    with rsa.enable(clf, target_cfg) as handle:
        clean_ok = torch.cat([attacks.accuracy(clf, x_all[i:i + bs], y_all[i:i + bs])
                              for i in range(0, n, bs)])
        clean_wrong = (~clean_ok).nonzero(as_tuple=True)[0].tolist()
        print(f"clean accuracy under the target: {100.0 * float(clean_ok.float().mean()):.2f}%"
              f"   (target's own record: {100.0 * (n - len(tgt_clean_wrong)) / n:.2f}%)")
        fingerprint = dict(mine, source=args.source,
                           source_window_rule=source_cfg.window_rule,
                           source_window=source_cfg.window,
                           source_freeze_mask_after=source_cfg.freeze_mask_after,
                           target_window_rule=target_cfg.window_rule,
                           target_freeze_mask_after=target_cfg.freeze_mask_after,
                           target_window=target_cfg.window,
                           target_renormalise=target_cfg.renormalise,
                           target_score_frame=target_cfg.score_frame,
                           target_backward=target_cfg.backward,
                           target_tag=args.target_tag,
                           attack=f"patch_autopgd({args.loss})", steps=args.steps,
                           init=args.init, schedule=args.schedule,
                           init_rng="paired_seed(seed, size, top, left, batch offset)",
                           aux_loss=("selection_evasion_margin"
                                     if args.source in AUX_SOURCES else None),
                           aux_weight=(args.aux_weight
                                       if args.source in AUX_SOURCES else 0.0),
                           transferred=args.source not in ("self",) + AUX_SOURCES)
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"record": "protocol", "size": size,
                                 "clean_wrong": clean_wrong,
                                 "clean_acc_pct": 100.0 * float(clean_ok.float().mean()),
                                 "fingerprint": fingerprint}) + "\n")

        if args.source == "self":
            check_self_protocol(dict(mine, attack=f"patch_autopgd({args.loss})",
                                     steps=args.steps, init=args.init,
                                     schedule=args.schedule,
                                     batch_sizes=[len(range(i, min(i + bs, n)))
                                                  for i in range(0, n, bs)],
                                     init_rng="paired_seed(seed, size, top, left, batch offset)",
                                     attack_survivors_only=False,
                                     loc_stride=args.loc_stride),
                                tgt_fp, set(clean_wrong), tgt_clean_wrong)

        for li, (top, left) in enumerate(locations, 1):
            t0 = time.time()
            wrong_idx: list[int] = []
            aux = (evasion_loss(clf, target_cfg, top, left, size, device)
                   if args.source in AUX_SOURCES else None)
            for s in range(0, n, bs):
                take = torch.arange(s, min(s + bs, n), device=device)
                config.seed_everything(paired_seed(size, top, left, s))
                handle.set_config(source_cfg)
                adv = attacks.patch_autopgd(clf, x_all[take], y_all[take],
                                            top=top, left=left, size=size,
                                            steps=args.steps, loss=args.loss,
                                            init=args.init, schedule=args.schedule,
                                            aux_loss=aux,
                                            aux_weight=(args.aux_weight if aux else 0.0))
                handle.set_config(target_cfg)
                bad = ~attacks.accuracy(clf, adv.detach(), y_all[take])
                wrong_idx += take[bad].tolist()

            lost = set(wrong_idx) | set(clean_wrong)
            per_loc[(top, left)] = lost
            with progress_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"record": "location", "size": size,
                                     "top": top, "left": left, "n_attacked": n,
                                     "clean_wrong": [], "wrong": sorted(lost)}) + "\n")
            union = set().union(*per_loc.values())
            dt = time.time() - t0
            print(f"  [{li:>3}/{len(locations)}] ({top:>3},{left:>3})  "
                  f"{len(lost):>3} wrong here, transfer worst-case so far "
                  f"{100.0 * (n - len(union)) / n:6.2f}%  {dt:5.1f}s  "
                  f"eta {dt * (len(locations) - li) / 3600:.1f} h")

    tgt = with_clean({l: tgt_per_loc[l] for l in locations}, tgt_clean_wrong)
    summary = summarise(n, locations, {args.source: per_loc}, tgt, set())
    self_replay = (check_self_replay(locations, per_loc, tgt)
                   if args.source == "self" else None)
    if self_replay is not None:
        summary["self_replay_disagreements"] = self_replay
    results.save(f"gate_a_{tag}", dict(
        purpose="Athalye sign 2 for the patch threat model: transferred patches vs this arm",
        tag=tag, settings=fingerprint,
        clean_acc_pct=100.0 * (n - len(clean_wrong)) / n,
        hours=round((time.time() - t_start) / 3600, 3),
        partial=args.loc_stride > 1 or n < config.SCALE.n_eval_images,
        **summary), n_eval_images=n, attack_steps=args.steps)
    report(args.target_tag, summary, self_replay)


def combine(args) -> None:
    """Union saved failure sets over shared locations without running attacks."""
    tgt_clean_wrong, tgt_per_loc, tgt_fp = read_target(args.target_tag, config.RESULTS_ROOT)
    n = tgt_fp["n_images"]
    sources, locations, unchecked = {}, None, set()
    for tag in [t.strip() for t in args.combine.split(",") if t.strip()]:
        clean, per_loc, fp = read_target(tag, config.RESULTS_ROOT)
        check_comparable(fp, tgt_fp)
        unchecked |= set(check_recorded_target(tag, fp, tgt_fp, args.target_tag))
        missing = [l for l in per_loc if l not in tgt_per_loc]
        if missing:
            raise SystemExit(f"{tag} swept {len(missing)} locations the target did not, "
                             f"starting {missing[:4]}")
        name = fp.get("source", tag)
        sources[name] = with_clean(per_loc, clean)
        here = sorted(per_loc)
        if locations is None:
            locations = here
        elif here != locations:
            shared = sorted(set(here) & set(locations))
            print(f"  NOTE  {tag} swept {len(here)} locations against "
                  f"{len(locations)} for the sources before it; the suite is reported "
                  f"over the {len(shared)} they share")
            locations = shared
    if not sources:
        raise SystemExit("--combine needs at least one tag")

    sources = {k: {l: v[l] for l in locations} for k, v in sources.items()}
    tgt = with_clean({l: tgt_per_loc[l] for l in locations}, tgt_clean_wrong)
    summary = summarise(n, locations, sources, tgt, set())
    self_replay = (check_self_replay(locations, sources["self"], tgt)
                   if list(sources) == ["self"] else None)
    if self_replay is not None:
        summary["self_replay_disagreements"] = self_replay
    print(f"suite over {n} images and {len(locations)} locations, "
          f"target {args.target_tag}")
    if unchecked:
        print(f"  NOTE  these target settings could not be verified against the record, "
              f"because at least one source predates their being recorded: "
              f"{', '.join(sorted(unchecked))}. Treat the union as unverified on them.")
    report(args.target_tag, summary, self_replay)
    if args.out:
        results.save(args.out, dict(
            purpose="transfer suite: every source against one target, and their union",
            target_tag=args.target_tag, sources=sorted(sources),
            combined_from=args.combine, **summary),
            n_eval_images=n, attack_steps=tgt_fp.get("steps"))


def with_clean(per_loc: dict, clean: set) -> dict:
    """Include global clean errors at every location: a clean error is never robust."""
    return {k: v | clean for k, v in per_loc.items()}


def summarise(n, locations, sources: dict, tgt_per_loc, tgt_clean_wrong):
    """Report source-only, pairwise and full-suite accuracy from clean-inclusive failure sets."""
    tgt = {l: tgt_per_loc[l] for l in locations}
    tgt_u = set().union(*tgt.values())

    def agg(d):
        u = set().union(*d.values())
        return {"worst_case_pct": 100.0 * (n - len(u)) / n,
                "mean_per_location_pct": sum(100.0 * (n - len(v)) / n
                                             for v in d.values()) / len(d)}

    out = {"target_own_attack": agg(tgt), "by_source": {}, "n_locations": len(locations)}
    suite = dict(tgt)
    for name, per_loc in sources.items():
        pair = {l: per_loc[l] | tgt[l] for l in locations}
        u = set().union(*per_loc.values())
        out["by_source"][name] = {
            "alone": agg(per_loc),
            "union_with_target": agg(pair),
            "images_broken_only_by_this_source": sorted(u - tgt_u),
            "locations_where_this_source_found_more": sorted(
                [list(l) for l in locations if per_loc[l] - tgt[l]]),
        }
        suite = {l: suite[l] | per_loc[l] for l in locations}

    out["suite_union"] = agg(suite)
    # Compare with sources alone; the full suite already contains every target failure.
    sources_u = set().union(*(set().union(*p.values()) for p in sources.values())) \
        if sources else set()
    out["images_broken_only_by_transfer"] = sorted(sources_u - tgt_u)
    out["images_broken_only_by_target_attack"] = sorted(tgt_u - sources_u)
    return out


def check_self_replay(locations, per_loc, tgt_per_loc) -> list:
    """Compare failure identities at each location; equal aggregate accuracy is insufficient."""
    bad = []
    for l in locations:
        mine, theirs = per_loc[l], tgt_per_loc[l]
        if mine != theirs:
            bad.append({"location": list(l),
                        "only_in_replay": sorted(mine - theirs),
                        "only_in_record": sorted(theirs - mine)})
    return bad


def report(target_tag, s, self_replay=None) -> None:
    w, m = "worst_case_pct", "mean_per_location_pct"
    print(f"\n{'':<34}{'worst case':>12}{'mean/location':>16}")
    t = s["target_own_attack"]
    print(f"  {target_tag + ', own attack':<32}{t[w]:11.2f}%{t[m]:15.2f}%")
    for name, d in s["by_source"].items():
        label = ("adaptive: " if name in AUX_SOURCES else "transfer from ") + name
        print(f"  {label:<32}{d['alone'][w]:11.2f}%{d['alone'][m]:15.2f}%")
        print(f"  {'  with the target attack':<32}{d['union_with_target'][w]:11.2f}%"
              f"{d['union_with_target'][m]:15.2f}%")
    u = s["suite_union"]
    print(f"  {'SUITE UNION, every attack':<32}{u[w]:11.2f}%{u[m]:15.2f}%")

    print()
    for name, d in s["by_source"].items():
        print(f"  {name}: {len(d['images_broken_only_by_this_source']):>3} images broken "
              f"by it alone, extra failures at "
              f"{len(d['locations_where_this_source_found_more']):>3} of "
              f"{s['n_locations']} locations")
    extra_img = len(s["images_broken_only_by_transfer"])
    extra_loc = sorted({tuple(l) for d in s["by_source"].values()
                        for l in d["locations_where_this_source_found_more"]})

    if self_replay is not None:
        if self_replay:
            first = self_replay[0]
            print(f"\nSELF-CONTROL FAILED: the replay disagrees with the record at "
                  f"{len(self_replay)} of {s['n_locations']} locations, first "
                  f"{first['location']} (+{len(first['only_in_replay'])} images the "
                  f"replay broke and the record did not, "
                  f"-{len(first['only_in_record'])} the other way).")
            print("Check the configuration and numerical reproducibility -- GPU model, "
                  "batch sizes, seeds -- before concluding the harness is wrong. Hold the "
                  "other sources until this is understood.")
        else:
            print(f"\nSelf-control passed: the replay's failure set is identical to the "
                  f"record at all {s['n_locations']} locations, image for image.")
        return

    kind = ("the adaptive attack" if set(s["by_source"]) <= set(AUX_SOURCES)
            else "a transfer")
    if extra_img:
        print(f"\n{extra_img} images are broken by {kind} and never by the target's "
              f"own attack. Those are valid adversarial examples its worst-case number "
              f"does not count, and the suite union above is the honest figure. This "
              f"does not by itself establish that the arm's improvement is an artefact.")
    elif extra_loc:
        print(f"\nNo image is broken by a transfer alone, but the transfers do break "
              f"images the target's own attack missed at {len(extra_loc)} individual "
              f"locations -- the target broke those images elsewhere instead. The worst "
              f"case is unchanged; the per-location means are not.")
    elif set(s["by_source"]) <= set(AUX_SOURCES):
        print("\nThe adaptive attack found no failure the target's own attack missed. "
              "Unlike a weak transfer, that is a real negative result about this "
              "objective at this weight -- it attacks the same defense with strictly "
              "more information -- but it bounds only this objective, not every "
              "selection-aware attack.")
    else:
        print("\nThe transfers found no failure the target's own attack missed, at any "
              "location. That is not evidence of robustness: a weak transfer between "
              "different defenses is the expected outcome and settles nothing about "
              "stronger attacks.")


if __name__ == "__main__":
    main()
