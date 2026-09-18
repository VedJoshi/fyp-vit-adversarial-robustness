"""Replay the saved attacks through candidate selection rules and classify.

Two questions. Does masking several separated blocks predict better than one
block of the same total area? And does the patch location, given for free, repair
the failures already on record? Every defence runs on the same images and the
same saved patches, differing only in where the mask goes:

  base     arm 5, one centred 3x3 block on the highest-scoring token
  multi4   the union of four centred 3x3 blocks, anchors chosen greedily with
           Chebyshev suppression at radius 3
  single6  one centred 6x6 block, 36 tokens against multi4's 36 nominal

The two oracle defences answer a different question: given the patch location,
does the intervention repair the failures the selection-aware attack already
produced against base?

  oracle_window   the maximum-coverage candidate window at the true location
  oracle_extent   exactly the tokens the patch intersects

No attack is optimised here. The patches were optimised against base, so every
number for a defence other than base is transfer from base, not robustness
against an attacker that knows the rule. For the oracles that gap is the point:
no deployable defence has the location, so their figures are an optimistic
reference for what a real finder could reach, not a target it can be held to.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import inputspace_screen  # noqa: E402
from fyp import attacks, config, data, metrics, models, results, rsa  # noqa: E402

DEFENCES = {
    "base": dict(),
    "multi4": dict(n_windows=4, nms_radius=3),
    "single6": dict(window=6),
    "oracle_window": dict(oracle="window"),
    "oracle_extent": dict(oracle="extent"),
    "finder_tv": dict(finder="tv"),
    "finder_blur": dict(finder="blur"),
}


def make_finder(name: str, window: int):
    """Selection from the input pixels, installed as a mask the defence reuses.

    The statistic is the one inputspace_screen predeclared; the geometry is the
    defence's own centred block, so the only difference from base is the ranking.
    """
    fn = inputspace_screen.RULES[name]

    def choose(xb):
        s = fn(xb, inputspace_screen.blur_residual(xb))
        return rsa.centred_token_mask(s.argmax(1), window)
    return choose


def masked_counts(handle) -> list[int]:
    """Tokens masked per image on the most recent pass, from the frozen block."""
    return handle.masks()[0][:, 1:].sum(1).tolist()


def run(clf, handle, x, y, bs: int, finder=None) -> tuple[list[bool], list[int]]:
    ok, sizes = [], []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs]
        if finder is not None:
            handle.set_mask(finder(xb))
        ok += attacks.accuracy(clf, xb, y[i:i + bs]).tolist()
        sizes += masked_counts(handle)
    return ok, sizes


def pct(flags) -> float:
    return 100.0 * sum(flags) / len(flags)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="selection_capture_dev")
    ap.add_argument("--defences", default=",".join(DEFENCES))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="selection_replay_dev")
    args = ap.parse_args()

    names = args.defences.split(",")
    unknown = set(names) - set(DEFENCES)
    if unknown:
        raise SystemExit(f"unknown defences {sorted(unknown)}; pick from {list(DEFENCES)}")
    if "base" not in names:
        raise SystemExit("base is the reference for fixed/broken and for the drift check")

    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    cap = torch.load(config.RESULTS_ROOT / f"{args.capture}.pt", weights_only=True)
    m = cap["meta"]
    size, weights = m["size"], m["weights"]
    n = len(cap["index"])

    ds, wnids, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    base_cfg = rsa.RSAConfig.for_patch(size, renormalise=m["renormalise"],
                                       window_rule=m["window_rule"],
                                       score_frame=m["score_frame"],
                                       backward=m["backward"],
                                       freeze_mask_after=m["freeze_mask_after"])
    assert base_cfg.window == m["window"], "the capture was taken under a different window"

    sub = data.dev_subset(ds, n)
    assert list(sub.indices) == cap["index"].tolist(),         "the development draw no longer matches the capture; rebuild the capture"
    xs, ys = [], []
    for x, y in data.make_loader(sub):
        xs.append(x.to(device))
        ys.append(y.to(device))
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    indices, bs = cap["index"].tolist(), args.batch_size

    print(f"\nreplay of {args.capture} on {n} development images, "
          f"{len(cap['by'])} locations, weights {weights}")
    print("no attack optimisation; every defence but base sees patches optimised "
          "against base\n")

    t_start = time.time()
    per_def: dict[str, dict] = {}
    for name in names:
        opts = dict(DEFENCES[name])
        oracle = opts.pop("oracle", None)
        finder_name = opts.pop("finder", None)
        cfg = dataclasses.replace(base_cfg, **opts)
        with rsa.enable(clf, cfg) as handle:
            finder = make_finder(finder_name, cfg.window) if finder_name else None
            clean_by_loc = {}
            if oracle is None:
                clean_ok, clean_sizes = run(clf, handle, x_all, y_all, bs, finder)
            else:
                # An oracle has no mask without a patch, so its clean cost is
                # measured once per location with the mask where the patch would
                # be. clean_ok is the conjunction, so an image counts as clean
                # only if it survives the mask at every location in the suite.
                clean_sizes, per_loc = [], []
                for loc in cap["by"]:
                    top, left = (int(v) for v in loc.split(","))
                    handle.set_oracle(top, left, size, mode=oracle, device=device)
                    ok, sizes = run(clf, handle, x_all, y_all, bs)
                    clean_by_loc[loc] = {"accuracy": pct(ok), "correct": ok,
                                         "masked_tokens": sum(sizes) / len(sizes)}
                    clean_sizes += sizes
                    per_loc.append(ok)
                clean_ok = [all(o[i] for o in per_loc) for i in range(n)]
                handle.clear_oracle()
            rec = {"config": cfg.as_dict(), "oracle": oracle, "finder": finder_name,
                   "clean_by_location": clean_by_loc, "clean_ok": clean_ok,
                   "clean_accuracy": pct(clean_ok),
                   "clean_masked_tokens": sum(clean_sizes) / len(clean_sizes),
                   "by": {}}
            shape = (f"oracle {oracle}" if oracle else f"finder {finder_name}" if finder_name
                     else f"{cfg.n_windows} x {cfg.window}x{cfg.window}"
                           f"{f' r{cfg.radius}' if cfg.n_windows > 1 else ''}")
            print(f"{name:>13}  {shape:<16}  "
                  f"clean {rec['clean_accuracy']:5.2f}%  "
                  f"masks {rec['clean_masked_tokens']:4.1f} tokens"
                  + ("  (clean is the conjunction over locations)" if oracle else ""))
            for loc, byw in cap["by"].items():
                top, left = (int(v) for v in loc.split(","))
                if oracle is not None:
                    handle.set_oracle(top, left, size, mode=oracle, device=device)
                for w in weights:
                    adv = x_all.clone()
                    adv[:, :, top:top + size, left:left + size] = (
                        byw[str(float(w))]["patch"].to(device))
                    ok, sizes = run(clf, handle, adv, y_all, bs, finder)
                    if oracle is not None:
                        # A window oracle masks 9 tokens and so does the score
                        # argmax; only the positions distinguish them, so check
                        # those rather than the count.
                        want = set(metrics.patch_token_indices(top, left, size))
                        got = set(handle.masks()[0][0].nonzero().flatten().tolist())
                        assert want <= got, (
                            f"{name} at ({top},{left}) missed patch tokens "
                            f"{sorted(want - got)}; the oracle is not installed")
                        if oracle == "extent":
                            assert got == want, f"{name} masked extra tokens {sorted(got - want)}"
                    rec["by"].setdefault(loc, {})[str(w)] = {
                        "correct": ok, "accuracy": pct(ok),
                        "masked_tokens": sum(sizes) / len(sizes)}
                    print(f"          ({top:>3},{left:>3}) w={w:<4} "
                          f"{pct(ok):5.2f}%  masks {sum(sizes) / len(sizes):4.1f}")
        per_def[name] = rec

    # The capture recorded base's own verdict; a mismatch means the replay drifted.
    drift = {}
    if "base" in per_def:
        for loc, byw in cap["by"].items():
            for w in weights:
                want = (~byw[str(float(w))]["wrong"]).tolist()
                got = per_def["base"]["by"][loc][str(w)]["correct"]
                d = [indices[i] for i in range(n) if want[i] != got[i]]
                if d:
                    drift[f"{loc}@{w}"] = d
        total = sum(len(v) for v in drift.values())
        print(f"\nbase reproduces the capture's own verdict on "
              f"{len(weights) * len(cap['by']) * n - total} of "
              f"{len(weights) * len(cap['by']) * n} cases")
        if drift:
            print(f"  {total} disagreements, likely float drift across GPU models: {drift}")

    n_set = len(cap["by"]) * len(weights)
    print(f"\n{'defence':>13} {'clean':>7} | " +
          "  ".join(f"w={w}" for w in weights) +
          f"  lowest  survives all {n_set}   fixed  broken")
    summary = {}
    for name in names:
        rec, accs = per_def[name], []
        for w in weights:
            a = [rec["by"][loc][str(w)]["accuracy"] for loc in cap["by"]]
            accs.append(sum(a) / len(a))
        lowest = min(rec["by"][loc][str(w)]["accuracy"]
                     for loc in cap["by"] for w in weights)
        # Per-image: clean-correct and correct under every saved attack in the suite.
        survives = [i for i in range(n) if rec["clean_ok"][i] and all(
            rec["by"][loc][str(w)]["correct"][i] for loc in cap["by"] for w in weights)]
        fixed, broken = [], []
        for loc in cap["by"]:
            for w in weights:
                b = per_def["base"]["by"][loc][str(w)]["correct"]
                c = rec["by"][loc][str(w)]["correct"]
                fixed += [indices[i] for i in range(n) if c[i] and not b[i]]
                broken += [indices[i] for i in range(n) if b[i] and not c[i]]
        clean_lost = [indices[i] for i in range(n)
                      if per_def["base"]["clean_ok"][i] and not rec["clean_ok"][i]]
        clean_gained = [indices[i] for i in range(n)
                        if rec["clean_ok"][i] and not per_def["base"]["clean_ok"][i]]
        summary[name] = {"clean_accuracy": rec["clean_accuracy"],
                         "mean_by_weight": accs,
                         "lowest_setting_accuracy": lowest,
                         "survives_whole_suite_pct": 100.0 * len(survives) / n,
                         "survives_whole_suite": len(survives),
                         "clean_lost_vs_base": clean_lost,
                         "clean_gained_vs_base": clean_gained,
                         "fixed_cases": len(fixed), "broken_cases": len(broken),
                         "fixed_images": sorted(set(fixed)),
                         "broken_images": sorted(set(broken))}
        print(f"{name:>13} {rec['clean_accuracy']:6.2f}% | " +
              "  ".join(f"{a:6.2f}%" for a in accs) +
              f" {lowest:6.2f}%  {100.0 * len(survives) / n:13.2f}%  "
              f"{len(fixed):>5}  {len(broken):>6}")
    print("\nlowest is the smallest of the per-setting accuracies, an aggregate. "
          "survives is per image: clean-correct and correct under every saved attack.")
    print("fixed and broken count image-location-weight cases against base, over all "
          f"{n * len(cap['by']) * len(weights)}; the image lists are distinct images.")

    results.save(args.out, dict(
        purpose="classification replay of the saved attacks through candidate "
                "selection rules; no attack is re-optimised",
        source=f"{args.capture}.pt",
        caveat="the patches were optimised against base, so every other defence is "
               "measured under transfer, not against an adapted attacker; the oracle "
               "arms additionally use the true patch location, which no deployable "
               "defence has, so they are an optimistic reference and not a ceiling",
        indexing="clean_ok and correct are positions in the development draw; "
                 "fixed_images and broken_images are dataset indices",
        settings=dict(defences={k: DEFENCES[k] for k in names},
                      **{k: m[k] for k in ("size", "steps", "weights", "locations",
                                           "window", "window_rule", "freeze_mask_after",
                                           "score_frame", "backward", "renormalise",
                                           "loss", "init", "schedule")}),
        index=indices,
        summary=summary,
        base_replay_drift=drift,
        per_defence=per_def,
        hours=round((time.time() - t_start) / 3600, 3)),
        n_eval_images=n, attack_steps=m["steps"])


if __name__ == "__main__":
    main()
