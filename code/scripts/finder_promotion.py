"""Aggregate every matched attack against each defence, then apply the promotion gate.

Each attack run unions only its own settings. A defence's real survival is the
union over every attack anyone has aimed at it, so the two sides have to be
combined per image before they are compared, and only over the locations both
sides actually ran.

Sources per defence, restricted to the common locations:

  finder    the adaptive pilot (every weight), the second pilot run, and the
            finder's failures under the saved transfers in the replay
  arm 5     its own block-0 evasion attack (every weight) and its failures under
            the same saved transfers

An image survives only if it is clean-correct and correct under every one of
those attacks at every common location.

This is development evidence: 128 images and two locations, not the 512-image
121-location protocol, and matched hardware does not change that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fyp import config, results  # noqa: E402

GATE_MARGIN = 10.0     # points of survival the finder must beat arm 5 by
GATE_CLEAN_DROP = 2.0  # points of clean accuracy it may give up against base


def load(name: str) -> dict:
    return json.loads((config.RESULTS_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finder-runs", default="finder_attack_dev_859868,"
                                             "finder_attack_smooth_dev_859868")
    ap.add_argument("--arm5-runs", default="selection_attack_arm5_matched_859868",
                    help="comma-separated; failures are unioned across them, never "
                         "substituted, so adding a run can only lower arm 5 survival")
    ap.add_argument("--replay", default="selection_replay_finder_dev")
    ap.add_argument("--finder-defence", default="finder_tv")
    ap.add_argument("--out", default="finder_promotion")
    args = ap.parse_args()

    finder_runs = [load(n) for n in args.finder_runs.split(",")]
    arm5_runs = [load(nm) for nm in args.arm5_runs.split(',')]
    replay = load(args.replay)
    n = len(finder_runs[0]["by_weight"]["0.0"][list(finder_runs[0]["by_weight"]["0.0"])[0]]
            ["correct"])

    pilot_locs = {loc for r in finder_runs for w in r["by_weight"] for loc in r["by_weight"][w]}
    replay_locs = set(replay["per_defence"]["base"]["by"])
    arm5_locs = set.intersection(*(set(r["per_location"]) for r in arm5_runs))
    common = sorted(pilot_locs & replay_locs & arm5_locs)
    print(f"common locations: {common}  ({len(pilot_locs)} in the pilots, "
          f"{len(replay_locs)} in the replay, {len(arm5_locs)} in the arm 5 run)")
    if not common:
        raise SystemExit("no location is present in all three sources; the comparison "
                         "would not be matched")

    # Per-image lists are positional, so they are only unionable if every source
    # drew the same images in the same order. Equal n is not enough: dev_subset's
    # draw also depends on the seed, the eval split it excludes and the dataset size.
    named = ([(args.replay, replay)]
             + list(zip(args.finder_runs.split(","), finder_runs))
             + list(zip(args.arm5_runs.split(","), arm5_runs)))
    identities = {nm: r["index"] for nm, r in named if "index" in r}
    ref_name, ref = next(iter(identities.items()))
    for name, ix in identities.items():
        if ix != ref:
            raise SystemExit(f"{name} and {ref_name} are different image draws; "
                             f"positional unions across them would be meaningless")
    draws = {nm: r["settings"]["draw"] for nm, r in named
             if "draw" in r.get("settings", {})}
    for name, d in draws.items():
        if d != next(iter(draws.values())):
            raise SystemExit(f"{name} drew from a different dataset or seed: {d}")
    missing = [nm for nm, _ in named if nm not in identities]
    print(f"image identities verified across {len(identities)} of {len(named)} sources"
          f", draw provenance across {len(draws)}"
          + (f"; NOT verifiable for {missing} (no index recorded, "
             f"re-run to record one)" if missing else ""))
    for name, r in (list(zip(args.arm5_runs.split(","), arm5_runs))
                    + list(zip(args.finder_runs.split(","), finder_runs))):
        got = r["settings"]["n_images"] if "n_images" in r["settings"] else None
        if got is not None and got != n:
            raise SystemExit(f"{name} ran {got} images against {n}")

    # Finder: failures from every pilot weight, plus the saved transfers in the replay.
    finder_fail: set[int] = set()
    for r in finder_runs:
        for w, locs in r["by_weight"].items():
            for loc, v in locs.items():
                if loc in common:
                    finder_fail |= {i for i, ok in enumerate(v["correct"]) if not ok}
    fr = replay["per_defence"][args.finder_defence]
    for loc in common:
        for w, v in fr["by"][loc].items():
            finder_fail |= {i for i, ok in enumerate(v["correct"]) if not ok}
    finder_clean_wrong = {i for i, ok in enumerate(fr["clean_ok"]) if not ok}

    # Arm 5: failures from every evasion weight, plus the same saved transfers.
    arm5_fail: set[int] = set()
    for r in arm5_runs:
        for loc, byw in r["per_location"].items():
            if loc in common:
                for w, v in byw.items():
                    arm5_fail |= set(v["wrong"])
    br = replay["per_defence"]["base"]
    for loc in common:
        for w, v in br["by"][loc].items():
            arm5_fail |= {i for i, ok in enumerate(v["correct"]) if not ok}
    arm5_clean_wrong = {i for i, ok in enumerate(br["clean_ok"]) if not ok}
    arm5_weights = sorted({float(w) for r in arm5_runs
                           for byw in r["per_location"].values() for w in byw})
    finder_settings = sum(len(r["by_weight"]) for r in finder_runs)
    print(f"budget: arm 5 {len(arm5_weights)} weights over {len(arm5_runs)} run(s); "
          f"finder {finder_settings} weight-and-start combinations. Unequal budgets "
          f"make this a sensitivity check, not a control.")

    base_clean = 100.0 * (n - len(arm5_clean_wrong)) / n
    rows = {}
    for name, fail, cw in (("arm 5", arm5_fail, arm5_clean_wrong),
                           (args.finder_defence, finder_fail, finder_clean_wrong)):
        surv = [i for i in range(n) if i not in fail and i not in cw]
        rows[name] = {"survives": len(surv), "survives_pct": 100.0 * len(surv) / n,
                      "clean_accuracy": 100.0 * (n - len(cw)) / n,
                      "attacked_failures": len(fail)}

    print(f"\n{'defence':>12} {'clean':>8} {'survives':>18} {'images ever broken':>20}")
    for name, r in rows.items():
        print(f"{name:>12} {r['clean_accuracy']:7.2f}% "
              f"{r['survives']:>10}/{n} = {r['survives_pct']:5.2f}% "
              f"{r['attacked_failures']:>15}")

    steps_per_source = {nm: r["settings"].get("steps") for nm, r in named}
    steps_seen = {v for v in steps_per_source.values() if v} or {0}

    margin = rows[args.finder_defence]["survives_pct"] - rows["arm 5"]["survives_pct"]
    clean_drop = base_clean - rows[args.finder_defence]["clean_accuracy"]
    pass_margin = margin >= GATE_MARGIN
    pass_clean = clean_drop <= GATE_CLEAN_DROP
    print(f"\ngate, fixed before the pilot ran:")
    print(f"  survival margin over arm 5   {margin:+6.2f} points   "
          f"need >= {GATE_MARGIN}   {'PASS' if pass_margin else 'FAIL'}")
    print(f"  clean accuracy given up      {clean_drop:+6.2f} points   "
          f"need <= {GATE_CLEAN_DROP}   {'PASS' if pass_clean else 'FAIL'}")
    print(f"  thresholds: {'MET' if pass_margin and pass_clean else 'NOT MET'}"
          f"  -- arithmetic on one union, not a promotion decision. Promotion also "
          f"requires the stage table to show the attacks have settled.")
    print("\nboth sides are unions over every attack aimed at them at the common "
          "locations. Development evidence: 128 images, "
          f"{len(common)} locations, not the 121-location protocol.")

    results.save(args.out, dict(
        purpose="matched failure-set union per defence, then the frozen promotion gate",
        sources=dict(finder_runs=args.finder_runs.split(","),
                     arm5_runs=args.arm5_runs.split(","),
                     replay=args.replay),
        caveat="development images and the locations both sides ran; not the "
               "512-image 121-location protocol. Attack-search budgets are not equal "
               "between the two defences, so this is a sensitivity check and not a "
               "control. Adding attacks against one side can only lower that side's "
               "survival, so a verdict change from a new comparator run is one-way and "
               "must be read as such.",
        common_locations=common, gate=dict(margin_points=GATE_MARGIN,
                                           clean_drop_points=GATE_CLEAN_DROP),
        rows=rows, margin_points=margin, clean_drop_points=clean_drop,
        thresholds_met=bool(pass_margin and pass_clean),
        thresholds_met_note="arithmetic on one union. Promotion also requires the "
                            "protocol's stage table to show the attacks have settled.",
        attack_steps_per_source=steps_per_source),
        n_eval_images=n, attack_steps=max(steps_seen))


if __name__ == "__main__":
    main()
