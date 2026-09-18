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
    ap.add_argument("--arm5-run", default="selection_attack_arm5_matched_859868")
    ap.add_argument("--replay", default="selection_replay_finder_dev")
    ap.add_argument("--finder-defence", default="finder_tv")
    ap.add_argument("--out", default="finder_promotion")
    args = ap.parse_args()

    finder_runs = [load(n) for n in args.finder_runs.split(",")]
    arm5 = load(args.arm5_run)
    replay = load(args.replay)
    n = len(finder_runs[0]["by_weight"]["0.0"][list(finder_runs[0]["by_weight"]["0.0"])[0]]
            ["correct"])

    pilot_locs = {loc for r in finder_runs for w in r["by_weight"] for loc in r["by_weight"][w]}
    replay_locs = set(replay["per_defence"]["base"]["by"])
    arm5_locs = set(arm5["per_location"])
    common = sorted(pilot_locs & replay_locs & arm5_locs)
    print(f"common locations: {common}  ({len(pilot_locs)} in the pilots, "
          f"{len(replay_locs)} in the replay, {len(arm5_locs)} in the arm 5 run)")
    if not common:
        raise SystemExit("no location is present in all three sources; the comparison "
                         "would not be matched")

    # Per-image lists are positional, so they are only unionable if every source
    # drew the same images in the same order.
    identities = {n: r["index"] for n, r in
                  [(args.replay, replay)] + list(zip(args.finder_runs.split(","), finder_runs))
                  if "index" in r}
    ref_name, ref = next(iter(identities.items()))
    for name, ix in identities.items():
        if ix != ref:
            raise SystemExit(f"{name} and {ref_name} are different image draws; "
                             f"positional unions across them would be meaningless")
    missing = [n for n in [args.arm5_run] + args.finder_runs.split(",")
               if n not in identities]
    print(f"image identities verified across {len(identities)} of "
          f"{1 + len(finder_runs) + 1} sources"
          + (f"; NOT verifiable for {missing} (no index recorded, "
             f"re-run to record one)" if missing else ""))
    for name, r in [(args.arm5_run, arm5)] + list(zip(args.finder_runs.split(","), finder_runs)):
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
    for loc, byw in arm5["per_location"].items():
        if loc in common:
            for w, v in byw.items():
                arm5_fail |= set(v["wrong"])
    br = replay["per_defence"]["base"]
    for loc in common:
        for w, v in br["by"][loc].items():
            arm5_fail |= {i for i, ok in enumerate(v["correct"]) if not ok}
    arm5_clean_wrong = {i for i, ok in enumerate(br["clean_ok"]) if not ok}

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

    margin = rows[args.finder_defence]["survives_pct"] - rows["arm 5"]["survives_pct"]
    clean_drop = base_clean - rows[args.finder_defence]["clean_accuracy"]
    pass_margin = margin >= GATE_MARGIN
    pass_clean = clean_drop <= GATE_CLEAN_DROP
    print(f"\ngate, fixed before the pilot ran:")
    print(f"  survival margin over arm 5   {margin:+6.2f} points   "
          f"need >= {GATE_MARGIN}   {'PASS' if pass_margin else 'FAIL'}")
    print(f"  clean accuracy given up      {clean_drop:+6.2f} points   "
          f"need <= {GATE_CLEAN_DROP}   {'PASS' if pass_clean else 'FAIL'}")
    print(f"  overall: {'PROMOTE to a full sweep' if pass_margin and pass_clean else 'DO NOT PROMOTE'}")
    print("\nboth sides are unions over every attack aimed at them at the common "
          "locations. Development evidence: 128 images, "
          f"{len(common)} locations, not the 121-location protocol.")

    results.save(args.out, dict(
        purpose="matched failure-set union per defence, then the frozen promotion gate",
        sources=dict(finder_runs=args.finder_runs.split(","), arm5=args.arm5_run,
                     replay=args.replay),
        caveat="development images and the locations both sides ran; not the "
               "512-image 121-location protocol. Attack-search budgets are not equal "
               "between the two defences: the finder was given five evasion weights "
               "and arm 5 four, so the finder faced the larger search.",
        common_locations=common, gate=dict(margin_points=GATE_MARGIN,
                                           clean_drop_points=GATE_CLEAN_DROP),
        rows=rows, margin_points=margin, clean_drop_points=clean_drop,
        promote=bool(pass_margin and pass_clean)), n_eval_images=n, attack_steps=100)


if __name__ == "__main__":
    main()
