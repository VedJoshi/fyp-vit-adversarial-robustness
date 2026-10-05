"""Within-defence readout for the knnfar1 adaptive protocol.

The primary measurement is each defence against itself: survival under the
classification-only arm (weight 0) and under the cumulative union of its own
selection-aware arms, per stage. Not a margin against arm 5 -- a difference
between two figures that are both still falling does not rank them.

Failures are unioned across runs and never substituted, so adding a stage can
only lower survival. An image survives only if it is clean-correct and correct
under every retained attack at every location.

Development evidence: 128 images and two locations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fyp import config, results  # noqa: E402

KNNFAR = ("selection_attack_knnfar1_uniform_s{}", "selection_attack_knnfar1_smooth_s{}")
ARM5 = ("selection_attack_arm5_uniform_s{}_knncmp", "selection_attack_arm5_smooth_s{}_knncmp")


def load(name: str) -> dict:
    return json.loads((config.RESULTS_ROOT / f"{name}.json").read_text())


def union(runs: list[str], zero_only: bool) -> tuple[set[int], set[int], list[int]]:
    """Positions broken at any location and any retained weight, plus clean errors."""
    broken: set[int] = set()
    clean_wrong: set[int] = set()
    index: list[int] | None = None
    for name in runs:
        d = load(name)
        if index is None:
            index = d["index"]
        elif d["index"] != index:
            raise SystemExit(f"{name} draws a different image set; the union would be meaningless")
        clean_wrong |= set(d["clean_wrong"])
        for by_weight in d["per_location"].values():
            for w, rec in by_weight.items():
                if not zero_only or float(w) == 0.0:
                    broken |= set(rec["wrong"])
    return broken, clean_wrong, index


def curve(templates: tuple[str, ...], stages: list[int]) -> list[dict]:
    rows = []
    previous = None
    for s in stages:
        cumulative = [t.format(k) for t in templates for k in stages if k <= s]
        b0, cw, index = union(cumulative, zero_only=True)
        ba, _, _ = union(cumulative, zero_only=False)
        n = len(index)
        survives = n - len(ba | cw)
        rows.append(dict(
            stage=s,
            runs=cumulative,
            survives_w0=n - len(b0 | cw),
            survives_w0_pct=100.0 * (n - len(b0 | cw)) / n,
            survives=survives,
            survives_pct=100.0 * survives / n,
            newly_broken=None if previous is None else previous - survives,
            clean_accuracy=100.0 * (n - len(cw)) / n,
            n_images=n,
        ))
        previous = survives
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="1,2,3")
    ap.add_argument("--primary-bar", type=float, default=55.0,
                    help="absolute survival required, from the frozen protocol")
    ap.add_argument("--clean-allowance", type=float, default=2.0)
    ap.add_argument("--stability", type=int, default=3,
                    help="newly broken images per defence allowed across the added stages")
    ap.add_argument("--out", default="knnfar_promotion")
    args = ap.parse_args()

    stages = [int(s) for s in args.stages.split(",")]
    knnfar = curve(KNNFAR, stages)
    arm5 = curve(ARM5, stages)

    final_k, final_a = knnfar[-1], arm5[-1]
    # Stage 1 is the baseline for this protocol; the extension is everything after it.
    added = {"knnfar1": knnfar[0]["survives"] - final_k["survives"],
             "arm 5": arm5[0]["survives"] - final_a["survives"]}
    met = dict(
        primary_bar=final_k["survives_pct"] >= args.primary_bar,
        clean_allowance=(final_a["clean_accuracy"] - final_k["clean_accuracy"]) <= args.clean_allowance,
        stability=max(added.values()) <= args.stability,
    )

    out = dict(
        purpose="within-defence survival curves for knnfar1 and its matched arm 5 comparator",
        protocol="02_direction/PROTOCOL_2026-09-20_knnfar.md",
        job="866737",
        caveat="128 development images, two locations. Not the 512-image, 121-location protocol.",
        gate=dict(primary_bar=args.primary_bar, clean_allowance=args.clean_allowance,
                  stability=args.stability),
        curves=dict(knnfar1=knnfar, **{"arm 5": arm5}),
        newly_broken_after_stage_1=added,
        thresholds_met=met,
        outcome=("carry to a broader frozen evaluation" if all(met.values())
                 else "unsettled within the evaluation budget" if not met["stability"]
                 else "stop the rule"),
        note=("Stability failing overrides the other two rows, per the frozen decision table. "
              "The margin against arm 5 is deliberately not the primary readout."),
    )
    results.save(args.out, out, n_eval_images=final_k["n_images"], attack_steps=200)

    for name, rows in (("knnfar1", knnfar), ("arm 5", arm5)):
        print(f"\n{name}")
        for r in rows:
            lost = "" if r["newly_broken"] is None else f"  lost {r['newly_broken']}"
            print(f"  s{r['stage']}  w=0 {r['survives_w0']:>3}/{r['n_images']} "
                  f"= {r['survives_w0_pct']:5.2f}%   union {r['survives']:>3}/{r['n_images']} "
                  f"= {r['survives_pct']:5.2f}%{lost}")
    print(f"\nnewly broken after stage 1: {added}, tolerance {args.stability}")
    print(f"thresholds met: {met}")
    print(f"outcome: {out['outcome']}")


if __name__ == "__main__":
    main()
