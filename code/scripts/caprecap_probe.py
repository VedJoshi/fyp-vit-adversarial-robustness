"""Probe: Chao2 lower bound on breakable images from saved attack runs, by stage.

Usage: python code/scripts/caprecap_probe.py code/results
"""
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(sys.argv[1])
load = lambda n: json.loads((R / f"{n}.json").read_text())
LOCS = ("60,60", "100,20")

# Stage at which each run entered the promotion aggregation.
ARM5 = {0: ["selection_attack_arm5_matched_859868", "selection_attack_arm5_wide_859966"],
        1: ["selection_attack_arm5_smooth_s1", "selection_attack_arm5_low_s1"],
        2: ["selection_attack_arm5_uniform_s2", "selection_attack_arm5_smooth_s2"],
        3: ["selection_attack_arm5_uniform_s3", "selection_attack_arm5_smooth_s3"],
        4: ["selection_attack_arm5_uniform_s4", "selection_attack_arm5_smooth_s4"],
        5: [f"selection_attack_arm5_{f}_s5o{o}" for o in (2, 3, 4) for f in ("uniform", "smooth")],
        6: ["selection_attack_arm5_uniform_s6", "selection_attack_arm5_smooth_s6"],
        7: ["selection_attack_arm5_uniform_s7", "selection_attack_arm5_smooth_s7"]}
FINDER = {0: ["finder_attack_dev_859868", "finder_attack_smooth_dev_859868",
              "finder_attack_smoothfix_dev_859926"],
          2: ["finder_attack_uniform_s2", "finder_attack_smooth_s2"],
          3: ["finder_attack_uniform_s3", "finder_attack_smooth_s3"],
          4: ["finder_attack_uniform_s4", "finder_attack_smooth_s4"],
          5: [f"finder_attack_{f}_s5o{o}" for o in (2, 3, 4) for f in ("uniform", "smooth")],
          6: ["finder_attack_uniform_s6", "finder_attack_smooth_s6"],
          7: ["finder_attack_uniform_s7", "finder_attack_smooth_s7"]}


def arm5_occasions(name, unit):
    """Sets of broken clean-correct images; unit is 'run' (union of weights) or 'weight'."""
    d = load(name)
    clean = set(d["clean_wrong"])
    by_w = {}
    for loc in LOCS:
        for w, rec in d["per_location"][loc].items():
            by_w.setdefault(w, set()).update(set(rec["wrong"]) - clean)
    return [set().union(*by_w.values())] if unit == "run" else list(by_w.values())


def finder_occasions(name, unit):
    d = load(name)
    by_w = {}
    for w, locs in d["by_weight"].items():
        for loc in LOCS:
            wrong = {i for i, ok in enumerate(locs[loc]["correct"]) if not ok}
            by_w.setdefault(w, set()).update(wrong)
    # Clean errors are wrong in every arm at every location; drop images wrong everywhere.
    always = set.intersection(*by_w.values())
    return ([set().union(*by_w.values()) - always] if unit == "run"
            else [s - always for s in by_w.values()]), always


def chao2(occ):
    t = len(occ)
    cnt = Counter(i for s in occ for i in s)
    S = len(cnt)
    f1 = sum(1 for v in cnt.values() if v == 1)
    f2 = sum(1 for v in cnt.values() if v == 2)
    f0 = (t - 1) / t * f1 * (f1 - 1) / (2 * (f2 + 1)) if t > 1 else float("nan")
    return S, f1, f2, S + f0


def table(label, stages, get, n_pop):
    for unit in ("run", "weight"):
        print(f"\n{label}, capture occasion = one {unit}   (population {n_pop} clean-correct images)")
        print(f"{'stage':>5} {'occasions':>9} {'found':>6} {'f1':>4} {'f2':>4} "
              f"{'Chao2 total':>11} {'predicted new':>13} {'new by stage 7':>14}")
        occ, rows = [], []
        for k in sorted(stages):
            for name in stages[k]:
                occ += get(name, unit)
            rows.append((k, len(occ), *chao2(occ)))
        final = rows[-1][2]
        for k, t, S, f1, f2, est in rows:
            print(f"{k:>5} {t:>9} {S:>6} {f1:>4} {f2:>4} {min(est, n_pop):>11.1f} "
                  f"{min(est, n_pop) - S:>13.1f} {final - S:>14}")


clean5 = set(load("selection_attack_arm5_smooth_s4")["clean_wrong"])
table("arm 5", ARM5, arm5_occasions, 128 - len(clean5))
always = finder_occasions("finder_attack_smooth_s4", "run")[1]
table("finder", FINDER, lambda n, u: finder_occasions(n, u)[0], 128 - len(always))
