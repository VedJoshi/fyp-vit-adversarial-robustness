"""Compare two Gate A progress files that should differ in one setting only.

    python scripts/compare_runs.py gate_a_bwd10_intended gate_a_bwd10_released

Written for the backward-pass control (`PREREGISTRATION_backward_2026-09-08.md`), which
turns on a statistic no existing tool produced: **the symmetric difference of the two
runs' `ever_wrong` sets**, the number of images whose robust/non-robust verdict changes.

Robust accuracy alone is the wrong statistic for a paired comparison. Two runs can break
different images and arrive at the same count, and that coincidence read as a null result
is exactly the error a control like this exists to avoid. The aggregate is reported too,
because the pre-registered rule uses both.

The fingerprints are diffed first and printed whatever the outcome. A paired comparison
is only a paired comparison if the pair differs in the field it claims to; two runs that
also differ in sample size or attack steps are not evidence about the field of interest,
and the diff is what shows that rather than the tag names.

Reads `<name>.progress.jsonl` from `--dir`, defaulting to the cluster results directory.
Files written before 8 September 2026 carry no protocol header, so their clean errors are
unrecoverable; those are reported and their `ever_wrong` is the post-attack union alone,
which is an upper bound. See `scripts/pilot_analysis.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def resolve(name: str, explicit_dir: str | None) -> Path:
    """Find `<name>.progress.jsonl` in the cluster results or the local results root.

    Cluster sweeps are scp'd into `cluster results/results/`; a run launched on this
    machine writes to `code/results/`. Searching both means the same command works for
    either without the caller having to remember which produced the file.
    """
    from fyp import config

    roots = ([Path(explicit_dir)] if explicit_dir else
             [Path(__file__).resolve().parents[2] / "cluster results" / "results",
              config.RESULTS_ROOT])
    for root in roots:
        p = root / f"{name}.progress.jsonl"
        if p.exists():
            return p
    searched = "\n  ".join(str(r) for r in roots)
    raise SystemExit(f"no progress file {name}.progress.jsonl in:\n  {searched}")


def read(path: Path):
    """`(ever_wrong, clean_wrong_or_None, fingerprint_or_None, n_locations)` per size."""
    if not path.exists():
        raise SystemExit(f"no such progress file: {path}")
    per_size: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        size = r["size"]
        e = per_size.setdefault(size, {"ever_wrong": set(), "clean_wrong": None,
                                       "fingerprint": None, "n_locations": 0})
        if r.get("record") == "protocol":
            e["clean_wrong"] = set(r["clean_wrong"])
            e["fingerprint"] = r["fingerprint"]
            e["ever_wrong"] |= set(r["clean_wrong"])
        else:
            e["ever_wrong"] |= set(r["wrong"])
            e["n_locations"] += 1
    return per_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="first run's tag, e.g. gate_a_bwd10_intended")
    ap.add_argument("b", help="second run's tag")
    ap.add_argument("--dir", default=None,
                    help="where to look; default searches 'cluster results/results' "
                         "then code/results")
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pa, pb = resolve(args.a, args.dir), resolve(args.b, args.dir)
    print(f"a: {pa}\nb: {pb}")
    A, B = read(pa), read(pb)
    n = args.n_images

    out = {"a": args.a, "b": args.b, "n_images": n, "per_size": {}}

    for size in sorted(set(A) | set(B)):
        if size not in A or size not in B:
            print(f"\n{size}px: present in only one run, skipped")
            continue
        a, b = A[size], B[size]

        print(f"\n=== {size}px ===")

        fa, fb = a["fingerprint"], b["fingerprint"]
        if fa is None or fb is None:
            missing = [t for t, f in ((args.a, fa), (args.b, fb)) if f is None]
            print(f"  no protocol header in: {', '.join(missing)}")
            print("  clean errors unrecoverable there; ever_wrong is the attacked union "
                  "alone and is an upper bound on robustness")
            differs = None
        else:
            differs = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
            if differs:
                print(f"  protocol differs in: {differs}")
                for k in differs:
                    print(f"    {k:<24} {args.a}: {fa.get(k)!r}")
                    print(f"    {'':<24} {args.b}: {fb.get(k)!r}")
            else:
                print("  protocol identical in every recorded field")

        if a["n_locations"] != b["n_locations"]:
            print(f"  WARNING  different location counts: {a['n_locations']} vs "
                  f"{b['n_locations']}; the comparison is not paired")

        ea, eb = a["ever_wrong"], b["ever_wrong"]
        ra, rb = 100 * (n - len(ea)) / n, 100 * (n - len(eb)) / n
        sym = ea ^ eb
        print(f"  locations              {a['n_locations']} / {b['n_locations']}")
        print(f"  worst-case robust acc  {ra:.4f}%  vs  {rb:.4f}%   "
              f"delta {rb - ra:+.4f} ({len(eb) - len(ea):+d} images)")
        print(f"  symmetric difference   {len(sym)} of {n} images "
              f"({100 * len(sym) / n:.2f}%)   {sorted(sym)[:12]}"
              f"{' ...' if len(sym) > 12 else ''}")
        only_a, only_b = sorted(ea - eb), sorted(eb - ea)
        print(f"    broken only by {args.a}: {len(only_a)} {only_a[:10]}")
        print(f"    broken only by {args.b}: {len(only_b)} {only_b[:10]}")

        out["per_size"][size] = {
            "protocol_differs_in": differs,
            "n_locations": [a["n_locations"], b["n_locations"]],
            "robust_acc_pct": [ra, rb],
            "robust_acc_delta_images": len(ea) - len(eb),
            "symmetric_difference": len(sym),
            "symmetric_difference_indices": sorted(sym),
            "only_a": sorted(ea - eb),
            "only_b": sorted(eb - ea),
            "clean_errors_recorded": [a["clean_wrong"] is not None,
                                      b["clean_wrong"] is not None],
        }

    print("\nThe pre-registered rule reads the symmetric difference first, then the "
          "worst-case delta.\nSee PREREGISTRATION_backward_2026-09-08.md.")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
