"""Compare Gate A failure sets, accuracy and protocol settings.

Match image identities across sample sizes and optionally restrict to shared
locations. Matching images does not match batch-dependent patch seeds.
Legacy records without clean errors can overestimate survival.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def resolve(name: str, explicit_dir: str | None) -> Path:
    """Find a progress file in the requested directory or the cluster/local result roots."""
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
    """Read failure sets, clean errors, fingerprints and locations per patch size.

    Image indices are positions within each run's own sample.
    """
    if not path.exists():
        raise SystemExit(f"no such progress file: {path}")
    per_size: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        size = r["size"]
        e = per_size.setdefault(size, {"ever_wrong": set(), "clean_wrong": None,
                                       "fingerprint": None, "n_locations": 0,
                                       "by_location": {}})
        if r.get("record") == "protocol":
            e["clean_wrong"] = set(r["clean_wrong"])
            e["fingerprint"] = r["fingerprint"]
            e["ever_wrong"] |= set(r["clean_wrong"])
        else:
            e["by_location"][(r["top"], r["left"])] = (
                set(r["wrong"]) | set(r.get("clean_wrong", ())))
            e["ever_wrong"] |= e["by_location"][(r["top"], r["left"])]
            e["n_locations"] += 1
    return per_size


def restrict(e: dict, locations) -> None:
    """Restrict location records in place, retaining global clean errors."""
    e["by_location"] = {k: v for k, v in e["by_location"].items() if k in locations}
    e["ever_wrong"] = set(e["clean_wrong"] or ()).union(*e["by_location"].values())
    e["n_locations"] = len(e["by_location"])


def reconcile(a: dict, b: dict, dataset_size: int, override: int | None):
    """Match image identities in place; return (sample_size, note).

    Check seed and class draw where recorded. Different sample sizes must nest;
    remapping their indices does not match the batch-dependent attack seeds.
    """
    from fyp import data

    na = (a["fingerprint"] or {}).get("n_images")
    nb = (b["fingerprint"] or {}).get("n_images")
    for field, what in (("sample_seed", "sampling seed"), ("wnids_sha1", "class draw")):
        got = {(e["fingerprint"] or {}).get(field) for e in (a, b)}
        if len(got) > 1:
            raise SystemExit(
                f"the {what} differs between the runs ({sorted(map(str, got))}); they "
                f"drew from different image populations and the recorded indices name "
                f"different images, whatever the sample sizes are")
    if na is None or nb is None:
        n = override or max(na or 0, nb or 0)
        if not n:
            raise SystemExit("no n_images in either fingerprint; pass --n-images")
        return n, "sample size taken from --n-images; no fingerprint to check it"
    if na == nb:
        return na, None

    big, small_n = (a, nb) if na > nb else (b, na)
    seed = (a["fingerprint"] or {}).get("sample_seed")
    keep = data.matched_positions(small_n, max(na, nb), dataset_size, seed=seed)
    index = {p: i for i, p in enumerate(keep)}
    remap = lambda s: {index[p] for p in s if p in index}          # noqa: E731
    big["ever_wrong"] = remap(big["ever_wrong"])
    big["clean_wrong"] = None if big["clean_wrong"] is None else remap(big["clean_wrong"])
    big["by_location"] = {k: remap(v) for k, v in big["by_location"].items()}
    note = (f"sample sizes differ ({na} vs {nb}); the {max(na, nb)}-image run was "
            f"restricted to the {small_n} image identities the smaller run drew. The "
            f"attack inits were never shared: paired_seed folds in the batch offset")
    return small_n, note


def per_location_mean(e: dict, n: int) -> float:
    """Mean location accuracy, including clean errors, as a percentage."""
    if not e["by_location"]:
        return float("nan")
    clean = set(e["clean_wrong"] or ())
    accs = [100.0 * (n - len(w | clean)) / n for w in e["by_location"].values()]
    return sum(accs) / len(accs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="first run's tag, e.g. gate_a_bwd10_intended")
    ap.add_argument("b", help="second run's tag")
    ap.add_argument("--dir", default=None,
                    help="where to look; default searches 'cluster results/results' "
                         "then code/results")
    ap.add_argument("--n-images", type=int, default=None,
                    help="only for files with no protocol header; otherwise the sample "
                         "size comes from the fingerprints")
    ap.add_argument("--dataset-size", type=int, default=5000,
                    help="images in the evaluation split, for reconciling two runs that "
                         "drew different sample sizes (ImageNet-100 val: 5000)")
    ap.add_argument("--common-locations", action="store_true",
                    help="restrict both runs to the locations they both swept, so the "
                         "worst-case union compares defenses and not location grids")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pa, pb = resolve(args.a, args.dir), resolve(args.b, args.dir)
    print(f"a: {pa}\nb: {pb}")
    A, B = read(pa), read(pb)

    out = {"a": args.a, "b": args.b, "per_size": {}}

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

        if args.common_locations:
            common = set(a["by_location"]) & set(b["by_location"])
            if not common:
                raise SystemExit(f"{size}px: the two runs share no location")
            before = (a["n_locations"], b["n_locations"])
            restrict(a, common)
            restrict(b, common)
            print(f"  restricted to the {len(common)} shared locations "
                  f"(from {before[0]} and {before[1]})")
        elif set(a["by_location"]) != set(b["by_location"]):
            only_a = len(set(a["by_location"]) - set(b["by_location"]))
            only_b = len(set(b["by_location"]) - set(a["by_location"]))
            print(f"  WARNING  the location grids differ: {a['n_locations']} vs "
                  f"{b['n_locations']} locations, {only_a} swept only by {args.a} and "
                  f"{only_b} only by {args.b}; the worst-case union is over different "
                  f"grids. Pass --common-locations.")

        n, note = reconcile(a, b, args.dataset_size, args.n_images)
        if note:
            print(f"  NOTE  {note}")
        ea, eb = a["ever_wrong"], b["ever_wrong"]
        ra, rb = 100 * (n - len(ea)) / n, 100 * (n - len(eb)) / n
        sym = ea ^ eb
        print(f"  locations              {a['n_locations']} / {b['n_locations']}")
        print(f"  worst-case robust acc  {ra:.4f}%  vs  {rb:.4f}%   "
              f"delta {rb - ra:+.4f} ({len(eb) - len(ea):+d} images)")
        ma, mb = per_location_mean(a, n), per_location_mean(b, n)
        print(f"  mean per-location acc  {ma:.4f}%  vs  {mb:.4f}%   delta {mb - ma:+.4f}")
        print(f"  symmetric difference   {len(sym)} of {n} images "
              f"({100 * len(sym) / n:.2f}%)   {sorted(sym)[:12]}"
              f"{' ...' if len(sym) > 12 else ''}")
        only_a, only_b = sorted(ea - eb), sorted(eb - ea)
        print(f"    broken only by {args.a}: {len(only_a)} {only_a[:10]}")
        print(f"    broken only by {args.b}: {len(only_b)} {only_b[:10]}")

        out["per_size"][size] = {
            "protocol_differs_in": differs,
            "n_images": n,
            "sample_note": note,
            "mean_per_location_pct": [ma, mb],
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
