"""Per-location analysis of the full-grid window-rule pilot.

Reads the `.progress.jsonl` files written by `gate_a.py` and reports, per reading of the
window rule: worst-case robust accuracy over locations, the mean per-location accuracy,
and the same figure split two ways -- by distance from the image border, and by whether
the patch is aligned so that it owns a whole token cell.

    python scripts/pilot_analysis.py [--dir DIR] [--tag TAG] [--json OUT]

Every `gate_a_<tag>_<variant>.progress.jsonl` in `--dir` becomes one row, `<variant>`
being whatever the run's `--tag` put after the shared prefix. Defaults read
`../cluster results/results` for tag `pilot20full`, the 20px six-reading pilot from
`slurm/pilot_window_rule.sbatch`, where the variants are the window rules. For the
reference-fidelity 2x2 from `slurm/pilot_reference_fidelity.sbatch` pass
`--tag ref20`, and the variants are `<score frame>_<renormalisation>`.

Incomplete runs are reported and excluded from the grouped statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TOKEN_PX = 16
#: Variants are discovered from the filenames. These are listed first when present, so
#: the six-reading pilot keeps the order its table is quoted in; anything else follows
#: alphabetically.
ORDER = ("ceil+1", "padded", "ceil", "multi", "token", "topk")


def max_cell_overlap(offset_px: int, size_px: int) -> int:
    """Pixels of the single most-covered token cell, for a patch at `offset_px`."""
    lo, hi = offset_px, offset_px + size_px
    cells = range(lo // TOKEN_PX, hi // TOKEN_PX + 1)
    return max(max(0, min(hi, TOKEN_PX * t + TOKEN_PX) - max(lo, TOKEN_PX * t)) for t in cells)


def load(path: Path, n_images: int) -> dict[tuple[int, int], float]:
    """Per-location robust accuracy, keyed by (top_px, left_px)."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["top"], r["left"])] = 100 * (n_images - len(set(r["wrong"]))) / n_images
    return out


def worst_case(path: Path, n_images: int) -> float:
    lost: set[int] = set()
    for line in path.read_text().splitlines():
        if line.strip():
            lost |= set(json.loads(line)["wrong"])
    return 100 * (n_images - len(lost)) / n_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[2] / "cluster results" / "results"))
    ap.add_argument("--tag", default="pilot20full")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--json", default=None, help="write the table here as JSON")
    args = ap.parse_args()

    d = Path(args.dir)
    prefix, suffix = f"gate_a_{args.tag}_", ".progress.jsonl"
    found = {p.name[len(prefix):-len(suffix)]: p for p in d.glob(f"{prefix}*{suffix}")}
    if not found:
        raise SystemExit(f"no progress files matching {prefix}*{suffix} in {d}")
    found = dict(sorted(found.items(),
                        key=lambda kv: (ORDER.index(kv[0]) if kv[0] in ORDER else len(ORDER),
                                        kv[0])))
    label_w = max(8, max(len(v) for v in found))

    offsets = sorted({t for p in found.values() for t, _ in load(p, args.n_images)})
    side = len(offsets)
    owns_cell = {o for o in offsets if max_cell_overlap(o, args.size) == TOKEN_PX}
    print(f"{side}x{side} locations, {args.size}px patch, {args.n_images} images\n")
    print("offset px       :" + "".join(f"{o:5d}" for o in offsets))
    print("px of best cell :" + "".join(f"{max_cell_overlap(o, args.size):5d}" for o in offsets))
    print("owns a cell     :" + "".join(f"{'yes' if o in owns_cell else '-':>5s}" for o in offsets))

    header = (f"\n{'variant':{label_w}s} {'locs':>7s} {'worst':>8s} {'mean':>8s} "
              f"{'border':>8s} {'interior':>9s} {'straddle':>9s} {'aligned':>8s} {'gap':>7s}")
    print(header)
    print("-" * (len(header) - 1))

    table = {}
    for rule, path in found.items():
        cells = load(path, args.n_images)
        wc = worst_case(path, args.n_images)
        n = len(cells)
        vals = list(cells.values())
        row = {"n_locations": n, "complete": n == side * side,
               "worst_case_pct": wc, "mean_per_location_pct": sum(vals) / len(vals)}

        groups: dict[int, list[float]] = {0: [], 1: [], 2: []}
        border, interior = [], []
        for (t, l), v in cells.items():
            groups[(t in owns_cell) + (l in owns_cell)].append(v)
            edge = t in (offsets[0], offsets[-1]) or l in (offsets[0], offsets[-1])
            (border if edge else interior).append(v)

        def avg(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        row.update(border_pct=avg(border), interior_pct=avg(interior),
                   straddle_pct=avg(groups[0]), aligned_pct=avg(groups[2]))
        row["alignment_gap"] = row["straddle_pct"] - row["aligned_pct"]
        table[rule] = row

        flag = "" if row["complete"] else "  PARTIAL"
        print(f"{rule:{label_w}s} {n:7d} {wc:7.2f}% {row['mean_per_location_pct']:7.2f}% "
              f"{row['border_pct']:7.2f}% {row['interior_pct']:8.2f}% "
              f"{row['straddle_pct']:8.2f}% {row['aligned_pct']:7.2f}% "
              f"{row['alignment_gap']:6.2f}{flag}")

    print("\nstraddle = the patch owns no whole token cell on either axis.")
    print("aligned  = it owns one cell outright on both axes.")
    print("gap      = robust accuracy the defense loses to patch-token alignment alone.")

    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
