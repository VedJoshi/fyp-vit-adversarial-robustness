"""Merge the per-size shards of a Gate A array job into one record.

    python code/scripts/merge_shards.py --tag reference --job 827620
    python code/scripts/merge_shards.py --tag undefended --job 828478 --compare reference

Reads `results/shards/gate_a_<tag>_<size>px.json` for each size and writes
`results/gate_a_<tag>.json`. The array tasks swept one size each, so the merge is
valid only if they agree on everything else: every `settings` key except `sizes`,
the seed and image count in `_meta`, and the whole `model_info` block. Each is
asserted, and a mismatch aborts before anything is written.

The merged record carries a `sources` array naming each shard, its device, its
write time and its hours, so the merge is re-checkable against the shards it came
from. `hours` is the sum over shards. `_meta.device` lists the distinct devices,
and `_meta.written_utc` is the latest shard's.

`--compare <tag>` prints the robust-accuracy column of `results/gate_a_<tag>.json`
beside the merged one. `--force` overwrites an existing output.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fyp import config  # noqa: E402

SHARDS = config.RESULTS_ROOT / "shards"

#: `settings` keys that must be identical across shards. `sizes` is the one key
#: that legitimately differs: each task records the single size it swept.
VARYING = {"sizes"}


def load_shards(tag: str, sizes: list[int]) -> dict[int, dict]:
    """Returns {size: shard record}. Raises FileNotFoundError naming every missing shard."""
    paths = {s: SHARDS / f"gate_a_{tag}_{s}px.json" for s in sizes}
    missing = [str(p.relative_to(config.VAULT_ROOT)) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing shards: " + ", ".join(missing))
    return {s: json.loads(p.read_text(encoding="utf-8")) for s, p in paths.items()}


def check_agreement(parts: dict[int, dict]) -> None:
    """Asserts the shards differ only in the size they swept."""
    sizes = sorted(parts)
    base = parts[sizes[0]]
    keys = set(base["settings"]) - VARYING

    for s in sizes[1:]:
        other = parts[s]["settings"]
        if set(other) - VARYING != keys:
            raise AssertionError(
                f"{s}px settings keys differ: "
                f"{sorted(set(other) - VARYING ^ keys)}"
            )
        for key in sorted(keys):
            if other[key] != base["settings"][key]:
                raise AssertionError(
                    f"{s}px settings[{key!r}] = {other[key]!r}, "
                    f"10px has {base['settings'][key]!r}"
                )
        for key in ("n_eval_images", "seed", "model", "scale", "dataset"):
            if parts[s]["_meta"][key] != base["_meta"][key]:
                raise AssertionError(
                    f"{s}px _meta[{key!r}] = {parts[s]['_meta'][key]!r}, "
                    f"10px has {base['_meta'][key]!r}"
                )
        if parts[s]["model_info"] != base["model_info"]:
            raise AssertionError(f"{s}px model_info differs from 10px")

    for s in sizes:
        if parts[s]["partial"]:
            raise AssertionError(f"{s}px is a partial sweep; refusing to merge")
        if str(s) not in parts[s]["per_size"]:
            raise AssertionError(f"{s}px shard has no per_size entry for {s}")
        row = parts[s]["per_size"][str(s)]
        if row["n_locations"] != row["n_locations_full"]:
            raise AssertionError(
                f"{s}px swept {row['n_locations']} of {row['n_locations_full']} locations"
            )


def merge(tag: str, parts: dict[int, dict], job: str | None, outcome: str | None) -> dict:
    """Returns the merged record."""
    sizes = sorted(parts)
    base = parts[sizes[0]]
    devices = sorted({p["_meta"]["device"] for p in parts.values()})
    job_note = f" (job {job})" if job else ""

    record = {
        "_meta": dict(
            base["_meta"],
            device=", ".join(devices),
            written_utc=max(p["_meta"]["written_utc"] for p in parts.values()),
            note=f"merged from {len(sizes)} per-size array tasks{job_note}; see sources",
        ),
        "purpose": base["purpose"],
        "tag": tag,
        "settings": dict(base["settings"], sizes=sizes),
        "per_size": {str(s): parts[s]["per_size"][str(s)] for s in sizes},
        "hours": round(sum(p["hours"] for p in parts.values()), 3),
        "partial": False,
        "sources": [
            {
                "size": s,
                "file": f"gate_a_{tag}_{s}px.json",
                "device": parts[s]["_meta"]["device"],
                "written_utc": parts[s]["_meta"]["written_utc"],
                "git_commit": parts[s]["_meta"]["git_commit"],
                "hours": parts[s]["hours"],
            }
            for s in sizes
        ],
        "model_info": base["model_info"],
    }
    if job:
        record["slurm_job"] = job
    if outcome:
        record["outcome"] = outcome
    return record


def readout(record: dict, compare: dict | None, compare_tag: str | None) -> str:
    """Returns the per-size table as text."""
    cols = f"{'size':>5} {'win':>4} {'clean':>7} {'T2':>7} {'dev':>6} | {'robust':>7} {'T1':>7} {'undef':>6} {'delta':>7}"
    if compare:
        cols += f" | {compare_tag[:6]:>6}"
    lines = [cols, "-" * len(cols)]
    for key in sorted(record["per_size"], key=int):
        d = record["per_size"][key]
        line = (
            f"{key:>5} {d['window']:>4} {d['clean_acc_pct']:>7.2f}"
            f" {d['clean_acc_reference_table2']:>7.2f}"
            f" {d['clean_acc_pct'] - d['clean_acc_reference_table2']:>+6.2f} |"
            f" {d['robust_acc_pct']:>7.2f} {d['robust_acc_reference_table1']:>7.2f}"
            f" {d['undefended_reference_table1']:>6.2f} {d['delta_vs_reference']:>+7.2f}"
        )
        if compare:
            other = compare["per_size"].get(key)
            line += f" | {other['robust_acc_pct']:>6.2f}" if other else f" | {'-':>6}"
        lines.append(line)
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True,
                   help="shard tag, e.g. 'reference' for gate_a_reference_<size>px.json")
    p.add_argument("--sizes", default="10,20,30,40,50",
                   help="comma-separated patch sizes to merge")
    p.add_argument("--job", help="Slurm job id, recorded in _meta.note and slurm_job")
    p.add_argument("--outcome", help="one-sentence outcome recorded on the merged file")
    p.add_argument("--compare", metavar="TAG",
                   help="print results/gate_a_<TAG>.json's robust column alongside")
    p.add_argument("--force", action="store_true", help="overwrite an existing output")
    p.add_argument("--dry-run", action="store_true", help="check and print, write nothing")
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    out = config.RESULTS_ROOT / f"gate_a_{args.tag}.json"
    if out.exists() and not (args.force or args.dry_run):
        print(f"{out.relative_to(config.VAULT_ROOT)} exists; pass --force to overwrite")
        return 1

    parts = load_shards(args.tag, sizes)
    check_agreement(parts)
    record = merge(args.tag, parts, args.job, args.outcome)

    compare = None
    if args.compare:
        path = config.RESULTS_ROOT / f"gate_a_{args.compare}.json"
        compare = json.loads(path.read_text(encoding="utf-8"))

    print(f"{len(sizes)} shards agree on settings, seed and model_info")
    print(f"{record['hours']} h across {record['_meta']['device']}\n")
    print(readout(record, compare, args.compare))

    if args.dry_run:
        print(f"\ndry run: {out.relative_to(config.VAULT_ROOT)} not written")
        return 0

    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(config.VAULT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
