"""E1 summary: union camouflage runs by stage and apply the stopping rule.

    python code/scripts/camouflage_summary.py OUT RECORD.json [RECORD.json ...]

A stage is the set of runs at one step count. For each cell (score, cap), successes
accumulate over locations, starts and stages; a cell is settled at the first stage after
the first that adds at most settle(n) newly broken images: 3 of 512, 1 of 128. Writes results/OUT.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from confidence_intervals import wilson  # noqa: E402
from fyp import config  # noqa: E402


def settle(n: int) -> int:
    return max(1, round(3 * n / 512))


def cells(rec: dict) -> dict[str, set[int]]:
    """Cell -> images broken in this run, unioned over its locations."""
    out: dict[str, set[int]] = {}
    for r in rec.get("unconstrained", {}).values():
        out.setdefault("none", set()).update(r["wrong"])
    for score, by_loc in rec.get("unconstrained_by_defence", {}).items():   # E2 records
        for r in by_loc.values():
            out.setdefault(f"none/{score}", set()).update(r["wrong"])
    for key, r in rec["capped"].items():
        _, sel, q = key.split("/")
        out.setdefault(f"{sel}/{q}", set()).update(r["success"])
    return out


def summarise(records: list[dict]) -> dict:
    first = records[0]
    for r in records[1:]:
        assert r["index"] == first["index"], "runs disagree on the images"
        for k in ("image_set", "size", "locations"):
            assert r["settings"][k] == first["settings"][k], f"runs disagree on {k}"
    n = len(first["index"])
    by_stage: dict[int, dict[str, set[int]]] = {}
    for r in records:
        stage = by_stage.setdefault(r["settings"]["steps"], {})
        for cell, hit in cells(r).items():
            stage.setdefault(cell, set()).update(hit)

    out = {}
    for cell in sorted({c for s in by_stage.values() for c in s}):
        cum, rows, settled = set(), [], None
        for steps in sorted(by_stage):
            if cell not in by_stage[steps]:
                continue
            new = by_stage[steps][cell] - cum
            cum |= new
            rows.append(dict(steps=steps, newly_broken=len(new), cumulative=len(cum)))
            if settled is None and len(rows) > 1 and len(new) <= settle(n):
                settled = steps
        lo, hi = wilson(len(cum), n)
        out[cell] = dict(broken=len(cum), n=n, pct=100 * len(cum) / n,
                         ci=[100 * lo, 100 * hi], settled_at=settled, stages=rows,
                         images=sorted(cum))
    return dict(n=n, image_set=first["settings"]["image_set"],
                settle_threshold=settle(n), cells=out)


def self_check():
    def rec(steps, wrong, success):
        return dict(index=[0, 1, 2, 3], unconstrained={"a": {"wrong": wrong}},
                    capped={"a/tv/0.5": {"success": success}},
                    settings=dict(image_set="dev", size=20, locations=[[0, 0]], steps=steps))
    assert settle(512) == 3 and settle(128) == 1
    s = summarise([rec(100, [0], [1]), rec(100, [2], []), rec(400, [0, 2], [1, 3, 0])])
    assert s["cells"]["none"]["broken"] == 2 and s["cells"]["none"]["settled_at"] == 400
    tv = s["cells"]["tv/0.5"]
    assert tv["broken"] == 3 and tv["settled_at"] is None          # stage 2 adds 2 > 1
    assert [r["newly_broken"] for r in tv["stages"]] == [1, 2]


def main():
    self_check()
    out, paths = sys.argv[1], sys.argv[2:]
    s = summarise([json.loads(Path(p).read_text(encoding="utf-8")) for p in paths])
    s["sources"] = [Path(p).name for p in paths]
    print(f"{s['n']} {s['image_set']} images; settled when a stage adds <= "
          f"{s['settle_threshold']}\n")
    print("| Cell | Broken | % | 95% CI | Per stage (new) | Settled |\n|---|---|---|---|---|---|")
    for cell, c in s["cells"].items():
        stages = ", ".join(f"{r['steps']}: +{r['newly_broken']}" for r in c["stages"])
        print(f"| {cell} | {c['broken']}/{c['n']} | {c['pct']:.2f} | {c['ci'][0]:.2f} to "
              f"{c['ci'][1]:.2f} | {stages} | {c['settled_at'] or 'no'} |")
    (config.RESULTS_ROOT / f"{out}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
