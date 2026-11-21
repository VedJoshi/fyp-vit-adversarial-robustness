"""Draw the sealed held-out test set from images no earlier draw could have touched.

    python code/scripts/draw_test_set.py [--check]

Every saved per-image index list must fall inside EXCLUSION, or the script stops. Writes
code/data/test_set.json once; --check re-derives the set and compares it with the file.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fyp import config, data  # noqa: E402

TEST_SEED = 20261121
N_TEST = 512
MAX_N = 1024
EVAL_SIZES = (32, 64, 128, 256, 512)
EXCLUSION = (f"every eval_subset draw of 1 to {MAX_N} images, and every dev_subset draw of 1 to "
             f"{MAX_N} images beside an evaluation set of {EVAL_SIZES}, all at seed {config.SEED}")


def touched(n_ds: int, seed: int = config.SEED) -> set[int]:
    out: set[int] = set()
    for n in range(1, MAX_N + 1):
        out |= set(random.Random(seed).sample(range(n_ds), n))
    for e in EVAL_SIZES:
        held = set(random.Random(seed).sample(range(n_ds), e))
        pool = [i for i in range(n_ds) if i not in held]
        for n in range(1, MAX_N + 1):
            out |= set(random.Random(seed + 1).sample(pool, n))
    return out


def recorded_indices(results: Path) -> set[int]:
    """Dataset indices named in saved records (`index` lists from the probe scripts)."""
    out: set[int] = set()
    for f in results.glob("*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(rec, dict) and isinstance(rec.get("index"), list):
            out |= set(rec["index"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    ds, _, _ = data.build_dataset("imagenet100")
    n_ds = len(ds)
    used = touched(n_ds)
    seen = recorded_indices(config.RESULTS_ROOT)
    assert seen <= used, f"{len(seen - used)} recorded indices fall outside the exclusion"
    assert set(data.eval_subset(ds, 512).indices) <= used
    assert set(data.dev_subset(ds, 128).indices) <= used

    pool = sorted(set(range(n_ds)) - used)
    idx = sorted(random.Random(TEST_SEED).sample(pool, N_TEST))
    root = Path(ds.root)
    rec = dict(
        purpose="sealed held-out test set: use once, for final figures only",
        written="2026-11-21",
        dataset="imagenet100/val", dataset_size=n_ds, seed=TEST_SEED, n=N_TEST,
        exclusion=dict(rule=EXCLUSION, n_excluded=len(used),
                       pool_size=len(pool), recorded_indices_checked=len(seen)),
        indices=idx, sha256=data.index_digest(idx),
        files=[Path(ds.samples[i][0]).relative_to(root).as_posix() for i in idx],
    )

    if args.check:
        old = json.loads(data.TEST_SET.read_text(encoding="utf-8"))
        assert old["indices"] == idx and old["sha256"] == rec["sha256"], "test set differs"
        data.test_subset(ds)
        print(f"ok: {data.TEST_SET} re-derives; sha256 {rec['sha256'][:16]}")
        return
    if data.TEST_SET.exists():
        sys.exit(f"{data.TEST_SET} exists; it is sealed. Use --check.")
    data.TEST_SET.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"wrote {data.TEST_SET}: {N_TEST} of {len(pool)} untouched images "
          f"({len(used)} excluded); sha256 {rec['sha256']}")


if __name__ == "__main__":
    main()
