"""Screen centred masking on disjoint development images.

Report peak membership, counterfactual centred coverage and actual mask
coverage separately for each attack generator. This is a nine-location screen.
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

from fyp import attacks, config, data, metrics, models, results, rsa  # noqa: E402
import gate_a  # noqa: E402
from checkpoint_control import geometry  # noqa: E402

SIZE = 20
LOC_STRIDE = 20
# Three locations per alignment group, including border cases.
LOCATIONS = {
    "aligned": [(0, 0), (160, 0), (80, 80)],
    "one axis": [(0, 20), (80, 100), (160, 180)],
    "straddling": [(20, 20), (40, 40), (180, 120)],
}
GENERATORS = {
    "mean ceil+1, unfrozen": dict(window_rule="ceil+1", freeze_mask_after=None),
    "centred, per layer": dict(window_rule="centred", freeze_mask_after=None),
    "centred, frozen at 0": dict(window_rule="centred", freeze_mask_after=0),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-images", type=int, default=128)
    p.add_argument("--n-eval-images", type=int, default=512,
                   help="the evaluation sample the development set must avoid")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tag", default="centred_screen")
    return p.parse_args()


def dev_subset(ds, n, n_eval, seed=config.SEED):
    """Draw n images outside the n_eval-image evaluation sample."""
    taken = set(data.eval_subset(ds, n_eval, seed=seed).indices)
    pool = [i for i in range(len(ds)) if i not in taken]
    if len(pool) < n:
        raise SystemExit(f"development set needs {n} images, "
                         f"{len(pool)} lie outside the evaluation sample")
    return Subset(ds, sorted(random.Random(seed + 1).sample(pool, n)))


def main():
    args = parse_args()
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    ds, wnids, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    sub = dev_subset(ds, args.n_images, args.n_eval_images)
    batches = [(x.to(device), y.to(device))
               for x, y in data.make_loader(sub, batch_size=args.batch_size)]

    undefended_ok = torch.cat([attacks.accuracy(clf, x, y) for x, y in batches])
    print(f"\ndevelopment set: {len(sub)} images disjoint from the {args.n_eval_images}-image "
          f"evaluation sample, undefended clean accuracy "
          f"{100 * undefended_ok.float().mean().item():.2f}%")

    side_grid = metrics.centred_block_side(SIZE, stride=LOC_STRIDE)
    side_free = metrics.centred_block_side(SIZE)
    sides = sorted({side_grid, side_free})
    print(f"centred block side: {side_grid} on the stride-{LOC_STRIDE} grid, "
          f"{side_free} for arbitrary offsets")

    record = {
        "purpose": "centred-rule feasibility screen; see 02_direction/SCREEN_PREREG_2026-09-13.md",
        "tag": args.tag,
        "settings": {
            "size": SIZE,
            "loc_stride": LOC_STRIDE,
            "attack": f"patch_autopgd(dlr), {args.steps} steps, uniform init, counter schedule",
            "generators": dict(GENERATORS),
            "init_rng": "paired_seed(seed, size, top, left, batch offset)",
            "clean_baseline": "defended, one pass per generator",
            "locations": {g: [list(loc) for loc in ls] for g, ls in LOCATIONS.items()},
            "n_dev_images": len(sub),
            "dev_disjoint_from_eval_n": args.n_eval_images,
            "centred_sides": sides,
            "side_on_grid": side_grid,
        },
        "per_generator": {},
        "defended_clean_pct": {},
    }

    for gen_name, gen_cfg in GENERATORS.items():
        window = (side_grid if gen_cfg["window_rule"] == "centred"
                  else metrics.rsa_window_size(SIZE))
        cfg = rsa.RSAConfig(patch_px=SIZE, window=window, score_frame="global", **gen_cfg)
        rows = {}
        # Clean correctness must be measured under this generator's defense.
        with rsa.enable(clf, cfg):
            clean_ok = torch.cat([attacks.accuracy(clf, x, y) for x, y in batches])
        clean_pct = 100 * clean_ok.float().mean().item()
        print(f"\n=== generator: {gen_name} (window {window}x{window}) === "
              f"defended clean {clean_pct:.2f}%")
        for group, locs in LOCATIONS.items():
            for (top, left) in locs:
                t0 = time.time()
                hits = {"argmax": [], "mask": [], **{f"c{s}": [] for s in sides}}
                robust, off = [], 0
                for x, y in batches:
                    ok = clean_ok[off:off + len(y)]
                    # Use the same initial patch across generators.
                    config.seed_everything(gate_a.paired_seed(SIZE, top, left, off))
                    off += len(y)
                    with rsa.enable(clf, cfg) as handle:
                        adv = attacks.patch_autopgd(clf, x, y, top=top, left=left,
                                                    size=SIZE, steps=args.steps)
                        g = geometry(clf, handle, adv, y, ok, top, left, cfg.window,
                                     window_rank=False, centred_sides=sides)
                    hits["argmax"].append(torch.stack(g["argmax_in_patch_by_layer"]))
                    hits["mask"].append(torch.stack(g["full_mask_by_layer"]))
                    for s in sides:
                        hits[f"c{s}"].append(torch.stack(g[f"centred_contain_{s}_by_layer"]))
                    robust.append(g["robust_acc_pct"] * len(y))
                cat = {k: torch.cat(v, dim=1) for k, v in hits.items()}      # (layers, N)
                rows[f"{top},{left}"] = {
                    "alignment": group,
                    "n": int(cat["argmax"].shape[1]),
                    "robust_acc_pct": sum(robust) / len(sub),
                    "argmax_in_patch_pct_by_layer":
                        [100 * v.item() for v in cat["argmax"].float().mean(1)],
                    "installed_mask_contains_pct_by_layer":
                        [100 * v.item() for v in cat["mask"].float().mean(1)],
                    **{f"centred_contain_{s}_pct_by_layer":
                       [100 * v.item() for v in cat[f"c{s}"].float().mean(1)]
                       for s in sides},
                }
                r = rows[f"{top},{left}"]
                print(f"  ({top:3},{left:3}) {group:10} {time.time() - t0:5.0f}s  "
                      f"robust {r['robust_acc_pct']:5.2f}%  block0 argmax "
                      f"{r['argmax_in_patch_pct_by_layer'][0]:5.1f}%  contain{side_grid} "
                      f"{r[f'centred_contain_{side_grid}_pct_by_layer'][0]:5.1f}%")
        record["per_generator"][gen_name] = rows
        record["defended_clean_pct"][gen_name] = clean_pct

    results.save(args.tag, record, n_eval_images=len(sub), attack_steps=args.steps)
    report(record, sides, side_grid)


def report(record, sides, side_grid):
    """Report membership and containment by layer and alignment, weighted by sample count."""
    labels = ([("argmax_in_patch_pct_by_layer", "argmax in patch"),
               ("installed_mask_contains_pct_by_layer",
                "installed mask contains the patch")]
              + [(f"centred_contain_{s}_pct_by_layer", f"centred {s}x{s} contains")
                 for s in sides])
    for gen_name, rows in record["per_generator"].items():
        print(f"\n\n### generator: {gen_name}")
        for key, label in labels:
            print(f"\n{label}, % by block")
            print(f"{'group':12} {'n':>6}  " + " ".join(f"{b:5}" for b in range(12)))
            for group in LOCATIONS:
                sel = [r for r in rows.values() if r["alignment"] == group]
                n = sum(r["n"] for r in sel)
                cols = [sum(r[key][b] * r["n"] for r in sel) / n for b in range(12)]
                print(f"{group:12} {n:6}  " + " ".join(f"{c:5.1f}" for c in cols))
            n = sum(r["n"] for r in rows.values())
            cols = [sum(r[key][b] * r["n"] for r in rows.values()) / n for b in range(12)]
            print(f"{'all':12} {n:6}  " + " ".join(f"{c:5.1f}" for c in cols))
    print(f"\nMembership implies containment at side {side_grid} on this grid, and the border "
          f"clamp makes the converse fail at 4 of these 9 locations, so an excess is expected. "
          f"Centred containment below membership at any block is an implementation fault; the "
          f"installed-mask row carries no such guarantee and is expected to fall below it.")
    print("Under the per-layer centred generator the installed mask is the centred block, so "
          "those two rows are equal at every block by construction.")
    print("Under the frozen-at-0 generator they coincide at block 0 only. Its installed row "
          "after block 0 is the real frozen mask; its centred rows after block 0 describe a "
          "reselection the arm does not perform.")


if __name__ == "__main__":
    main()
