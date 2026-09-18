"""Development pilot: attack multi4 adaptively and fold the result into a suite.

multi4 is arm 5 with the mask set to the union of four centred 3x3 blocks. This
pilot optimises fresh attacks against it and reports them beside the attacks it
has already been measured under:

  transfer_w0    saved patches optimised against base, classification only
  transfer_w0.5  saved patches optimised against base, selection-aware
  clf            freshly optimised against multi4, classification only
  union_w<...>   freshly optimised against multi4, with union_evasion_loss

A new attack should add failures, never erase known ones, so the suite union is
reported alongside each source. Development images only; no protocol number.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import attacks, config, data, models, results, rsa  # noqa: E402
from gate_a import paired_seed  # noqa: E402
from selection_attack import union_evasion_loss, union_mask_escaped  # noqa: E402


def survivors(clean_ok, by_loc: dict) -> list[int]:
    """Positions that are clean-correct and correct at every location in by_loc."""
    return [i for i in range(len(clean_ok))
            if clean_ok[i] and all(c[i] for c in by_loc.values())]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="selection_capture_dev")
    ap.add_argument("--n-windows", type=int, default=4)
    ap.add_argument("--nms-radius", type=int, default=3)
    ap.add_argument("--aux-weights", default="0.5,2",
                    help="weights for union_evasion_loss; a classification-only "
                         "attack is always run beside them")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="selection_pilot_dev")
    args = ap.parse_args()

    aux_weights = [float(v) for v in args.aux_weights.split(",") if float(v)]
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    cap = torch.load(config.RESULTS_ROOT / f"{args.capture}.pt", weights_only=True)
    m = cap["meta"]
    size, n = m["size"], len(cap["index"])
    locations = [tuple(int(v) for v in loc.split(",")) for loc in cap["by"]]

    ds, wnids, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    cfg = dataclasses.replace(
        rsa.RSAConfig.for_patch(size, renormalise=m["renormalise"],
                                window_rule=m["window_rule"], score_frame=m["score_frame"],
                                backward=m["backward"],
                                freeze_mask_after=m["freeze_mask_after"]),
        n_windows=args.n_windows, nms_radius=args.nms_radius)

    sub = data.dev_subset(ds, n)
    assert list(sub.indices) == cap["index"].tolist(),         "the development draw no longer matches the capture; rebuild the capture"
    xs, ys = [], []
    for x, y in data.make_loader(sub):
        xs.append(x.to(device))
        ys.append(y.to(device))
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    indices, bs = cap["index"].tolist(), args.batch_size

    print(f"\npilot against {args.n_windows} x {cfg.window}x{cfg.window} windows at "
          f"radius {cfg.radius}, {n} development images, {len(locations)} locations, "
          f"{args.steps} steps")
    print(f"fresh attacks: classification only, plus union_evasion_loss at "
          f"{aux_weights}\n")

    t_start = time.time()
    sources: dict[str, dict] = {}
    with rsa.enable(clf, cfg):
        clean_ok = torch.cat([attacks.accuracy(clf, x_all[i:i + bs], y_all[i:i + bs])
                              for i in range(0, n, bs)]).tolist()
        print(f"clean accuracy under multi4: {100.0 * sum(clean_ok) / n:.2f}%\n")

        # transferred: the saved patches, replayed through multi4
        for w in m["weights"]:
            name, by_loc, esc = f"transfer_w{w}", {}, {}
            for loc, byw in cap["by"].items():
                top, left = (int(v) for v in loc.split(","))
                adv = x_all.clone()
                adv[:, :, top:top + size, left:left + size] = (
                    byw[str(float(w))]["patch"].to(device))
                by_loc[loc] = torch.cat(
                    [attacks.accuracy(clf, adv[i:i + bs], y_all[i:i + bs])
                     for i in range(0, n, bs)]).tolist()
                esc[loc] = 100.0 * float(torch.cat(
                    [union_mask_escaped(clf, adv[i:i + bs], cfg, top, left, size)
                     for i in range(0, n, bs)]).float().mean())
            sources[name] = {"correct": by_loc, "escape_pct": esc, "transferred": True}

        # fresh: optimised against multi4 itself
        for w in [0.0] + aux_weights:
            name = "clf" if w == 0.0 else f"union_w{w}"
            by_loc, esc = {}, {}
            for top, left in locations:
                aux = union_evasion_loss(clf, cfg, top, left, size, device) if w else None
                ok, escaped = [], []
                for s in range(0, n, bs):
                    take = torch.arange(s, min(s + bs, n), device=device)
                    config.seed_everything(paired_seed(size, top, left, s))
                    adv = attacks.patch_autopgd(clf, x_all[take], y_all[take],
                                                top=top, left=left, size=size,
                                                steps=args.steps, loss=m["loss"],
                                                init=m["init"], schedule=m["schedule"],
                                                aux_loss=aux, aux_weight=w).detach()
                    ok += attacks.accuracy(clf, adv, y_all[take]).tolist()
                    escaped += union_mask_escaped(clf, adv, cfg, top, left,
                                                  size).tolist()
                by_loc[f"{top},{left}"] = ok
                esc[f"{top},{left}"] = 100.0 * sum(escaped) / len(escaped)
            sources[name] = {"correct": by_loc, "escape_pct": esc, "transferred": False}

    print(f"{'source':>14} {'mean':>7} {'lowest':>7} {'survives':>9} {'escape':>7}")
    print("-" * 50)
    summary = {}
    for name, rec in sources.items():
        accs = [100.0 * sum(c) / n for c in rec["correct"].values()]
        surv = survivors(clean_ok, rec["correct"])
        esc = sum(rec["escape_pct"].values()) / len(rec["escape_pct"])
        summary[name] = {"mean_accuracy": sum(accs) / len(accs),
                         "lowest_location_accuracy": min(accs),
                         "survives_all_locations": len(surv),
                         "survives_pct": 100.0 * len(surv) / n,
                         "mean_escape_pct": esc,
                         "by_location_accuracy": dict(zip(rec["correct"], accs)),
                         "escape_pct": rec["escape_pct"],
                         "transferred": rec["transferred"]}
        print(f"{name:>14} {summary[name]['mean_accuracy']:6.2f}% "
              f"{min(accs):6.2f}% {100.0 * len(surv) / n:8.2f}% {esc:6.2f}%")

    # suite union: clean-correct and correct under every source at every location
    all_loc = {f"{name}@{loc}": c
               for name, rec in sources.items() for loc, c in rec["correct"].items()}
    union = survivors(clean_ok, all_loc)
    # Dropping a source can only grow the survivor set; the growth is what it contributes.
    added = {}
    for name in sources:
        without = {k: v for k, v in all_loc.items() if not k.startswith(name + "@")}
        added[name] = sorted({indices[i] for i in survivors(clean_ok, without)}
                             - {indices[i] for i in union})
    print(f"\nsuite union over {len(all_loc)} settings: {100.0 * len(union) / n:.2f}% "
          f"({len(union)}/{n} images survive everything)")
    for name, imgs in added.items():
        print(f"  dropping {name:>14} would return {len(imgs)} images to the union")

    results.save(args.out, dict(
        purpose="adaptive pilot for the multi-window selection rule, with the "
                "transferred attacks retained in the same suite",
        source=f"{args.capture}.pt",
        caveat="development images, four locations, one defence configuration; "
               "escape is measured on the returned candidate, and a positive "
               "union margin is a sufficient condition that ties can break",
        indexing="correct and clean_ok are positions in the development draw; "
                 "index maps them to dataset indices",
        settings=dict(defence=cfg.as_dict(), steps=args.steps,
                      aux_weights=aux_weights, batch_size=bs,
                      **{k: m[k] for k in ("size", "loss", "init", "schedule",
                                           "locations", "weights")}),
        index=indices, clean_ok=clean_ok,
        clean_accuracy=100.0 * sum(clean_ok) / n,
        summary=summary,
        suite_union_pct=100.0 * len(union) / n,
        suite_union_images=sorted(indices[i] for i in union),
        images_returned_by_dropping=added,
        sources=sources,
        hours=round((time.time() - t_start) / 3600, 3)),
        n_eval_images=n, attack_steps=args.steps)


if __name__ == "__main__":
    main()
