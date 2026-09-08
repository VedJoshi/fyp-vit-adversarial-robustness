"""Compare pre-softmax logit gaps across ViT architectures on the same images.

Jain & Dutta (CVPR 2024) Figure 2 reports gaps of roughly 0-45 at block 1, 0-200 at
block 6 and 0-1000 at block 12 for a normally-trained ViT-B/16 on ImageNet-100. The
float32 threshold is `metrics.FLOAT32_UNDERFLOW_GAP` = 103: past it every non-maximal
softmax entry underflows to exactly zero, the attention row is exactly one-hot and its
gradient is exactly zero. That is the justification for Adaptive Attention Scaling.

Measurement 1 in `M3_instrumentation.ipynb` found gaps of 3-10 on DeiT-S, an order of
magnitude below the threshold and two below their block-12 range. Two explanations were
open: the `1/sqrt(d_k)` scaling, and the model together with its training recipe. Both
models use 16x16 patches at 224 px and head_dim 64 (768/12 and 384/6), so the scaling is
identical across them and this run rules it out. What it leaves is one architecture
measured on two checkpoints that differ in pretraining data, fine-tuning and
regularisation at once, so it is evidence that architecture alone does not explain the
difference, not an isolation of which of those does.

Every model is measured on the same images, a seeded draw of `--batch` images from the
evaluation split.

Writes `results/m1_logit_gap_by_model.json` and
`results/figures/m1_logit_gaps_by_model.png`.

Usage::

    python scripts/logit_gap_by_model.py --batch 64
    python scripts/logit_gap_by_model.py --batch 64 --models vit_base_patch16_224

`--batch` is both the sample size and the batch size: the capture holds
`depth x B x heads x N x N` floats, which is 1.4 GB for ViT-B/16 at 64 images.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fyp import config, data, hooks, metrics, models, results  # noqa: E402

DEFAULT_MODELS = ("deit_small_patch16_224", "vit_base_patch16_224")

# Jain & Dutta Figure 2, read off the plotted ranges for a normally-trained ViT-B/16.
JAIN_DUTTA_FIG2 = {"block_1": [0, 45], "block_6": [0, 200], "block_12": [0, 1000]}


def measure(model_name: str, x: torch.Tensor, class_indices: list[int]) -> dict:
    """Per-layer logit-gap statistics and underflow fraction for one model.

    The capture is verified on this architecture before the gaps are read, since the
    gap is computed from reconstructed pre-softmax logits rather than from a tensor
    timm exposes directly.
    """
    clf, info = models.load_model(class_indices=class_indices, model_name=model_name)

    verification = hooks.verify_all_layers(clf, x[:2])

    with hooks.AttentionCapture(clf, store=("logits",)) as cap:
        with torch.no_grad():
            clf(x)

        layers = []
        for L in range(info["depth"]):
            g = metrics.logit_gap(cap[L].logits, reduce="none")
            s = metrics.logit_span(cap[L].logits)
            layers.append({
                "layer": L,
                "median": g.median().item(),
                "p99": g.flatten().quantile(0.99).item(),
                "max": g.max().item(),
                "underflow_pct": 100 * metrics.underflow_fraction(cap[L].logits).mean().item(),
                # top1 - min, the weaker underflow event: the smallest entry of the row
                # vanishing rather than every non-maximal entry. It crosses the threshold
                # first, so it bounds the partial gradient loss that top1 - top2 misses.
                "span_median": s.median().item(),
                "span_max": s.max().item(),
                "partial_underflow_pct":
                    100 * metrics.underflow_fraction(cap[L].logits, mode="partial").mean().item(),
            })

    peak_gap = max(r["max"] for r in layers)
    peak_span = max(r["span_max"] for r in layers)
    # The training recipe is the remaining explanation for any difference, so record
    # which checkpoint timm actually resolved rather than just the architecture name.
    cfg = getattr(clf.model, "pretrained_cfg", {}) or {}
    out = {
        "model": model_name,
        "checkpoint_tag": cfg.get("tag"),
        "checkpoint_hf_id": cfg.get("hf_hub_id"),
        "depth": info["depth"],
        "num_heads": info["num_heads"],
        "head_dim": info["head_dim"],
        "scale": info["scale"],
        "layers": layers,
        "max_gap_any_layer": peak_gap,
        "max_underflow_pct_any_layer": max(r["underflow_pct"] for r in layers),
        "max_span_any_layer": peak_span,
        "max_partial_underflow_pct_any_layer": max(r["partial_underflow_pct"] for r in layers),
        "reaches_underflow_regime": peak_gap > metrics.FLOAT32_UNDERFLOW_GAP,
        "reaches_partial_underflow_regime": peak_span > metrics.FLOAT32_UNDERFLOW_GAP,
        "gap_grows_with_depth": layers[-1]["median"] > layers[0]["median"],
        "capture_verification": verification["worst"],
    }

    del clf, cap
    torch.cuda.empty_cache()
    return out


def plot(records: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for rec in records:
        xs = [r["layer"] for r in rec["layers"]]
        ax.plot(xs, [r["median"] for r in rec["layers"]], "o-", label=f"{rec['model']} median")
        ax.plot(xs, [r["max"] for r in rec["layers"]], "^:", alpha=0.7,
                label=f"{rec['model']} max")

    ax.axhline(metrics.FLOAT32_UNDERFLOW_GAP, color="crimson", lw=1.5,
               label=f"float32 underflow ({metrics.FLOAT32_UNDERFLOW_GAP:.0f})")
    ax.axhspan(JAIN_DUTTA_FIG2["block_12"][0], JAIN_DUTTA_FIG2["block_12"][1],
               color="grey", alpha=0.12, label="Jain & Dutta Fig. 2, block 12 range")
    ax.set_yscale("log")
    ax.set_xlabel("block")
    ax.set_ylabel("pre-softmax logit gap")
    ax.set_title("Logit gap by depth and architecture")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--batch", type=int, default=config.SCALE.batch_size,
                    help="images to measure; capture holds depth x B x H x N x N floats")
    args = ap.parse_args()

    config.seed_everything()
    print(config.describe())

    ds, _, class_indices = data.build_dataset()
    # The sample is drawn at exactly the size that gets measured. `eval_subset` sorts
    # its draw so the loader reads the dataset in order, and ImageFolder is ordered by
    # class, so taking the first batch of a larger subset would take the lowest indices
    # and cover only the first few class folders. A peak statistic over a sample that
    # misses most classes is not a peak over the model.
    sub = data.eval_subset(ds, args.batch)
    loader = data.make_loader(sub, batch_size=args.batch)
    x, y = next(iter(loader))
    x = x.to(config.get_device())
    n_classes_covered = int(torch.unique(y).numel())
    print(f"\nmeasuring {args.models} on the same {x.shape[0]} images, "
          f"drawn from {len(ds)} and covering {n_classes_covered} classes\n")

    records = []
    for name in args.models:
        rec = measure(name, x, class_indices)
        records.append(rec)
        print(f"{rec['model']}  depth {rec['depth']}, heads {rec['num_heads']}, "
              f"head_dim {rec['head_dim']}, scale {rec['scale']:.4f}")
        print(f"{'layer':>5} {'median':>9} {'p99':>9} {'max':>9} {'underflow %':>12}"
              f" {'span max':>9} {'partial %':>10}")
        for r in rec["layers"]:
            print(f"{r['layer']:5d} {r['median']:9.2f} {r['p99']:9.2f} {r['max']:9.2f} "
                  f"{r['underflow_pct']:12.2f} {r['span_max']:9.2f} "
                  f"{r['partial_underflow_pct']:10.2f}")
        print(f"  checkpoint {rec['checkpoint_tag']}, capture verified to "
              f"{max(rec['capture_verification'].values()):.3e}")
        print(f"  max gap over all layers {rec['max_gap_any_layer']:.2f} against a threshold of "
              f"{metrics.FLOAT32_UNDERFLOW_GAP:.0f}, underflow regime reached: "
              f"{rec['reaches_underflow_regime']}")
        print(f"  max span (top1 - min) over all layers {rec['max_span_any_layer']:.2f}, "
              f"partial underflow regime reached: {rec['reaches_partial_underflow_regime']}")
        print(f"  median gap grows with depth: {rec['gap_grows_with_depth']} "
              f"(block 0 {rec['layers'][0]['median']:.2f} -> "
              f"block {rec['depth'] - 1} {rec['layers'][-1]['median']:.2f}); "
              f"Jain & Dutta Fig. 2 grows\n")

    plot(records, config.FIGURES_ROOT / "m1_logit_gaps_by_model.png")

    results.save("m1_logit_gap_by_model", n_eval_images=int(x.shape[0]),
                 attack_steps=None, payload={
        # The invocation, so the record can be reproduced from the record. The default
        # batch is the scale's, which is not the size this was first run at.
        "invocation": f"python scripts/logit_gap_by_model.py --batch {args.batch} "
                      f"--models {' '.join(args.models)}",
        "n_images": int(x.shape[0]),
        "n_classes_covered": n_classes_covered,
        "n_classes_available": config.N_CLASSES,
        "sample": "data.eval_subset(ds, n_images) - a seeded draw of exactly the "
                  "measured size, not the front of a larger subset",
        "underflow_threshold": metrics.FLOAT32_UNDERFLOW_GAP,
        "underflow_events": {
            "complete": "top1 - top2 > threshold: every non-maximal entry is exactly 0",
            "partial": "top1 - min > threshold: the smallest entry is exactly 0",
        },
        "jain_dutta_fig2_vit_b16": JAIN_DUTTA_FIG2,
        "models": {rec["model"]: rec for rec in records},
        "any_model_reaches_underflow": any(r["reaches_underflow_regime"] for r in records),
        "any_model_reaches_partial_underflow":
            any(r["reaches_partial_underflow_regime"] for r in records),
    })


if __name__ == "__main__":
    main()
