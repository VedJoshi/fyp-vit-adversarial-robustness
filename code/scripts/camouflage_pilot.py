"""E1: attack success on the undefended model when the patch must look ordinary.

The patch tokens' selector score is capped at a quantile of the same image's
other tokens, so a patch that meets the cap ranks below that share of natural
tokens. One run is one stage and one start; camouflage_summary.py unions them.
Protocol: 02_direction/PROTOCOL_DRAFT_2026-11-21_camouflage.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import attacks, config, data, hooks, metrics, models, results  # noqa: E402
from finder_attack import SEED_STRIDE, finder_scores, smooth_init  # noqa: E402
from gate_a import paired_seed  # noqa: E402

SELECTORS = ("anomaly", "knnfar1", "tv")


def selector_scores(clf, x: torch.Tensor, selector: str) -> torch.Tensor:
    """(B, 196) per-token score; block-0 values for the first two, pixels for tv."""
    if selector == "tv":
        return finder_scores("tv", x)
    with hooks.AttentionCapture(clf, layers=[0], store=("v",), detach=False) as cap:
        clf(x)
    return metrics.select_scores(cap.data[0].v, rule=selector)


def excess(s: torch.Tensor, patch: torch.Tensor, q: float) -> torch.Tensor:
    """Highest patch-token score minus the q-quantile of the other tokens; <= 0 meets the cap."""
    return s[:, patch].max(1).values - s[:, ~patch].quantile(q, dim=1)


def patch_rank(s: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
    """Share of non-patch tokens scoring below the highest patch token."""
    return (s[:, ~patch] < s[:, patch].max(1, keepdim=True).values).float().mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-set", choices=("dev", "test"), default="dev",
                    help="test is the sealed held-out set; final runs only")
    ap.add_argument("--n-images", type=int, default=128, help="dev only")
    ap.add_argument("--smooth-init", action="store_true",
                    help="start from a blurred patch (init='given') instead of a uniform draw")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shifts every initial patch draw; a new value is a fresh start")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--selectors", default=" ".join(SELECTORS))
    ap.add_argument("--quantiles", default="1.0,0.9,0.75,0.5")
    ap.add_argument("--weights", default="1,10,100",
                    help="penalty weights on the cap violation, in clean score std units")
    ap.add_argument("--locations", default="60,60 100,20")
    ap.add_argument("--save-patches", action="store_true",
                    help="also write <out>_patches.pt for replay under a defence (E2/E3)")
    ap.add_argument("--out", default="probe_camouflage")
    args = ap.parse_args()

    quantiles = [float(v) for v in args.quantiles.split(",")]
    weights = [float(v) for v in args.weights.split(",")]
    locations = [tuple(int(v) for v in loc.split(",")) for loc in args.locations.split()]

    config.seed_everything()
    device = config.get_device()
    print(config.describe())
    ds, _, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    sub = data.test_subset(ds) if args.image_set == "test" else data.dev_subset(ds, args.n_images)
    xs, ys = zip(*[(x.to(device), y.to(device)) for x, y in data.make_loader(sub)])
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    n, bs = len(x_all), args.batch_size
    clean_ok = torch.cat([attacks.accuracy(clf, x_all[i:i + bs], y_all[i:i + bs])
                          for i in range(0, n, bs)])
    print(f"\n{n} {args.image_set} images, undefended clean accuracy "
          f"{100 * clean_ok.float().mean():.2f}%, {args.steps} steps, "
          f"{'smooth' if args.smooth_init else 'uniform'} start, seed offset {args.seed_offset}\n")

    def attack(top, left, aux=None, w=0.0, accept=None):
        out = []
        for s in range(0, n, bs):
            xb = x_all[s:s + bs]
            config.seed_everything(paired_seed(args.size, top, left,
                                               s + SEED_STRIDE * args.seed_offset))
            start = smooth_init(xb, top, left, args.size) if args.smooth_init else xb
            out.append(attacks.patch_autopgd(
                clf, start, y_all[s:s + bs], top=top, left=left, size=args.size,
                steps=args.steps, init="given" if args.smooth_init else "uniform",
                aux_loss=aux and (lambda adv, s=s: aux(adv, s)),
                aux_weight=w, accept=accept and (lambda adv, s=s: accept(adv, s))))
        return torch.cat(out)

    @torch.no_grad()
    def wrong(adv):
        return torch.cat([clf(adv[i:i + bs]).argmax(1) != y_all[i:i + bs]
                          for i in range(0, n, bs)])

    payload = dict(purpose="attack success against the allowed selector score of the patch",
                   settings=dict(size=args.size, steps=args.steps, quantiles=quantiles,
                                 weights=weights, locations=[list(l) for l in locations],
                                 image_set=args.image_set, smooth_init=args.smooth_init,
                                 seed_offset=args.seed_offset, draw=data.draw_provenance(ds),
                                 test_sha256=data.index_digest(sub.indices)
                                 if args.image_set == "test" else None),
                   index=list(sub.indices),
                   indexing="index[i] is the dataset index of position i",
                   clean_wrong=(~clean_ok).nonzero().flatten().tolist(),
                   unconstrained={}, capped={})
    # Patch pixels for E2/E3 replay: key -> (positions, (k, 3, size, size)); capped keys
    # keep only patches that meet the cap.
    patches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def keep(key, adv, sel, top, left):
        idx = sel.nonzero().flatten()
        patches[key] = (idx.cpu(), adv[idx, :, top:top + args.size, left:left + args.size].cpu())

    for top, left in locations:
        loc = f"{top},{left}"
        patch = torch.zeros(config.N_IMAGE_TOKENS, dtype=torch.bool, device=device)
        patch[metrics.patch_token_indices(top, left, args.size,
                                          include_cls_offset=False)] = True

        t0 = time.time()
        adv = attack(top, left)
        base = wrong(adv) & clean_ok
        if args.save_patches:
            keep(f"{loc}/none", adv, torch.ones_like(base), top, left)
        rec = {"wrong": base.nonzero().flatten().tolist(), "patch_rank": {}}
        line = f"({top:>3},{left:>3}) unconstrained  wrong {100 * base.float().mean():6.2f}%"
        for sel in args.selectors.split():
            with torch.no_grad():
                rank = torch.cat([patch_rank(selector_scores(clf, adv[i:i + bs], sel), patch)
                                  for i in range(0, n, bs)])
            rec["patch_rank"][sel] = rank.tolist()
            line += f"   {sel} rank median {float(rank.median()):.3f}"
        payload["unconstrained"][loc] = rec
        print(f"{line}   {time.time() - t0:.0f}s")

        for sel in args.selectors.split():
            with torch.no_grad():
                std = torch.cat([selector_scores(clf, x_all[i:i + bs], sel).std(1)
                                 for i in range(0, n, bs)])
            for q in quantiles:
                t0 = time.time()

                def viol(adv, s, q=q, sel=sel):
                    e = excess(selector_scores(clf, adv, sel), patch, q)
                    return e / std[s:s + len(adv)]

                def accept(adv, s):
                    with torch.no_grad():
                        return viol(adv, s) <= 0

                got = torch.zeros(n, dtype=torch.bool, device=device)
                by_w = {}
                for w in weights:
                    adv = attack(top, left, aux=lambda a, s: -viol(a, s).clamp_min(0),
                                 w=w, accept=accept)
                    with torch.no_grad():
                        ok = torch.cat([viol(adv[i:i + bs], i) <= 0 for i in range(0, n, bs)])
                    hit = wrong(adv) & ok & clean_ok
                    if args.save_patches:
                        keep(f"{loc}/{sel}/{q}/{w}", adv, ok, top, left)
                    by_w[w] = hit.nonzero().flatten().tolist()
                    got |= hit
                payload["capped"][f"{loc}/{sel}/{q}"] = {
                    "success": got.nonzero().flatten().tolist(), "by_weight": by_w}
                print(f"    {sel:<8} cap at quantile {q:<5} success {100 * got.float().mean():6.2f}%  "
                      f"per weight {[len(v) for v in by_w.values()]}   {time.time() - t0:.0f}s")
                results.save(args.out, payload, n_eval_images=n, attack_steps=args.steps)
                if args.save_patches:
                    torch.save(patches, config.RESULTS_ROOT / f"{args.out}_patches.pt")


def self_check():
    s = torch.tensor([[0.1, 0.9, 0.3, 0.2, 0.5]])
    patch = torch.tensor([False, True, False, False, False])
    assert excess(s, patch, 1.0) > 0 and patch_rank(s, patch) == 1.0
    s[0, 1] = 0.25
    assert excess(s, patch, 1.0) < 0 and excess(s, patch, 0.5).abs() < 1e-6
    assert patch_rank(s, patch) == 0.5


if __name__ == "__main__":
    self_check()
    main()
