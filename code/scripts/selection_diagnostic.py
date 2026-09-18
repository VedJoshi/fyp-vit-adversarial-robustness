"""Measure why the decoy attack moves arm 5's mask off the patch.

Records three things per image, for the classification-only attack and the
selection-aware one, on development images only:

  coverage    does the chosen block cover the patch fully, partly, or not at all
  cover rank  how many anchors score strictly higher than the best covering
              anchor. Ties make rank 0 compatible with a non-covering choice,
              so coverage is the authoritative column and rank is context
  reference   both of the above recomputed with the clean image's mean, which
              isolates how much of the ranking change comes from the shared
              scoring reference moving rather than from the patch tokens' own
              scores falling

Only the centred rule frozen at block 0 is supported; the coverage geometry
and the auxiliary loss both assume it.

The clean mean is a diagnostic substitution. No deployable defense has the
clean counterpart, and the comparison bears on ranking, not on why the image
is misclassified.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import attacks, config, data, hooks, metrics, models, results, rsa  # noqa: E402
from gate_a import paired_seed  # noqa: E402
from selection_attack import evasion_loss  # noqa: E402

FULL, PARTIAL, MISS = "full", "partial", "miss"


def block_tokens(mod, anchor: int, window: int, rule: str) -> set[int]:
    """Tokens masked when `anchor` wins, via the defense's own corner rule."""
    g = config.GRID
    scores = torch.full((1, g * g), -1.0)
    scores[0, anchor] = 1.0
    r0, c0, h, w = mod._window_corner(scores, window, g, rule)
    return {int(r) * g + int(c)
            for r in range(int(r0), int(r0) + h)
            for c in range(int(c0), int(c0) + w)}


def coverage_by_anchor(mod, top: int, left: int, size: int, cfg) -> list[str]:
    """FULL/PARTIAL/MISS for every anchor at one location."""
    foot = set(metrics.patch_token_indices(top, left, size, include_cls_offset=False))
    out = []
    for t in range(config.GRID ** 2):
        hit = foot & block_tokens(mod, t, cfg.window, cfg.window_rule)
        out.append(FULL if hit == foot else (PARTIAL if hit else MISS))
    return out


def block0_values(clf, x: torch.Tensor) -> torch.Tensor:
    """Block-0 value vectors before any replacement; shape (B, H, N, D)."""
    with torch.no_grad(), hooks.AttentionCapture(clf, layers=[0], store=("v",)) as cap:
        clf(x)
    return cap.data[0].v


def read(scores: torch.Tensor, covers: torch.Tensor,
         cover_cat: list[str]) -> list[dict]:
    """Per image: chosen anchor, its coverage, and the best covering anchor's rank."""
    chosen = scores.argmax(1)
    best_cover = scores.masked_fill(~covers, float("-inf")).max(1).values
    rank = (scores > best_cover.unsqueeze(1)).sum(1)
    return [{"chosen": int(c), "coverage": cover_cat[int(c)], "cover_rank": int(r)}
            for c, r in zip(chosen, rank)]


def summarise(recs: list[dict], n: int) -> dict:
    cov = [r["coverage"] for r in recs]
    rank = [r["cover_rank"] for r in recs]
    return {
        "full_pct": 100.0 * sum(c == FULL for c in cov) / n,
        "partial_pct": 100.0 * sum(c == PARTIAL for c in cov) / n,
        "miss_pct": 100.0 * sum(c == MISS for c in cov) / n,
        "cover_rank_median": statistics.median(rank),
        "cover_rank_le1_pct": 100.0 * sum(r <= 1 for r in rank) / n,
        "cover_rank_le2_pct": 100.0 * sum(r <= 2 for r in rank) / n,
        "cover_rank_le4_pct": 100.0 * sum(r <= 4 for r in rank) / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-images", type=int, default=128)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="the frozen development setting; arm 0 is always run beside it")
    ap.add_argument("--locations", default="60,60 0,0 100,20 140,160")
    ap.add_argument("--loss", default="dlr")
    ap.add_argument("--init", default="uniform")
    ap.add_argument("--schedule", default="counter")
    ap.add_argument("--score-frame", default="global")
    ap.add_argument("--renorm", default="uniform")
    ap.add_argument("--backward", default="intended")
    ap.add_argument("--out", default="selection_diagnostic_dev")
    ap.add_argument("--capture", action="store_true",
                    help="also write <out>.pt with patch pixels, all 196 scores under "
                         "both references, and the reference vectors")
    args = ap.parse_args()

    locations = [tuple(int(v) for v in loc.split(",")) for loc in args.locations.split()]
    weights = [0.0, args.aux_weight]

    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    ds, wnids, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    cfg = rsa.RSAConfig.for_patch(args.size, renormalise=args.renorm,
                                  window_rule="centred",
                                  score_frame=args.score_frame, backward=args.backward,
                                  freeze_mask_after=0)

    sub = data.dev_subset(ds, args.n_images)
    xs, ys = [], []
    for x, y in data.make_loader(sub):
        xs.append(x.to(device))
        ys.append(y.to(device))
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    n, bs = len(x_all), args.batch_size
    # dev_subset sorts its draw and make_loader does not shuffle, so position i is indices[i]
    indices = list(sub.indices)

    print(f"\nselection diagnostic on {n} development images, disjoint from the "
          f"{config.SCALE.n_eval_images}-image protocol sample")
    print(f"{len(locations)} locations, {args.steps} steps, window {cfg.window} "
          f"({cfg.window_rule}), freeze {cfg.freeze_mask_after}, weights {weights}\n")

    t_start = time.time()
    per_loc: dict[str, dict] = {}
    with rsa.enable(clf, cfg) as handle:
        mod = handle.modules[0]
        clean_ok = torch.cat([attacks.accuracy(clf, x_all[i:i + bs], y_all[i:i + bs])
                              for i in range(0, n, bs)]).tolist()
        print(f"clean accuracy {100.0 * sum(clean_ok) / n:.2f}% under the defense\n")
        cap_all: dict = {}
        if args.capture:
            cs, cm = [], []
            for i in range(0, n, bs):
                v = block0_values(clf, x_all[i:i + bs])
                cs.append(metrics.rsa_token_scores(v, frame=cfg.score_frame).cpu())
                cm.append(metrics.score_frame_mean(v, frame=cfg.score_frame)
                          .squeeze(2).squeeze(1).cpu())
            cap_all = {"index": torch.tensor(indices), "clean_ok": torch.tensor(clean_ok),
                       "clean_scores": torch.cat(cs), "clean_mu": torch.cat(cm), "by": {}}
        for top, left in locations:
            covers = metrics.centred_coverage(top, left, args.size, cfg.window).to(device)
            cover_cat = coverage_by_anchor(mod, top, left, args.size, cfg)
            assert all(cover_cat[t] == FULL for t in covers.nonzero().flatten().tolist()), \
                "centred_coverage and the corner rule disagree on which anchors cover"
            print(f"({top:>3},{left:>3})  {int(covers.sum())} covering anchors")

            by_w: dict[str, dict] = {}
            clean_recs: list[dict] = []
            for w in weights:
                aux = evasion_loss(clf, cfg, top, left, args.size, device) if w else None
                own, ref, wrong = [], [], set()
                grab: dict[str, list] = {"patch": [], "scores_own": [], "scores_ref": [],
                                         "mu_adv": []}
                for s in range(0, n, bs):
                    take = torch.arange(s, min(s + bs, n), device=device)
                    config.seed_everything(paired_seed(args.size, top, left, s))
                    adv = attacks.patch_autopgd(clf, x_all[take], y_all[take],
                                                top=top, left=left, size=args.size,
                                                steps=args.steps, loss=args.loss,
                                                init=args.init, schedule=args.schedule,
                                                aux_loss=aux, aux_weight=w).detach()
                    wrong |= set(take[~attacks.accuracy(clf, adv, y_all[take])].tolist())

                    v_adv = block0_values(clf, adv)
                    v_cln = block0_values(clf, x_all[take])
                    mu_cln = metrics.score_frame_mean(v_cln, frame=cfg.score_frame)
                    s_adv = metrics.rsa_token_scores(v_adv, frame=cfg.score_frame)
                    s_ref = metrics.rsa_token_scores(v_adv, mu=mu_cln)

                    if args.capture:
                        patch = adv[:, :, top:top + args.size,
                                    left:left + args.size].contiguous()
                        rebuilt = x_all[take].clone()
                        rebuilt[:, :, top:top + args.size, left:left + args.size] = patch
                        assert torch.equal(rebuilt, adv),                             "the saved patch does not reconstruct the attacked input"
                        if s == 0:
                            assert torch.equal(clf(rebuilt).argmax(1), clf(adv).argmax(1)),                                 "the reconstructed input predicts differently"
                            assert torch.allclose(
                                metrics.rsa_token_scores(block0_values(clf, rebuilt),
                                                         frame=cfg.score_frame), s_adv),                                 "the reconstructed input scores differently"
                        grab["patch"].append(patch.cpu())
                        grab["scores_own"].append(s_adv.cpu())
                        grab["scores_ref"].append(s_ref.cpu())
                        grab["mu_adv"].append(
                            metrics.score_frame_mean(v_adv, frame=cfg.score_frame)
                            .squeeze(2).squeeze(1).cpu())

                    own += read(s_adv, covers, cover_cat)
                    ref += read(s_ref, covers, cover_cat)
                    if w == 0.0:
                        s_cln = metrics.rsa_token_scores(v_cln, frame=cfg.score_frame)
                        clean_recs += read(s_cln, covers, cover_cat)

                escaped = [i for i, r in enumerate(own) if r["coverage"] != FULL]
                restored = [i for i in escaped if ref[i]["coverage"] == FULL]
                improved = [i for i in escaped
                            if ref[i]["cover_rank"] < own[i]["cover_rank"]]
                per_image = [{"index": indices[i],
                              "wrong": i in wrong,
                              "clean_ok": bool(clean_ok[i]),
                              "chosen": own[i]["chosen"],
                              "coverage": own[i]["coverage"],
                              "cover_rank": own[i]["cover_rank"],
                              "ref_chosen": ref[i]["chosen"],
                              "ref_coverage": ref[i]["coverage"],
                              "ref_cover_rank": ref[i]["cover_rank"]}
                             for i in range(n)]
                by_w[str(w)] = {
                    "wrong_indices": sorted(indices[i] for i in wrong),
                    "attacked": summarise(own, n),
                    "clean_reference": summarise(ref, n),
                    "escaped": len(escaped),
                    "reference_restores_pct": 100.0 * len(restored) / max(len(escaped), 1),
                    "reference_improves_rank_pct": 100.0 * len(improved) / max(len(escaped), 1),
                    "per_image": per_image,
                }
                if args.capture:
                    cap_all["by"].setdefault(f"{top},{left}", {})[str(w)] = {
                        **{k: torch.cat(v) for k, v in grab.items()},
                        "wrong": torch.tensor([i in wrong for i in range(n)]),
                        "chosen": torch.tensor([r["chosen"] for r in own]),
                        "cover_rank": torch.tensor([r["cover_rank"] for r in own]),
                        "coverage": [r["coverage"] for r in own],
                    }
                a, r = by_w[str(w)]["attacked"], by_w[str(w)]["clean_reference"]
                print(f"    weight {w:>4}  full {a['full_pct']:5.1f}%  "
                      f"partial {a['partial_pct']:5.1f}%  miss {a['miss_pct']:5.1f}%  "
                      f"rank~{a['cover_rank_median']:>5.1f}  "
                      f"rank<=2 {a['cover_rank_le2_pct']:5.1f}%")
                print(f"                clean reference: full {r['full_pct']:5.1f}%  "
                      f"rank~{r['cover_rank_median']:>5.1f}  "
                      f"restores {by_w[str(w)]['reference_restores_pct']:5.1f}% of "
                      f"{len(escaped)} escapes  "
                      f"improves rank {by_w[str(w)]['reference_improves_rank_pct']:5.1f}%")
            by_w["clean_images"] = summarise(clean_recs, n)
            by_w["clean_images"]["per_image"] = [dict(index=indices[i], **clean_recs[i])
                                                 for i in range(n)]
            print(f"    clean images  full {by_w['clean_images']['full_pct']:5.1f}%  "
                  f"rank~{by_w['clean_images']['cover_rank_median']:>5.1f}")
            per_loc[f"{top},{left}"] = by_w

    print("\nReading it: cover_rank<=2 high under weight 0.5 is a reason to test top-k "
          "union. reference_restores high is a reason to test a stabilised scoring "
          "reference. Both high is a reason to test both, not evidence that two "
          "mechanisms contribute. Development images: report nothing here as a result.")

    results.save(args.out, dict(
        purpose="why the decoy attack moves arm 5's mask: coverage, rank, reference",
        indexing="every saved image identifier is a dataset index, never a position "
               "in the development draw; join on per_image.index and require clean_ok "
               "when counting images the attack newly broke",
        cover_rank="anchors scoring strictly higher than the best covering anchor; "
                   "ties let rank 0 coexist with a non-covering choice, so coverage "
                   "is authoritative and rank is context",
        settings=dict(size=args.size, steps=args.steps, weights=weights,
                      locations=[list(l) for l in locations],
                      window=cfg.window, window_rule=cfg.window_rule,
                      freeze_mask_after=cfg.freeze_mask_after,
                      score_frame=cfg.score_frame, backward=cfg.backward,
                      renormalise=cfg.renormalise, loss=args.loss, init=args.init,
                      schedule=args.schedule, batch_size=bs,
                      dev_disjoint_from=config.SCALE.n_eval_images),
        clean_ok=clean_ok,
        per_location=per_loc,
        capture=f"{args.out}.pt" if args.capture else None,
        hours=round((time.time() - t_start) / 3600, 3)),
        n_eval_images=n, attack_steps=args.steps)

    if args.capture:
        cap_all["meta"] = dict(size=args.size, steps=args.steps, weights=weights,
                               locations=[list(l) for l in locations],
                               window=cfg.window, window_rule=cfg.window_rule,
                               freeze_mask_after=cfg.freeze_mask_after,
                               score_frame=cfg.score_frame, backward=cfg.backward,
                               renormalise=cfg.renormalise, loss=args.loss,
                               init=args.init, schedule=args.schedule, batch_size=bs,
                               patch_slice="adv[:, :, top:top+size, left:left+size]",
                               note="scores_ref uses the attacked values with the clean "
                                    "reference; index is a dataset index")
        path = config.RESULTS_ROOT / f"{args.out}.pt"
        torch.save(cap_all, path)
        print(f"wrote {path}  [{path.stat().st_size / 1e6:.1f} MB]")


if __name__ == "__main__":
    main()
