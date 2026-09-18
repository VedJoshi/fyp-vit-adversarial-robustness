"""Save block-0 value vectors for the clean images and the saved attacks.

The existing capture holds the 196 anomaly scores but not the vectors they are
computed from, which limits offline screening to functions of the score map.
That family is exhausted: see rescore_screen.py.

This adds the vectors, so a rule that scores the same tokens differently can be
screened without a GPU. No attack is optimised; the attacked inputs are rebuilt
from the saved patch pixels, so this is forward passes only.

Vectors are stored as float16. At 128 images, 6 heads, 196 tokens and 64
channels that is about 19 MB per set, and one set per location and weight plus
one clean set.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import config, data, hooks, metrics, models, rsa  # noqa: E402


def block0_values(clf, x: torch.Tensor) -> torch.Tensor:
    """Block-0 value vectors before any replacement; (B, H, N, D), CLS included."""
    with torch.no_grad(), hooks.AttentionCapture(clf, layers=[0], store=("v",)) as cap:
        clf(x)
    return cap.data[0].v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="selection_capture_dev")
    ap.add_argument("--locations", default="",
                    help="subset of the capture's locations, e.g. '60,60 100,20'; "
                         "empty takes all of them")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="selection_values_dev")
    args = ap.parse_args()

    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    cap = torch.load(config.RESULTS_ROOT / f"{args.capture}.pt", weights_only=True)
    m = cap["meta"]
    size, n = m["size"], len(cap["index"])
    want = args.locations.split() if args.locations else list(cap["by"])
    missing = set(want) - set(cap["by"])
    if missing:
        raise SystemExit(f"locations {sorted(missing)} are not in {args.capture}")

    ds, wnids, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    cfg = rsa.RSAConfig.for_patch(size, renormalise=m["renormalise"],
                                  window_rule=m["window_rule"], score_frame=m["score_frame"],
                                  backward=m["backward"],
                                  freeze_mask_after=m["freeze_mask_after"])

    sub = data.dev_subset(ds, n)
    assert list(sub.indices) == cap["index"].tolist(),         "the development draw no longer matches the capture; rebuild the capture"
    xs, ys = [], []
    for x, y in data.make_loader(sub):
        xs.append(x.to(device))
        ys.append(y.to(device))
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    bs = args.batch_size

    def values(x):
        return torch.cat([block0_values(clf, x[i:i + bs]).half().cpu()
                          for i in range(0, n, bs)])

    t_start = time.time()
    out = {"index": cap["index"], "clean_ok": cap["clean_ok"],
           "label": y_all.cpu(), "clean_v": values(x_all), "by": {}}
    print(f"\nclean values {tuple(out['clean_v'].shape)} {out['clean_v'].dtype}")

    # The saved scores must come back out of the saved vectors, or the two disagree.
    got = metrics.rsa_token_scores(out["clean_v"].float(), frame=cfg.score_frame)
    err = (got - cap["clean_scores"]).abs().max()
    assert err < 2e-2, f"clean scores do not reproduce from the saved vectors ({err:.2e})"
    print(f"clean scores reproduce from the vectors, max abs error {float(err):.2e}")

    for loc in want:
        top, left = (int(v) for v in loc.split(","))
        for w in m["weights"]:
            adv = x_all.clone()
            adv[:, :, top:top + size, left:left + size] = (
                cap["by"][loc][str(float(w))]["patch"].to(device))
            v = values(adv)
            got = metrics.rsa_token_scores(v.float(), frame=cfg.score_frame)
            err = (got - cap["by"][loc][str(float(w))]["scores_own"]).abs().max()
            assert err < 2e-2, f"{loc} w={w}: scores do not reproduce ({err:.2e})"
            out["by"].setdefault(loc, {})[str(w)] = v
            print(f"({top:>3},{left:>3}) w={w:<4} {tuple(v.shape)}  "
                  f"score error {float(err):.2e}")

    out["meta"] = dict(source=f"{args.capture}.pt", locations=want,
                       dtype="float16", cls_included=True,
                       note="block-0 value vectors before RSA replacement; "
                            "rsa_token_scores on these reproduces the capture's scores",
                       **{k: m[k] for k in ("size", "steps", "weights", "window",
                                            "window_rule", "freeze_mask_after",
                                            "score_frame", "backward", "renormalise")})
    path = config.RESULTS_ROOT / f"{args.out}.pt"
    torch.save(out, path)
    print(f"\nwrote {path}  [{path.stat().st_size / 1e6:.1f} MB, "
          f"{(time.time() - t_start) / 60:.1f} min]")


if __name__ == "__main__":
    main()
