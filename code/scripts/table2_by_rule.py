"""Clean accuracy under RSA at each patch size, under every reading of the window rule.

Step 6d.2. RSA's Table 2 is the clean-accuracy cost of the defense; `ceil+1` with
`renormalise="zero"` matches it within 0.78 points at every size, and that agreement is
what made it the first reading implemented. A reading that fixes Table 1 is only useful
if it does not break Table 2, so this records the deviation for all six before the
location sweep runs.

    python scripts/table2_by_rule.py                    # 512 images, all five sizes
    python scripts/table2_by_rule.py --n-images 128     # quicker
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from fyp import attacks, config, data, models, rsa

TABLE2_RSA = {0: 93.16, 10: 92.58, 20: 90.43, 30: 90.43, 40: 88.67, 50: 83.98}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="10,20,30,40,50")
    p.add_argument("--n-images", type=int, default=config.SCALE.n_eval_images)
    p.add_argument("--renorm", default=rsa.DEFAULT_RENORMALISATION,
                   choices=rsa.RENORMALISATIONS)
    p.add_argument("--score-frame", default=rsa.DEFAULT_SCORE_FRAME,
                   choices=rsa.SCORE_FRAMES)
    p.add_argument("--tag", default="by_rule")
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    config.seed_everything()
    device = config.get_device()
    print(config.describe())

    ds, wnids, class_indices = data.build_dataset()
    sub = data.eval_subset(ds, n=args.n_images)
    loader = data.make_loader(sub)
    clf, info = models.load_model(class_indices=class_indices, device=device)

    batches = []
    seen = 0
    for xb, yb in loader:
        if seen >= args.n_images:
            break
        take = min(xb.shape[0], args.n_images - seen)
        batches.append((xb[:take].to(device), yb[:take].to(device)))
        seen += take
    print(f"clean accuracy under RSA, {seen} images, renormalise {args.renorm!r}\n")

    undefended = 0
    for xb, yb in batches:
        with torch.no_grad():
            undefended += (clf(xb).argmax(1) == yb).sum().item()
    print(f"undefended: {100 * undefended / seen:.2f}%\n")

    header = "rule      " + "".join(f"{s:>8d}px" for s in sizes) + "   max dev"
    print(header)
    print("Table 2   " + "".join(f"{TABLE2_RSA[s]:>10.2f}" for s in sizes))
    print("-" * len(header))

    record = {"n_images": seen, "renormalise": args.renorm,
              "score_frame": args.score_frame, "sizes": sizes,
              "undefended": 100 * undefended / seen, "table2_rsa": TABLE2_RSA,
              "by_rule": {}, "model": info}
    t0 = time.time()
    for rule in rsa.WINDOW_RULES:
        accs, devs = [], []
        for size in sizes:
            cfg = rsa.RSAConfig.for_patch(size, renormalise=args.renorm, window_rule=rule,
                                          score_frame=args.score_frame)
            correct = 0
            with rsa.enable(clf, cfg):
                for xb, yb in batches:
                    with torch.no_grad():
                        correct += (clf(xb).argmax(1) == yb).sum().item()
            acc = 100 * correct / seen
            accs.append(acc)
            devs.append(abs(acc - TABLE2_RSA[size]))
        record["by_rule"][rule] = {"clean_acc": accs, "abs_dev": devs, "max_dev": max(devs)}
        print(f"{rule:9s} " + "".join(f"{a:>10.2f}" for a in accs) + f"   {max(devs):>7.2f}")

    print(f"\n({time.time() - t0:.0f}s)")
    out = config.RESULTS_ROOT / f"table2_{args.tag}.json"
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
