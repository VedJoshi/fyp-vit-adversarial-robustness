"""E2: replay saved E1 patches under the defence that selects on the same score.

    python code/scripts/camouflage_replay.py RUN [RUN ...]

RUN is an E1 output name: results/RUN.json and results/RUN_patches.pt. Writes
results/RUN_e2.json in E1's layout, so camouflage_summary.py stages it the same way.
A success is an image the defence classifies correctly without the patch and wrongly with it.
Forward passes only.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fyp import attacks, config, data, models, results, rsa  # noqa: E402
from finder_attack import install_finder  # noqa: E402

DEFENCE = {"anomaly": "arm 5 (centred, frozen at block 0, anomaly anchor)",
           "knnfar1": "knnfar1 (centred, frozen at block 0, knnfar1 anchor)",
           "tv": "pixel finder (centred, frozen at block 0, tv anchor)"}


class Defended:
    """The plain model with one matching defence installed, or none."""

    def __init__(self, clf, score: str | None, size: int):
        self.clf, self.score = clf, score
        self.cfg = rsa.RSAConfig.for_patch(size, window_rule="centred", freeze_mask_after=0,
                                           select_rule="knnfar1" if score == "knnfar1"
                                           else "anomaly")

    def __enter__(self):
        if self.score is None:
            return self
        self.handle = rsa.enable(self.clf, self.cfg).__enter__()
        self.hook = (install_finder(self.clf, self.handle, "tv", self.cfg.window)
                     if self.score == "tv" else None)
        return self

    def __exit__(self, *exc):
        if self.score is not None:
            if self.hook is not None:
                self.hook.remove()
            self.handle.__exit__(*exc)

    @torch.no_grad()
    def correct(self, x, y, bs):
        return torch.cat([attacks.accuracy(self.clf, x[i:i + bs], y[i:i + bs])
                          for i in range(0, len(x), bs)])


def paste(x, pos, patch, top, left):
    adv = x[pos].clone()
    s = patch.shape[-1]
    adv[:, :, top:top + s, left:left + s] = patch.to(x.device)
    return adv


def replay(run: str, clf, ds, device, bs=64) -> dict:
    rec = json.loads((config.RESULTS_ROOT / f"{run}.json").read_text(encoding="utf-8"))
    patches = torch.load(config.RESULTS_ROOT / f"{run}_patches.pt", weights_only=True)
    st = rec["settings"]
    sub = (data.test_subset(ds) if st["image_set"] == "test"
           else data.dev_subset(ds, len(rec["index"])))
    assert list(sub.indices) == rec["index"], "image draw differs from the E1 run"
    xs, ys = zip(*[(x.to(device), y.to(device)) for x, y in data.make_loader(sub)])
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    size = st["size"]
    plain_ok = ~torch.isin(torch.arange(len(x_all)), torch.tensor(rec["clean_wrong"]))

    out = dict(purpose="E2: E1 patches replayed under the matching defence",
               source=run, settings=st, index=rec["index"], defences=DEFENCE,
               clean_correct={}, unconstrained_by_defence={}, capped={},
               plain_mismatches={})
    for score in DEFENCE:
        with Defended(clf, score, size) as d:
            ok = d.correct(x_all, y_all, bs).cpu()
            out["clean_correct"][score] = ok.nonzero().flatten().tolist()
            for key, (pos, patch) in patches.items():
                loc, *rest = key.split("/")
                top, left = map(int, loc.split(","))
                if rest != ["none"] and rest[0] != score:
                    continue
                hit = ~d.correct(paste(x_all, pos, patch, top, left), y_all[pos], bs).cpu()
                hit &= ok[pos]
                broken = pos[hit].tolist()
                if rest == ["none"]:
                    out["unconstrained_by_defence"].setdefault(score, {})[loc] = {"wrong": broken}
                else:
                    cell = out["capped"].setdefault(f"{loc}/{rest[0]}/{rest[1]}",
                                                    {"success": [], "by_weight": {}})
                    cell["by_weight"][rest[2]] = broken
                    cell["success"] = sorted(set(cell["success"]) | set(broken))
        print(f"  {score:<8} defended clean {100 * ok.float().mean():.2f}%")

    # Pasting must reproduce E1 on the plain model: same images broken, per weight.
    with Defended(clf, None, size) as d:
        for key, (pos, patch) in patches.items():
            loc, *rest = key.split("/")
            if rest == ["none"]:
                continue
            top, left = map(int, loc.split(","))
            wrong = ~d.correct(paste(x_all, pos, patch, top, left), y_all[pos], bs).cpu()
            mine = set(pos[wrong & plain_ok[pos]].tolist())
            e1 = set(rec["capped"][f"{loc}/{rest[0]}/{rest[1]}"]["by_weight"][rest[2]])
            if mine != e1:
                out["plain_mismatches"][key] = sorted(mine ^ e1)
    print(f"  plain-model replay differs from E1 in {len(out['plain_mismatches'])} cells")
    return out


def self_check():
    x = torch.zeros(3, 3, 8, 8)
    adv = paste(x, torch.tensor([2, 0]), torch.ones(2, 3, 2, 2), 1, 4)
    assert adv.shape == (2, 3, 8, 8) and adv[:, :, 1:3, 4:6].eq(1).all() and adv.sum() == 2 * 3 * 2 * 2
    assert x.sum() == 0


def main():
    self_check()
    config.seed_everything()
    device = config.get_device()
    ds, _, class_indices = data.build_dataset()
    clf, _ = models.load_model(class_indices=class_indices, device=device)
    for run in sys.argv[1:]:
        print(run)
        out = replay(run, clf, ds, device)
        results.save(f"{run}_e2", out, n_eval_images=len(out["index"]),
                     attack_steps=out["settings"]["steps"])


if __name__ == "__main__":
    main()
