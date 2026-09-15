"""Check selection coverage, evasion margins, gradients and disjoint sampling."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import config, metrics, rsa  # noqa: E402

G = config.GRID
LOCATIONS = [(60, 60), (0, 0), (0, 120), (200, 200), (100, 0), (140, 160), (47, 47)]

# 1. the helper against the defense it describes
cfg = rsa.RSAConfig.for_patch(20, window_rule="centred", freeze_mask_after=0)
mod = rsa.RSAAttention.__new__(rsa.RSAAttention)
mod.cfg = cfg
for top, left in LOCATIONS:
    cov = metrics.centred_coverage(top, left, 20, cfg.window)
    patch = set(metrics.patch_token_indices(top, left, 20, include_cls_offset=False))
    for t in range(G * G):
        scores = torch.full((1, G * G), -1.0)
        scores[0, t] = 1.0
        r0, c0, h, w = mod._window_corner(scores, cfg.window, G, "centred")
        block = {int(r) * G + int(c)
                 for r in range(int(r0), int(r0) + h)
                 for c in range(int(c0), int(c0) + w)}
        if bool(cov[t]) != patch.issubset(block):
            raise AssertionError(
                f"centred_coverage disagrees with rsa at ({top},{left}) token {t}: "
                f"helper says {bool(cov[t])}, the real block is {sorted(block)}")
print(f"1. coverage geometry       OK  matches rsa._window_corner on all {G * G} tokens "
      f"at {len(LOCATIONS)} locations")

# 2. Covering anchors can differ from patch tokens.
def cover_and_foot(top, left, size, window=3):
    cov = set(metrics.centred_coverage(top, left, size, window).nonzero().flatten().tolist())
    return cov, set(metrics.patch_token_indices(top, left, size, include_cls_offset=False))

cov, foot = cover_and_foot(112, 112, 10)
assert len(foot) == 1 and len(cov) == 9 and len(cov - foot) == 8, (len(foot), len(cov))
cov_corner, foot_corner = cover_and_foot(0, 0, 10)
assert len(foot_corner) == 1 and len(cov_corner) == 4, (len(foot_corner), len(cov_corner))
cov_off, foot_off = cover_and_foot(47, 47, 20)
assert len(foot_off) == 9 and len(cov_off) == 1 and len(foot_off - cov_off) == 8
assert metrics.centred_block_side(20, stride=None) == 5
cov_on, foot_on = cover_and_foot(60, 60, 20)
assert cov_on == foot_on and len(cov_on) == 4
print("2. coverage vs footprint   OK  8 covering tokens outside a 1-token patch; the "
      "corner clamp cuts 9 to 4; off-grid 8 of 9 footprint tokens do not cover")

# 3. Check anchor counts across the full grid.
from collections import Counter  # noqa: E402

from fyp import attacks  # noqa: E402

counts, outside_total, by_count = Counter(), 0, {}
for top, left in attacks.patch_locations(20):
    cov = set(metrics.centred_coverage(top, left, 20, cfg.window).nonzero().flatten().tolist())
    foot = set(metrics.patch_token_indices(top, left, 20, include_cls_offset=False))
    counts[len(cov)] += 1
    outside_total += len(cov - foot)
    by_count.setdefault(len(cov), (top, left))
    assert foot <= cov, f"a patch token that does not cover its own patch at ({top},{left})"
assert dict(counts) == {4: 81, 6: 36, 9: 4}, dict(counts)
assert outside_total == 92, outside_total
assert len(metrics.centred_coverage(100, 20, 20, cfg.window).nonzero()) == 6
assert len(metrics.centred_coverage(20, 20, 20, cfg.window).nonzero()) == 9
print(f"3. anchors across the grid OK  81 locations with 4 anchors, 36 with 6, 4 with 9; "
      f"{outside_total} anchors outside a footprint")

# 4. the objective's sign tracks the outcome it stands in for
covers = metrics.centred_coverage(60, 60, 20, cfg.window)
patch_interior = set(metrics.patch_token_indices(60, 60, 20, include_cls_offset=False))


def margin(scores):
    return (scores.masked_fill(covers, float("-inf")).max(1).values
            - scores.masked_fill(~covers, float("-inf")).max(1).values)


peak_on_patch = torch.zeros(1, G * G)
peak_on_patch[0, sorted(patch_interior)[0]] = 5.0
peak_off_patch = torch.zeros(1, G * G)
off = next(t for t in range(G * G) if not covers[t])
peak_off_patch[0, off] = 5.0
assert float(margin(peak_on_patch)) < 0, "a peak inside the patch must score negative"
assert float(margin(peak_off_patch)) > 0, "a peak outside must score positive"
both = torch.cat([peak_on_patch, peak_off_patch])
assert (~covers[both.argmax(1)] == (margin(both) > 0)).all(), \
    "the margin's sign must agree with whether the real argmax escapes"
cov_a = metrics.centred_coverage(112, 112, 10, 3)
outside_but_covered = next(t for t in range(G * G)
                           if cov_a[t] and t not in cover_and_foot(112, 112, 10)[1])
peak = torch.zeros(1, G * G)
peak[0, outside_but_covered] = 5.0
m = (peak.masked_fill(cov_a, float("-inf")).max(1).values
     - peak.masked_fill(~cov_a, float("-inf")).max(1).values)
assert float(m) < 0, "a peak outside the patch that is still covered must not score as a win"
tie = torch.zeros(1, G * G)
tie[0, sorted(patch_interior)[0]] = 5.0
tie[0, off] = 5.0
assert float(margin(tie)) == 0.0, "equal maxima must give a zero margin"
print("4. evasion objective       OK  sign agrees with the real argmax away from ties; "
      "a covered token outside the patch is not an escape")

# 5. The evasion gradient reaches the patch pixels.
from _stub_vit import _StubViT  # noqa: E402
from selection_attack import evasion_loss, selection_scores  # noqa: E402

torch.manual_seed(0)
stub = _StubViT().eval()
top, left, size = 100, 20, 20            # a 6-anchor location, not a 4-anchor one
aux = evasion_loss(stub, cfg, top, left, size, torch.device("cpu"))
x = torch.rand(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, requires_grad=True)
val = aux(x)
assert val.shape == (2,), val.shape
assert torch.isfinite(val).all(), "the objective must be finite"
val.sum().backward()
g = x.grad
assert g is not None and torch.isfinite(g).all(), "the gradient must exist and be finite"
support = attacks.patch_mask(x.shape, top, left, size, x.device).bool().expand_as(x)
assert (g[support] != 0).any(), "no gradient reaches the pixels the attacker may change"
assert float(g[support].abs().sum()) > 0
with torch.no_grad():
    s_cap = selection_scores(stub, x.detach(), cfg)
assert s_cap.shape == (2, G * G), s_cap.shape
print("5. gradient path           OK  evasion_loss is finite and differentiable into the "
      "patch, on a 6-anchor location")

# 6. development images are disjoint from the protocol sample
import random  # noqa: E402

held = set(random.Random(config.SEED).sample(range(5000), 512))
pool = [i for i in range(5000) if i not in held]
dev = set(random.Random(config.SEED + 1).sample(pool, 128))
assert not (dev & held), f"{len(dev & held)} development images are in the eval sample"
print("6. dev/eval disjoint       OK  128 development images, 0 shared with the 512")

print("\nall selection checks passed")
