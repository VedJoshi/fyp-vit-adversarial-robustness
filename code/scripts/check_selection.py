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

# 7. Non-covering anchors can union to cover, so top-k cannot test anchors singly.
import itertools  # noqa: E402

from selection_diagnostic import FULL, MISS, PARTIAL, block_tokens, coverage_by_anchor  # noqa: E402

mod_c = rsa.RSAAttention.__new__(rsa.RSAAttention)
mod_c.cfg = cfg
bits = {t: sum(1 << i for i in block_tokens(mod_c, t, cfg.window, "centred"))
        for t in range(G * G)}
pair_locs, pair_total = 0, 0
for top, left in attacks.patch_locations(20):
    cov = set(metrics.centred_coverage(top, left, 20, cfg.window).nonzero().flatten().tolist())
    foot = sum(1 << i for i in
               metrics.patch_token_indices(top, left, 20, include_cls_offset=False))
    non = [bits[t] for t in range(G * G) if t not in cov]
    hit = sum(1 for a, b in itertools.combinations(non, 2) if foot & ~(a | b) == 0)
    pair_locs += hit > 0
    pair_total += hit
assert (pair_locs, pair_total) == (105, 972), (pair_locs, pair_total)
print(f"7. union coverage          OK  {pair_locs} of 121 locations admit a non-covering "
      f"pair whose union covers ({pair_total} pairs); a per-anchor test is wrong")

# 8. The diagnostic's three categories agree with centred_coverage and partition.
cat = coverage_by_anchor(mod_c, 100, 20, 20, cfg)
cov8 = metrics.centred_coverage(100, 20, 20, cfg.window)
assert [c == FULL for c in cat] == cov8.tolist(), "FULL must mean exactly centred_coverage"
assert set(cat) <= {FULL, PARTIAL, MISS} and cat.count(FULL) == 6
assert cat.count(PARTIAL) > 0 and cat.count(MISS) > 0
print(f"8. coverage categories     OK  6 full, {cat.count(PARTIAL)} partial, "
      f"{cat.count(MISS)} miss at (100,20)")

# 9. An externally supplied reference substitutes for the scores' own mean.
v9 = torch.randn(2, 6, G * G + 1, 8)
own9 = metrics.rsa_token_scores(v9, frame="global")
same9 = metrics.rsa_token_scores(v9, mu=metrics.score_frame_mean(v9, frame="global"))
assert torch.allclose(own9, same9), "passing the same mean must not change the scores"
other9 = metrics.rsa_token_scores(v9, mu=metrics.score_frame_mean(torch.randn_like(v9),
                                                                 frame="global"))
assert not torch.allclose(own9, other9), "a different mean must change the scores"
print("9. reference substitution  OK  rsa_token_scores(mu=) is exact and does change "
      "the ranking input")

# 10. The screening NMS reduces to argmax and to plain top-k; radius 3 is not disjoint.
from selection_screen import anchor_blocks, nms_select  # noqa: E402

torch.manual_seed(1)
sc = torch.randn(8, G * G)
assert torch.equal(nms_select(sc, 1, 3).squeeze(1), sc.argmax(1)), "k=1 must be argmax"
top4 = sc.argsort(dim=1, descending=True, stable=True)[:, :4]
assert torch.equal(nms_select(sc, 4, 1), top4), "radius 1 must be plain top-k"
blocks10 = anchor_blocks(3, "centred")
for row in nms_select(sc, 4, 3).tolist():
    assert len(set(row)) == 4, "selections must be distinct"
# Anchors 3 apart are disjoint inside the grid but not once a border shifts a block in.
assert blocks10[7 * G + 1] & blocks10[7 * G + 4] == 0, "interior blocks 3 apart are disjoint"
assert blocks10[0] & blocks10[3] != 0,     "a border block shifts inward, so radius 3 does not guarantee disjointness"
ties = torch.zeros(1, G * G)
ties[0, 5] = ties[0, 9] = 1.0
assert nms_select(ties, 2, 1).tolist() == [[5, 9]], "ties must break on the lower index"
print("10. screening selection    OK  k=1 is argmax, radius 1 is top-k, ties break low; "
      "radius 3 separates interior blocks but border blocks still overlap")

# 11. The defence's multi-window mask is the union of the screening rule's blocks.
def _fake(k, r):
    """An RSAAttention with just enough state to call select()."""
    m = rsa.RSAAttention.__new__(rsa.RSAAttention)
    m.cfg = rsa.RSAConfig(patch_px=20, window=3, window_rule="centred",
                          freeze_mask_after=0, n_windows=k, nms_radius=r)
    m.layer_idx, m.shared, m.oracle_mask = 0, {}, None
    return m

torch.manual_seed(2)
v11 = torch.randn(6, 4, G * G + 1, 8)
sc11 = metrics.rsa_token_scores(v11, frame="global")
for k, r in ((1, 3), (4, 3), (4, 1), (3, 2), (2, 5)):
    got = _fake(k, r).select(v11)
    assert not got[:, 0].any(), "CLS must never be masked"
    for b, row in enumerate(nms_select(sc11, k, r).tolist()):
        want = 0
        for a in row:
            want |= blocks10[a]
        have = sum(1 << t for t in got[b, 1:].nonzero().flatten().tolist())
        assert have == want, f"live mask differs from the screening union at k={k} r={r}"
    assert (got[:, 1:].sum(1) <= k * 9).all(), "the union cannot exceed k blocks"
base11 = _fake(1, 3).select(v11)
grew11 = _fake(2, 1).select(v11)
assert bool(((grew11 | base11) == grew11).all()),     "the first anchor is still the argmax, so k=2 must contain the baseline mask"
print("11. multi-window mask      OK  matches the screening union at 5 settings and "
      "contains the baseline block")

# 12. A positive union margin really does keep that token out of the greedy union.
from selection_attack import (anchor_contains, union_evasion_loss,  # noqa: E402
                             union_margins)

K12, R12 = 4, 3
contains12 = anchor_contains(3, "centred")
torch.manual_seed(3)
checked12 = positive12 = 0
for top, left in [(60, 60), (0, 0), (100, 20), (140, 160), (0, 200), (200, 200)]:
    foot12 = metrics.patch_token_indices(top, left, 20, include_cls_offset=False)
    sc12 = torch.randn(64, G * G)
    marg = union_margins(sc12, contains12, foot12, K12, R12)
    sel12 = rsa.nms_anchors(sc12, K12, R12)
    for b in range(len(sc12)):
        union = set()
        for a in sel12[b].tolist():
            union |= set(contains12[a].nonzero().flatten().tolist())
        for j, t in enumerate(foot12):
            checked12 += 1
            if marg[b, j] > 0:
                positive12 += 1
                assert t not in union,                     f"margin {float(marg[b, j]):.4f} > 0 but token {t} is still masked"
assert positive12 > 0, "the random draws produced no positive margin to test"
print(f"12. union margin sound     OK  {positive12} positive margins over {checked12} "
      f"token cases, every one exposed by the real greedy selection")

# 13. Windows that miss the patch singly can cover it together; the margin sees that.
top13, left13 = 60, 60
foot13 = metrics.patch_token_indices(top13, left13, 20, include_cls_offset=False)
pair13 = None
for a in range(G * G):
    for b in range(G * G):
        if max(abs(a // G - b // G), abs(a % G - b % G)) < R12:
            continue
        ta = set(contains12[a].nonzero().flatten().tolist())
        tb = set(contains12[b].nonzero().flatten().tolist())
        if not set(foot13) <= ta and not set(foot13) <= tb and set(foot13) <= ta | tb:
            pair13 = (a, b)
            break
    if pair13:
        break
assert pair13, "no jointly covering, individually non-covering pair at (60,60)"
covers13_all = metrics.centred_coverage(top13, left13, 20, 3)
# Four separated non-covering anchors, the first two jointly covering, all scoring above
# every covering anchor, so the retired rank-four margin is strictly positive here.
chosen13 = list(pair13)
for a in range(G * G):
    if len(chosen13) == K12:
        break
    if covers13_all[a] or any(max(abs(a // G - c // G), abs(a % G - c % G)) < R12
                              for c in chosen13):
        continue
    chosen13.append(a)
assert len(chosen13) == K12, chosen13
sc13 = torch.full((1, G * G), 0.5)                # every covering anchor sits at 0.5
sc13[0, chosen13] = torch.tensor([4.0, 3.0, 2.0, 1.0])
sel13 = rsa.nms_anchors(sc13, K12, R12)[0].tolist()
assert sel13 == chosen13, (sel13, chosen13)
union13 = set()
for a in sel13:
    union13 |= set(contains12[a].nonzero().flatten().tolist())
assert set(foot13) <= union13, "the constructed anchors must cover the patch together"
# The retired rank-k objective is satisfied: the fourth-best non-covering anchor scores
# strictly above the best covering one, and no chosen anchor covers the patch alone.
retired13 = sc13[0, chosen13].min() - sc13.masked_fill(~covers13_all, float("-inf")).max()
assert float(retired13) > 0, float(retired13)
assert not covers13_all[chosen13].any()
assert (union_margins(sc13, contains12, foot13, K12, R12) <= 0).all(),     "a jointly covering selection must not show a positive margin for any patch token"
print(f"13. joint coverage caught  OK  anchors {chosen13} cover (60,60) together, none "
      f"alone; the retired rank-{K12} margin reads +{float(retired13):.1f} and the union "
      f"margin stays non-positive")

# 14. The union objective is finite and differentiable into the patch pixels.
cfg14 = rsa.RSAConfig(patch_px=20, window=3, window_rule="centred",
                      freeze_mask_after=0, n_windows=K12, nms_radius=R12)
stub14 = _StubViT()
x14 = torch.rand(3, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, requires_grad=True)
aux14 = union_evasion_loss(stub14, cfg14, 60, 60, 20, x14.device)
val14 = aux14(x14)
assert val14.shape == (3,) and torch.isfinite(val14).all(), val14
(g14,) = torch.autograd.grad(val14.sum(), x14)
assert torch.isfinite(g14).all(), "the union objective produced a non-finite gradient"
inside14 = g14[:, :, 60:80, 60:80].abs().sum()
assert inside14 > 0, "no gradient reaches the permitted patch pixels"
assert g14.abs().sum() > inside14, "the score depends on tokens outside the patch too"
try:
    union_evasion_loss(stub14, rsa.RSAConfig.for_patch(20, window_rule="centred"),
                       60, 60, 20, x14.device)
except ValueError as e:
    assert "multi-window" in str(e), e
else:
    raise AssertionError("a single-window config should be rejected")
print(f"14. union gradient          OK  finite on {tuple(val14.shape)}, "
      f"{float(inside14):.3e} of gradient mass inside the patch, single-window rejected")

# 15. The degenerate steps do not bind and do not poison the margin.
sc15 = torch.zeros(1, G * G)
cov15 = contains12[:, foot13[0]].nonzero().flatten().tolist()
sc15[0, cov15] = -5.0                     # every covering anchor is the worst on the grid
m15 = union_margins(sc15, contains12, foot13, K12, R12)
assert torch.isfinite(m15).all() and float(m15[0, 0]) > 0, m15
# Radius 14 suppresses the whole grid after one pick, so later steps have no anchor left.
m15b = union_margins(sc15, contains12, foot13, K12, G)
assert torch.isfinite(m15b).all() and (m15b <= 0).all(),     "a step with no exposing anchor must drive the margin non-positive"
print("15. degenerate steps        OK  no covering anchor left does not bind, no "
      "exposing anchor left is non-positive, both finite")

# 16. Escape measured from the scores agrees with the mask the defence actually applied.
from selection_attack import union_mask_escaped  # noqa: E402

x16 = torch.rand(6, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
with rsa.enable(stub14, cfg14) as h16:
    with torch.no_grad():
        stub14(x16)
    applied = h16.masks()[0][:, 1:]                       # (B, N_img), CLS dropped
seen16 = 0
for top, left in [(60, 60), (0, 0), (140, 160)]:
    foot16 = metrics.patch_token_indices(top, left, 20, include_cls_offset=False)
    want = ~applied[:, foot16].all(1)
    got = union_mask_escaped(stub14, x16, cfg14, top, left, 20)
    assert torch.equal(want, got),         f"union_mask_escaped disagrees with the applied mask at ({top},{left})"
    seen16 += int(want.numel())
print(f"16. escape vs applied mask OK  agrees on {seen16} image-location cases against "
      f"the mask the defence installed")

# 17. A step that cannot cover the token must not cap the margin or cut its gradient.
#     Two anchors 3 apart on row 7; every anchor covering the target is suppressed by the
#     first pick, so step 2 onward cannot cover it and only step 1 may bind.
t17, exp17, cov17 = 7 * G + 7, 7 * G + 7, 8 * G + 8
contains17 = torch.zeros_like(contains12)
contains17[cov17, t17] = True                  # one anchor covers t, adjacent to the pick
sc17 = torch.zeros(1, G * G)
sc17[0, exp17] = 5.0
sc17[0, cov17] = 2.0                           # step 1 margin 3.0; the pick then
sc17 = sc17.requires_grad_(True)               # suppresses cov17, so steps 2-4 cannot bind
m17 = union_margins(sc17, contains17, [t17], K12, R12)
assert abs(float(m17[0, 0]) - 3.0) < 1e-6,     f"a non-binding step capped the margin: got {float(m17[0, 0])}, want 3.0"
(g17,) = torch.autograd.grad(m17.sum(), sc17)
assert g17.abs().sum() > 0, "the margin lost its gradient to a substituted constant"
assert float(g17[0, exp17]) > 0 and float(g17[0, cov17]) < 0,     "gradient must push the exposing anchor up and the covering anchor down"
# With no covering anchor anywhere, nothing binds: a positive constant, no gradient.
sc17b = torch.randn(1, G * G).requires_grad_(True)
free17 = torch.zeros_like(contains12)      # no anchor covers t at all
m17b = union_margins(sc17b, free17, [t17], K12, R12)
assert float(m17b[0, 0]) > 0 and float(torch.autograd.grad(
    m17b.sum(), sc17b, allow_unused=True)[0].abs().sum()) == 0,     "an always-exposed token should read positive with no gradient to give"
print(f"17. non-binding steps      OK  margin {float(m17[0, 0]):.1f} survives a step that "
      f"cannot cover, gradient reaches both anchors")

# 18. The rescoring screen's neighbourhood mean is edge-correct and monotone-invariant.
from rescore_screen import pool, rules  # noqa: E402

flat18 = torch.full((2, G * G), 2.5)
for k in (2, 3, 5):
    assert torch.allclose(pool(flat18, k), flat18),         f"a constant map must pool to itself at k={k}, including at the borders"
spike18 = torch.zeros(1, G * G)
spike18[0, 0] = 9.0
assert abs(float(pool(spike18, 3)[0, 0]) - 9.0 / 4) < 1e-6,     "a corner token averages over its 4 valid cells, not 9"
torch.manual_seed(5)
s18 = torch.randn(16, G * G)
for name, fn in rules([1.0]).items():
    assert fn(s18).shape == s18.shape, name
# Any strictly increasing per-token map leaves the argmax alone; this is why the screen
# only contains rules that read a token's neighbours.
for f18 in (lambda v: 3 * v + 7, torch.expm1, lambda v: v ** 3):
    assert torch.equal(f18(s18).argmax(1), s18.argmax(1)),         "a monotone transform changed the argmax"
print("18. rescoring helpers     OK  neighbourhood mean is edge-correct, monotone "
      "transforms leave the argmax fixed")

print("\nall selection checks passed")
