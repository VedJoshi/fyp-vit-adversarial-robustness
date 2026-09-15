"""Check window ties, token anchoring and frozen-mask reuse."""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

from fyp import config, metrics, rsa  # noqa: E402
from checkpoint_control import NO_WINDOW_RANK  # noqa: E402

GRID = config.GRID
W = 3
ok = 0


def check(name, detail):
    global ok
    ok += 1
    print(f"{name:<34} OK  {detail}")


class _Src(nn.Module):
    """The attributes `rsa.RSAAttention` copies out of a timm Attention."""

    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = self.k_norm = nn.Identity()
        self.attn_drop = self.proj_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.num_heads, self.head_dim, self.scale = 1, 1, 1.0


def planted(values):
    """A (1, 1, N, 1) value tensor whose token scores are |value - mean|."""
    v = torch.zeros(1, 1, GRID * GRID + 1, 1)
    v[0, 0, 1:, 0] = values
    return v


torch.manual_seed(0)
# Texture breaks mean-pooling ties while preserving a shared maximum.
BACKGROUND = torch.rand(GRID * GRID) * 0.5


def peak_at(r, c, value=9.0, background=False):
    vals = BACKGROUND.clone() if background else torch.zeros(GRID * GRID)
    vals[r * GRID + c] = value
    return vals


# 1. tie counting against brute force
for (r, c), expected in (((6, 6), W * W), ((0, 0), 1), ((0, 6), W), ((6, 0), W)):
    scores = peak_at(r, c, background=True)[None, :]
    ws_max = metrics.rsa_window_scores(scores, W, pool="max")
    ws_mean = metrics.rsa_window_scores(scores, W)
    brute = sum(
        1
        for i in range(GRID - W + 1)
        for j in range(GRID - W + 1)
        if r in range(i, i + W) and c in range(j, j + W)
    )
    assert brute == expected, ((r, c), brute, expected)
    got = metrics.rsa_tied_windows(ws_max).item()
    assert got == expected, ((r, c), got, expected)
    assert metrics.rsa_tied_windows(ws_mean).item() == 1, (r, c)

blank = peak_at(6, 6)[None, :]
assert metrics.rsa_tied_windows(metrics.rsa_window_scores(blank, W)).item() == W * W
assert metrics.rsa_tied_windows(
    metrics.rsa_window_scores(blank, W, pool="max")).item() == W * W
check("1. tied windows", f"on texture, max ties {W * W} at an interior peak, {W} on an "
                         f"edge, 1 at a corner while mean ties 1; on a blank grid both "
                         f"tie {W * W}")

# 2. ties are what the max argmax is actually resolving
scores = peak_at(6, 6, background=True)[None, :]
ws = metrics.rsa_window_scores(scores, W, pool="max")
r0, c0 = metrics.rsa_argmax_window(ws)[0].tolist()
assert 6 in range(r0, r0 + W) and 6 in range(c0, c0 + W), (r0, c0)
winners = (ws.flatten(1)[0] == ws.max()).nonzero().flatten()
n = ws.shape[-1]
covered = {
    frozenset(
        (rr, cc)
        for rr in range(int(i) // n, int(i) // n + W)
        for cc in range(int(i) % n, int(i) % n + W)
    )
    for i in winners
}
assert len(covered) == W * W, len(covered)
check("2. ties are distinct masks", f"{W * W} tied windows, {len(covered)} distinct "
                                    f"token sets, argmax takes {(r0, c0)}")

# 3. the frozen mask
cfg_free = rsa.RSAConfig(patch_px=20, window=W, window_rule="ceil+1", score_frame="global")
cfg_frozen = rsa.RSAConfig(patch_px=20, window=W, window_rule="ceil+1",
                           score_frame="global", freeze_mask_after=0)


def run_blocks(cfg, per_layer_values):
    """One forward pass' worth of `select` calls, three blocks sharing one dict."""
    shared = {}
    mods = [rsa.RSAAttention(_Src(1), cfg, layer_idx=i, shared=shared) for i in range(3)]
    return [m.select(planted(v)) for m, v in zip(mods, per_layer_values)]


vals = [peak_at(6, 6), peak_at(2, 2), peak_at(10, 10)]
free = run_blocks(cfg_free, vals)
assert not torch.equal(free[0], free[1]) and not torch.equal(free[1], free[2])
frozen = run_blocks(cfg_frozen, vals)
assert torch.equal(frozen[0], free[0]), "block 0 must select exactly as it would unfrozen"
assert torch.equal(frozen[1], frozen[0]) and torch.equal(frozen[2], frozen[0])
check("3. frozen mask", "unfrozen blocks select three different masks; frozen blocks "
                        "1-2 reuse block 0's")

# 4. the frozen arm refuses the states where it would be wrong
orphan = rsa.RSAAttention(_Src(1), cfg_frozen, layer_idx=1, shared={})
try:
    orphan.select(planted(peak_at(6, 6)))
except RuntimeError as e:
    assert "wrote no mask" in str(e), e
else:
    raise AssertionError("a later block with no stored mask must raise")

shared = {}
first = rsa.RSAAttention(_Src(1), cfg_frozen, layer_idx=0, shared=shared)
second = rsa.RSAAttention(_Src(1), cfg_frozen, layer_idx=1, shared=shared)
first.select(planted(peak_at(6, 6)))
two = torch.cat([planted(peak_at(6, 6)), planted(peak_at(2, 2))], dim=0)
try:
    second.select(two)
except RuntimeError as e:
    assert "batch" in str(e), e
else:
    raise AssertionError("a batch-size change under a frozen mask must raise")
check("4. frozen mask guards", "reuse before any write raises; a batch-size change raises")

# 5. the token-anchored rule and the geometry contract
vals = torch.zeros(GRID * GRID)
vals[0] = 9.0
for r in range(6, 9):
    for c in range(6, 9):
        vals[r * GRID + c] = 2.0


def cells(rule):
    cfg = rsa.RSAConfig(patch_px=20, window=W, window_rule=rule, score_frame="global")
    m = rsa.RSAAttention(_Src(1), cfg, layer_idx=0, shared={})
    mask = m.select(planted(vals))
    return {(i // GRID, i % GRID) for i in mask[0, 1:].nonzero().flatten().tolist()}


assert (0, 0) not in cells("ceil+1"), "mean was expected to abandon the corner"
assert (0, 0) in cells("token"), "token must anchor on the strongest token"
assert (0, 0) in cells("max"), "max must contain the peak"
assert (0, 0) in cells("centred"), "centred must contain the peak it is placed on"
assert cells("centred") == {(r, c) for r in range(W) for c in range(W)}, cells("centred")
assert set(NO_WINDOW_RANK) == {"multi", "token", "topk", "centred"}, NO_WINDOW_RANK
assert set(NO_WINDOW_RANK) <= set(rsa.WINDOW_RULES), NO_WINDOW_RANK
assert "max" not in NO_WINDOW_RANK, "max shares ceil+1's candidate set and keeps its rank"
check("5. token anchor and contract", "token and max hold the peak, mean does not; "
                                      f"{len(NO_WINDOW_RANK)} rules drop the window rank")

# 6. the frozen arm as `enable` installs it
K, N_BLOCKS = 5, 12


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Src(1)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(N_BLOCKS))


model = _Model()
cfg_k = rsa.RSAConfig(patch_px=20, window=W, window_rule="ceil+1", score_frame="global",
                      freeze_mask_after=K)
handle = rsa.enable(model, cfg_k)
masks = [model.blocks[i].attn.select(planted(peak_at((3 * i) % 13, (5 * i + 1) % 13,
                                                     background=True)))
         for i in range(N_BLOCKS)]
windows = [tuple(model.blocks[i].attn.last_window[0].tolist()) for i in range(N_BLOCKS)]
assert len(set(windows[:K + 1])) == K + 1, \
    f"blocks up to k must still select independently, got {windows[:K + 1]}"
assert all(torch.equal(m, masks[K]) for m in masks[K + 1:]), \
    "every block after k must reuse block k's mask"
assert model.blocks[K].attn.layer_idx == K, "layer_idx must be the block index"

handle.set_config(replace(cfg_k, freeze_mask_after=None))
assert model.blocks[0].attn.shared == {}, "set_config must clear the frozen mask"
handle.remove()

try:
    rsa.enable(_Model(), cfg_k, layers=[0, 1, 2])
except ValueError as e:
    assert "not installed" in str(e), e
else:
    raise AssertionError("freeze_mask_after outside `layers` must raise")
check("6. frozen mask through enable", f"k={K} on {N_BLOCKS} blocks: 0-{K} select {K + 1} "
                                       f"different windows, "
                                       f"{K + 1}-{N_BLOCKS - 1} reuse; set_config clears it; "
                                       f"a partial install omitting block {K} raises")

print(f"\n{ok}/6 checks passed")
