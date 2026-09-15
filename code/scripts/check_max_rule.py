"""Check max pooling, its candidate windows and selected masks."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fyp import config, metrics, rsa  # noqa: E402

GRID = config.GRID
W = 3


def brute_max(scores, window, grid=GRID):
    """Max over each wholly-inside window, written out."""
    g = scores.view(grid, grid)
    n = grid - window + 1
    return torch.tensor([[g[r:r + window, c:c + window].max()
                          for c in range(n)] for r in range(n)])


def check(name, detail):
    print(f"{name:<30}OK  {detail}")


# 1. the pooling is the maximum
torch.manual_seed(0)
s = torch.rand(1, GRID * GRID)
got = metrics.rsa_window_scores(s, W, pool="max")[0]
assert torch.allclose(got, brute_max(s[0], W)), "unpadded max disagrees with brute force"
assert got.shape == (GRID - W + 1, GRID - W + 1), got.shape

pad = metrics.rsa_window_scores(s, W, pool="max", padded=True)[0]
assert pad.shape == (GRID, GRID), pad.shape
assert torch.isfinite(pad).all(), "a -inf pad reached the output"
g = s[0].view(GRID, GRID)
assert torch.allclose(pad[0, 0], g[0:2, 0:2].max()), (pad[0, 0], g[0:2, 0:2].max())
try:
    metrics.rsa_window_scores(s, W, pool="median")
    raise AssertionError("an unknown pool should raise")
except ValueError:
    pass
check("1. max pooling", f"matches brute force on {(GRID - W + 1) ** 2} windows, "
                        f"padded corner over 4 cells")

# 2. Duplicate peaks change the mean; spreading a fixed total changes the max.
one = torch.zeros(1, GRID * GRID)
one[0, 5 * GRID + 5] = 1.0
two = one.clone()
two[0, 5 * GRID + 6] = 1.0

mean_1 = metrics.rsa_window_scores(one, W).max().item()
mean_2 = metrics.rsa_window_scores(two, W).max().item()
max_1 = metrics.rsa_window_scores(one, W, pool="max").max().item()
max_2 = metrics.rsa_window_scores(two, W, pool="max").max().item()
assert abs(mean_2 - 2 * mean_1) < 1e-6, (mean_1, mean_2)
assert max_1 == max_2 == 1.0, (max_1, max_2)

half = torch.zeros(1, GRID * GRID)
half[0, 5 * GRID + 5] = half[0, 5 * GRID + 6] = 0.5
assert abs(metrics.rsa_window_scores(half, W).max().item() - mean_1) < 1e-6
assert metrics.rsa_window_scores(half, W, pool="max").max().item() == 0.5
check("2. same-peak multiplicity", f"mean {mean_1:.4f} to {mean_2:.4f} (2x), "
                                   f"max {max_1:.1f} to {max_2:.1f}; at a halved peak "
                                   f"the max halves")

# 3. the candidate set is ceil+1's
assert rsa._candidate_windows("max", W) == rsa._candidate_windows("ceil+1", W)
pairs = 0
for size in (10, 20, 30, 40, 50):
    w = metrics.rsa_window_size(size)
    for top in range(0, 224 - size + 1, 20):
        for left in range(0, 224 - size + 1, 20):
            a = rsa.oracle_window(top, left, size, w, window_rule="max")
            b = rsa.oracle_window(top, left, size, w, window_rule="ceil+1")
            assert a == b, (size, top, left, a, b)
            pairs += 1
check("3. candidate set", f"identical to ceil+1; oracle_window agrees at {pairs} pairs")


# 4. end to end: the selected window contains the top token
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


def selected_cells(rule, values, w=W):
    cfg = rsa.RSAConfig(patch_px=20, window=w, window_rule=rule, score_frame="global")
    m = rsa.RSAAttention(_Src(1), cfg)
    mask = m.select(planted(values))
    return {(i // GRID, i % GRID) for i in mask[0, 1:].nonzero().flatten().tolist()}


vals = torch.zeros(GRID * GRID)
vals[0] = 9.0
for r in range(6, 9):
    for c in range(6, 9):
        vals[r * GRID + c] = 2.0

by_mean = selected_cells("ceil+1", vals)
by_max = selected_cells("max", vals)
assert (0, 0) not in by_mean, f"mean was expected to miss the corner, got {sorted(by_mean)}"
assert (0, 0) in by_max, f"max must contain the top token, got {sorted(by_max)}"
assert len(by_mean) == len(by_max) == W * W, (len(by_mean), len(by_max))
check("4. selection end to end", f"mean picks {sorted(by_mean)[0]}, "
                                 f"max picks the corner; both mask {W * W} tokens")

# 5. every rule has a pool, and only max is max
assert set(rsa.WINDOW_POOL) == set(rsa.WINDOW_RULES), set(rsa.WINDOW_RULES) ^ set(rsa.WINDOW_POOL)
assert [r for r, p in rsa.WINDOW_POOL.items() if p == "max"] == ["max"], rsa.WINDOW_POOL
check("5. WINDOW_POOL", f"total over {len(rsa.WINDOW_RULES)} rules, max names one")

print("\nall checks passed")
