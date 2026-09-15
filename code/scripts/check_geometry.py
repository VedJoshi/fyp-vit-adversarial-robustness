"""Check geometry aggregation, including medians across unequal batches."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from gate_a import reduce_geometry

LAYERS = 12


def batch(token, window, mask, overlap=None):
    """Build a geometry record with one (B,) tensor per layer and metric."""
    assert len(token) == len(window) == len(mask), "case has inconsistent B"
    if overlap is None:
        overlap = [1.0 if m else 0.0 for m in mask]
    return {
        "token_rank_by_layer": [torch.tensor(token, dtype=torch.float)] * LAYERS,
        "window_rank_by_layer": [torch.tensor(window, dtype=torch.float)] * LAYERS,
        "full_mask_by_layer": [torch.tensor(mask, dtype=torch.bool)] * LAYERS,
        "overlap_by_layer": [torch.tensor(overlap, dtype=torch.float)] * LAYERS,
        "n_patch_tokens": 4,
    }


def check(name, detail):
    print(f"{name:<28}OK  {detail}")


# 1. one batch, the reduction is the plain median and mean
out = reduce_geometry([batch([1, 1, 3, 5], [2, 4, 6, 8], [True, True, False, False])])
assert len(out["median_token_rank_by_layer"]) == LAYERS
# torch.median uses the lower middle value for even-sized samples.
assert out["median_token_rank_by_layer"][0] == 1.0, out["median_token_rank_by_layer"][0]
assert out["median_window_rank_by_layer"][0] == 4.0
assert out["full_mask_pct_by_layer"][0] == 50.0
assert out["n_images"] == 4
check("1. single batch", "median 1, window 4, mask 50.0%, n=4")

# 2. two batches pool before reducing, and n_images counts every image
out = reduce_geometry([
    batch([1, 1], [1, 1], [True, True]),
    batch([9, 9, 9, 9], [9, 9, 9, 9], [False, False, False, False]),
])
assert out["n_images"] == 6, out["n_images"]
assert abs(out["full_mask_pct_by_layer"][0] - 100 / 3) < 1e-4, out["full_mask_pct_by_layer"][0]
assert out["median_token_rank_by_layer"][0] == 9.0
check("2. pooling across batches", "n=6 from 2+4, mask 33.3%, median 9 not 5")

# 3. the bug this guards: median of medians disagrees with the true median
batches = [batch([1, 2, 3], [1, 1, 1], [True, True, True]),
           batch([4, 5, 6, 7, 8, 9, 10], [1] * 7, [True] * 7)]
out = reduce_geometry(batches)
union = torch.cat([b["token_rank_by_layer"][0] for b in batches])
naive = torch.tensor([b["token_rank_by_layer"][0].median() for b in batches]).median()
assert out["median_token_rank_by_layer"][0] == union.median().item() == 5.0
assert naive.item() == 2.0
check("3. not median-of-medians", f"true {union.median().item():.0f}, naive {naive.item():.0f}")

# 4. ranks are per layer, so a layer that differs is not flattened away
per_layer = {
    "token_rank_by_layer": [torch.tensor([float(i + 1)]) for i in range(LAYERS)],
    "window_rank_by_layer": [torch.tensor([1.0])] * LAYERS,
    "full_mask_by_layer": [torch.tensor([i % 2 == 0]) for i in range(LAYERS)],
    "overlap_by_layer": [torch.tensor([1.0 if i % 2 == 0 else 0.25])
                         for i in range(LAYERS)],
    "n_patch_tokens": 4,
}
out = reduce_geometry([per_layer])
assert out["median_token_rank_by_layer"] == [float(i + 1) for i in range(LAYERS)]
assert out["full_mask_pct_by_layer"][0] == 100.0
assert out["full_mask_pct_by_layer"][1] == 0.0
check("4. layers stay separate", "token rank 1..12, mask alternates 100/0")

# 5. partial overlap is visible where full containment reads zero
out = reduce_geometry([batch([1, 1], [1, 1], [False, False], overlap=[0.75, 0.25])])
assert out["full_mask_pct_by_layer"][0] == 0.0
assert abs(out["mean_overlap_pct_by_layer"][0] - 50.0) < 1e-4, out["mean_overlap_pct_by_layer"][0]
assert out["n_patch_tokens"] == 4
check("5. overlap vs containment", "containment 0.0%, mean overlap 50.0% of 4 tokens")

print("\nall checks passed")
