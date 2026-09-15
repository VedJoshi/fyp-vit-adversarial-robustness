"""Check oracle masks, reversal, paired seeds and sample matching without data."""
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fyp import attacks, config, data, metrics, rsa  # noqa: E402
from gate_a import paired_seed  # noqa: E402
from _stub_vit import GRID, PX, _Recorder, _StubViT  # noqa: E402

SIZES = (10, 20, 30, 40, 50)


def brute_force_window(top, left, size, window):
    """Max-coverage window over every w x w window inside the grid, ties by (row, col)."""
    cells = [(i // GRID, i % GRID) for i in
             metrics.patch_token_indices(top, left, size, include_cls_offset=False)]
    best, best_key = None, None
    for r in range(GRID - window + 1):
        for c in range(GRID - window + 1):
            covered = sum(r <= pr < r + window and c <= pc < c + window for pr, pc in cells)
            key = (-covered, r, c)
            if best_key is None or key < best_key:
                best, best_key = (r, c, window, window), key
    return best


torch.manual_seed(0)
model = _StubViT().eval()
x = torch.rand(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
y = torch.tensor([1, 4])

# 1. Installation
cfg = rsa.RSAConfig.for_patch(20)
top, left = 60, 60                       # straddles: 2x2 tokens against a 3x3 window
with rsa.enable(model, cfg) as handle:
    expected = set(metrics.patch_token_indices(top, left, 20))
    handle.set_oracle(top, left, 20, mode="extent")
    with torch.no_grad():
        model(x)
    for layer, mask in handle.masks().items():
        got = set(mask[0].nonzero().flatten().tolist())
        assert got == expected, f"layer {layer} masked {sorted(got)}, expected {sorted(expected)}"
        assert not mask[:, config.CLS_INDEX].any(), f"layer {layer} masked CLS"
    n_extent = len(expected)

    handle.set_oracle(top, left, 20, mode="window")
    with torch.no_grad():
        model(x)
    win_masks = handle.masks()
    for layer, mask in win_masks.items():
        assert int(mask[0].sum()) == cfg.window ** 2, \
            f"layer {layer} masked {int(mask[0].sum())} tokens, expected {cfg.window ** 2}"
    n_window = int(win_masks[0][0].sum())
    assert set(win_masks[0][0].nonzero().flatten().tolist()) >= expected, \
        "the window oracle does not cover the patch it was given"
print(f"1. installation             OK  extent masks {n_extent} tokens, "
      f"window masks {n_window} at ({top},{left}), 20px")

# 2. The window oracle
checked = 0
for size in SIZES:
    w = metrics.rsa_window_size(size)
    for t, l in attacks.patch_locations(size):
        got = rsa.oracle_window(t, l, size, w)
        assert got == brute_force_window(t, l, size, w), \
            f"{size}px at ({t},{l}): {got} is not the max-coverage window"
        r, c, h, wd = got
        cells = [(i // GRID, i % GRID) for i in
                 metrics.patch_token_indices(t, l, size, include_cls_offset=False)]
        assert all(r <= pr < r + h and c <= pc < c + wd for pr, pc in cells), \
            f"{size}px at ({t},{l}): window {got} does not cover the patch"
        assert h * wd == w * w, f"{size}px at ({t},{l}): window {got} is not {w}x{w}"
        checked += 1
assert len(rsa._candidate_windows("ceil+1", 3)) == (GRID - 2) ** 2
assert len(rsa._candidate_windows("padded", 3)) == GRID * GRID
assert len(rsa._candidate_windows("multi", 3)) == sum(
    (GRID - h + 1) * (GRID - wd + 1) for h in (2, 3) for wd in (2, 3))
try:
    rsa._candidate_windows("topk", 3)
    raise AssertionError("topk should have no candidate window set")
except ValueError:
    pass
print(f"2. oracle_window            OK  max coverage and full cover at all "
      f"{checked} (size, location) pairs")

# 3. Reversal
with rsa.enable(model, cfg) as handle:
    with torch.no_grad():
        before_logits = model(x).clone()
    before = {i: m.clone() for i, m in handle.masks().items()}

    handle.set_oracle(top, left, 20, mode="window")
    with torch.no_grad():
        model(x)
    during = {i: m.clone() for i, m in handle.masks().items()}
    assert any(not torch.equal(before[i], during[i]) for i in before), \
        "the oracle mask matched the argmax mask everywhere, so this proves nothing"

    handle.clear_oracle()
    with torch.no_grad():
        after_logits = model(x).clone()
    after = handle.masks()
    for i in before:
        assert torch.equal(before[i], after[i]), f"layer {i} mask changed across the oracle"
    assert torch.equal(before_logits, after_logits), "logits changed across the oracle"
print("3. clear_oracle             OK  masks and logits bit-identical to the argmax path")

# 4. Pairing
seeds = {paired_seed(s, t, l, o) for s in SIZES for t, l in ((0, 0), (0, 20), (20, 0))
         for o in (0, 32)}
assert len(seeds) == len(SIZES) * 3 * 2, "paired_seed collides across locations"


def attack_once(arm_model, seed):
    config.seed_everything(seed)
    return attacks.patch_autopgd(arm_model, x, y, top=top, left=left, size=20, steps=3)


with rsa.enable(model, cfg) as handle:
    handle.set_oracle(top, left, 20, mode="window")
    seed = paired_seed(20, top, left, 0)
    a1 = attack_once(model, seed)
    a2 = attack_once(model, seed)
    assert torch.equal(a1, a2), "the same arm under the same seed produced two patches"
    unpaired = attacks.patch_autopgd(model, x, y, top=top, left=left, size=20, steps=3)
    assert not torch.equal(a1, unpaired), \
        "the attack is deterministic without the seed, so pairing proves nothing"

inits = {}
for arm in ("window", "extent"):
    rec = _Recorder(model)
    with rsa.enable(model, cfg) as handle:
        handle.set_oracle(top, left, 20, mode=arm)
        config.seed_everything(paired_seed(20, top, left, 0))
        attacks.patch_autopgd(rec, x, y, top=top, left=left, size=20, steps=1)
    inits[arm] = rec.first
assert torch.equal(inits["window"], inits["extent"]), \
    "the two arms started from different patches"
print("4. paired_seed              OK  one arm reproduces itself, two arms share an init")

# 5. Sample matching
N_DS = 5000                                       # ImageNet-100 val
small = sorted(random.Random(config.SEED).sample(range(N_DS), 128))
large = sorted(random.Random(config.SEED).sample(range(N_DS), 512))
assert set(small) <= set(large), "the 128-image draw is not inside the 512-image draw"
assert small != large[:128], "the 128-image draw is the prefix; the mapping is moot"
keep = data.matched_positions(128, 512, N_DS, seed=config.SEED)
assert len(set(keep)) == 128, "matched_positions returned duplicate positions"
assert [large[p] for p in keep] == small, "matched_positions names the wrong images"
assert keep != list(range(128)), "slicing the first 128 positions would have worked"
for n_ds in (2000, 3000, 4000):
    try:
        data.matched_positions(128, 512, n_ds, seed=config.SEED)
    except ValueError:
        pass
    else:
        raise AssertionError(f"non-nested draws at dataset_size={n_ds} did not raise")
print(f"5. sample matching          OK  128 inside 512 by identity, not by prefix; "
      f"first 3 positions {keep[:3]}; non-nesting at 2000/3000/4000 raises")

print("\nall checks passed")
