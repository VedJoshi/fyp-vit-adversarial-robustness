"""Check centred-mask geometry and installation using random weights."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fyp import config, metrics, rsa  # noqa: E402

GRID = config.GRID
PATCH = config.PATCH_SIZE


def corner(scores: torch.Tensor, w: int):
    """The shipped ``"centred"`` branch, called unbound because it ignores `self`."""
    return rsa.RSAAttention._window_corner(None, scores, w, GRID, "centred")


def peaked(cell: int, n: int = 1) -> torch.Tensor:
    """A (n, GRID*GRID) score map whose argmax is `cell`."""
    s = torch.zeros(n, GRID * GRID)
    s[:, cell] = 1.0
    return s


def spans(patch_px: int, offsets) -> set[int]:
    """Token spans a patch of `patch_px` takes over `offsets`, on one axis."""
    return {(t + patch_px - 1) // PATCH - t // PATCH + 1 for t in offsets}


def check_block_side():
    """The guaranteed side, on the evaluation grid and off it."""
    assert metrics.centred_block_side(20, stride=20) == 3
    assert metrics.centred_block_side(20, stride=None) == 5
    assert spans(20, [t for t in range(224 - 19) if t % 20 == 0]) == {2}
    assert spans(20, range(PATCH)) == {2, 3}
    for px in (10, 30, 40, 50):
        assert metrics.centred_block_side(px, stride=20) > metrics.rsa_window_size(px), px
    assert metrics.centred_block_side(20, stride=20) == metrics.rsa_window_size(20) == 3
    assert metrics.centred_block_side(0) == 0
    print("block side      ok   20px: 3 on the stride-20 grid, 5 off it")


def check_offgrid_counterexample():
    """A 20 px patch at pixel 47 spans three cells, so a 3-wide block can miss one."""
    top = left = 47
    cells = sorted(i // GRID for i in
                   metrics.patch_token_indices(top, left, 20, include_cls_offset=False))
    rows = sorted(set(cells))
    assert rows == [2, 3, 4], rows

    r0, c0, h, w = corner(peaked(2 * GRID + 2), 3)
    covered = set(range(int(r0[0]), int(r0[0]) + h))
    assert 4 not in covered, covered

    r0, _, h, _ = corner(peaked(2 * GRID + 2), metrics.centred_block_side(20))
    assert set(rows) <= set(range(int(r0[0]), int(r0[0]) + h))
    print("off-grid        ok   pixel 47 spans tokens 2,3,4; a 3-block on token 2 misses 4")


def check_border_shift():
    """At the edge the block shifts inward: still inside, still w*w tokens."""
    w = 3
    for cell, want in (
        (0, (0, 0)),                                   # top-left corner
        (GRID * GRID - 1, (GRID - w, GRID - w)),       # bottom-right corner
        (GRID * 7 + 0, (6, 0)),                        # left edge, interior row
        (GRID * 0 + 7, (0, 6)),                        # top edge, interior column
        (GRID * 7 + 7, (6, 6)),                        # interior
    ):
        r0, c0, h, wd = corner(peaked(cell), w)
        assert (int(r0[0]), int(c0[0])) == want, (cell, int(r0[0]), int(c0[0]), want)
        assert 0 <= int(r0[0]) <= GRID - h and 0 <= int(c0[0]) <= GRID - wd
        assert h * wd == w * w
    print("border shift    ok   block stays inside the grid and keeps 9 tokens")


def check_containment_on_grid():
    """Check coverage from every patch token on the stride-20 grid."""
    w = metrics.centred_block_side(20, stride=20)
    locs = [t for t in range(224 - 19) if t % 20 == 0]
    assert len(locs) == 11
    checked = 0
    for top in locs:
        for left in locs:
            cells = [(i // GRID, i % GRID) for i in
                     metrics.patch_token_indices(top, left, 20, include_cls_offset=False)]
            assert len(cells) == 4
            for pr, pc in cells:
                r0, c0, h, wd = corner(peaked(pr * GRID + pc), w)
                r0, c0 = int(r0[0]), int(c0[0])
                assert all(r0 <= r < r0 + h and c0 <= c < c0 + wd for r, c in cells), \
                    (top, left, pr, pc, r0, c0)
                checked += 1
    assert checked == 121 * 4
    print(f"containment     ok   {checked} peak placements over 121 locations, all covered")


def check_config_guards():
    """for_patch sizes odd, an even side still blocks contiguously, guards hold."""
    cfg = rsa.RSAConfig.for_patch(20, window_rule="centred")
    assert cfg.window == 3 and cfg.masked_tokens == 9
    assert cfg.allowed_masked_counts == frozenset({9})

    # Even sides are never auto-sized but are allowed for size-matched comparisons.
    assert rsa.RSAConfig(patch_px=40, window=6, window_rule="centred").masked_tokens == 36
    for t in range(GRID * GRID):
        r0, c0, h, w = corner(peaked(t), 6)
        assert (h, w) == (6, 6) and 0 <= int(r0) <= GRID - 6 and 0 <= int(c0) <= GRID - 6
        assert int(r0) <= t // GRID < int(r0) + 6 and int(c0) <= t % GRID < int(c0) + 6,             f"an even block must contain its anchor; token {t} gave ({int(r0)},{int(c0)})"

    rejected = [(dict(n_windows=0), "below 1"),
                (dict(n_windows=2, window_rule="topk"), "centred rule only"),
                (dict(n_windows=9, nms_radius=5, window_rule="centred"),
                 "suppress the whole")]
    # One window is always selectable, whatever the radius implies about suppression.
    for side in (3, 7, 8, GRID):
        rsa.RSAConfig(patch_px=20, window=side, window_rule="centred")
    for bad, msg in rejected:
        try:
            rsa.RSAConfig(patch_px=20, window=3, **bad)
        except ValueError as e:
            assert msg in str(e), e
        else:
            raise AssertionError(f"{bad} should be rejected")
    assert "centred" in rsa.WINDOW_RULES
    import checkpoint_control
    assert "centred" in checkpoint_control.NO_WINDOW_RANK
    assert rsa.RSAConfig.for_patch(20, window_rule="centred").window == 3
    assert rsa.RSAConfig.for_patch(40, window_rule="centred").window == 7,         "40 px needs the centred guarantee, not rsa_window_size's 4"
    assert metrics.rsa_window_size(40, rule="centred") == 4,         "the trap this guards is still there; the guard is in for_patch"
    for k in (None, 0, 5):
        assert rsa.RSAConfig.for_patch(
            20, window_rule="centred", freeze_mask_after=k).freeze_mask_after == k
    print("config guards   ok   even sides block contiguously, 9 tokens, multi-window "
          "guards reject, registered in both tuples; "
          "for_patch sizes the block by the guarantee and carries freeze_mask_after")


def check_batch_independence():
    """Each image picks its own block; the rule does not reduce over the batch."""
    s = torch.zeros(3, GRID * GRID)
    for i, cell in enumerate((0, GRID * 7 + 7, GRID * GRID - 1)):
        s[i, cell] = 1.0
    r0, c0, _, _ = corner(s, 3)
    assert r0.tolist() == [0, 6, GRID - 3] and c0.tolist() == [0, 6, GRID - 3]
    print("per-image       ok   three peaks give three different blocks")


def check_installed_masks():
    """Check the installed masks on a random-weight model, including freezing."""
    try:
        import timm
    except ImportError:
        print("installed masks  skip  timm not importable")
        return

    torch.manual_seed(0)
    model = timm.create_model("deit_small_patch16_224", pretrained=False,
                              num_classes=config.N_CLASSES).eval()
    x = torch.randn(4, 3, 224, 224)

    cfg = rsa.RSAConfig.for_patch(20, window_rule="centred")
    with rsa.enable(model, cfg) as h:
        with torch.no_grad():
            model(x)
        for mod in h.modules:
            mask = mod.last_mask
            assert not mask[:, 0].any(), "CLS was masked"
            grid = mask[:, 1:].view(-1, GRID, GRID)
            assert (grid.flatten(1).sum(1) == cfg.masked_tokens).all()
            for b in range(grid.shape[0]):
                rs, cs = grid[b].nonzero(as_tuple=True)
                assert int(rs.max() - rs.min()) == cfg.window - 1
                assert int(cs.max() - cs.min()) == cfg.window - 1
                assert (int(rs.min()), int(cs.min())) == (int(mod.last_window[b, 0]),
                                                          int(mod.last_window[b, 1]))
    print("installed masks ok   12 blocks: 9 tokens, contiguous 3x3, CLS untouched")

    frozen = dataclasses.replace(cfg, freeze_mask_after=0)
    with rsa.enable(model, frozen) as h:
        with torch.no_grad():
            model(x)
        first = h.modules[0].last_mask
        for mod in h.modules[1:]:
            assert torch.equal(mod.last_mask, first), "frozen mask changed after block 0"
    print("freeze at 0     ok   every block reuses block 0's mask")


def check_screen_predicate():
    """Check counterfactual coverage: membership implies containment, not conversely."""
    from checkpoint_control import centred_contains

    locs = [t for t in range(224 - 19) if t % 20 == 0]
    scores = torch.full((GRID * GRID, GRID * GRID), -1.0)
    scores[torch.arange(GRID * GRID), torch.arange(GRID * GRID)] = 1.0
    extra = {3: 0, 5: 0}
    for top in locs:
        for left in locs:
            cells = metrics.patch_token_indices(top, left, 20, include_cls_offset=False)
            member = torch.zeros(GRID * GRID, dtype=torch.bool)
            member[list(cells)] = True
            r0, c0 = top // config.PATCH_SIZE, left // config.PATCH_SIZE
            r1, c1 = (top + 19) // config.PATCH_SIZE, (left + 19) // config.PATCH_SIZE
            for side in (3, 5):
                got = centred_contains(scores, side, GRID, r0, r1, c0, c1)
                assert (got | member).equal(got), (top, left, side, "a member peak must contain")
                extra[side] += int((got & ~member).sum())
    # Border shifts allow some peaks outside the footprint to cover the patch.
    assert extra[3] == 92 and extra[5] == 1632, extra
    print(f"screen predicate ok  membership implies containment at all 121 locations; "
          f"border clamp adds {extra[3]} non-member peaks at side 3, {extra[5]} at side 5")


def main():
    check_block_side()
    check_offgrid_counterexample()
    check_border_shift()
    check_containment_on_grid()
    check_config_guards()
    check_batch_independence()
    check_installed_masks()
    check_screen_predicate()
    print("\nall centred-rule checks passed")


if __name__ == "__main__":
    main()
