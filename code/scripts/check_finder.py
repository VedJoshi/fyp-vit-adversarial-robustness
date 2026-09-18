"""Check the input-space finder and the objective that attacks it.

Runs on random weights and synthetic images: no dataset and no checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import finder_attack  # noqa: E402
import inputspace_screen  # noqa: E402
from fyp import config, metrics, rsa  # noqa: E402

GRID = config.GRID
SIZE, WINDOW = 20, 3
LOC = (60, 60)


def images(n: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(n, 3, 224, 224)


def loud_patch(x: torch.Tensor, top: int, left: int, size: int = SIZE) -> torch.Tensor:
    """A high-frequency patch, the kind an unconstrained attack produces."""
    adv = x.clone()
    t = torch.zeros(size, size)
    t[::2, ::2] = 1.0
    t[1::2, 1::2] = 1.0
    adv[:, :, top:top + size, left:left + size] = t
    return adv


def check_score_shapes():
    """Every rule returns one score per image token, finite and differentiable."""
    x = images().requires_grad_(True)
    for name in inputspace_screen.RULES:
        s = finder_attack.finder_scores(name, x)
        assert s.shape == (8, GRID * GRID), (name, s.shape)
        assert torch.isfinite(s).all(), name
        g = torch.autograd.grad(s.sum(), x, retain_graph=True)[0]
        assert torch.isfinite(g).all() and g.abs().sum() > 0, name
    print(f"1. score shapes         OK  {len(inputspace_screen.RULES)} rules give "
          f"(B, {GRID * GRID}) finite differentiable maps")


def check_tile_order():
    """Tile k of the score map is the pixels the k-th token reads."""
    x = torch.zeros(1, 3, 224, 224)
    for token in (0, 1, GRID, 195):
        r, c = token // GRID, token % GRID
        y = x.clone()
        y[:, :, r * 16:(r + 1) * 16:2, c * 16:(c + 1) * 16] = 1.0
        assert int(finder_attack.finder_scores("tv", y).argmax(1)) == token, token
    print("2. tile order           OK  a striped tile is the argmax at its own token index")


def check_margin_matches_argmax():
    """A positive margin means the real argmax picks a non-covering anchor."""
    covers = metrics.centred_coverage(*LOC, SIZE, WINDOW)
    x = images(16)
    aux = finder_attack.evasion_loss("tv", *LOC, SIZE, WINDOW, torch.device("cpu"))
    agree = ties = 0
    for adv in (x, loud_patch(x, *LOC)):
        m = aux(adv)
        escaped, tied = finder_attack.hard_escape("tv", adv, *LOC, SIZE, WINDOW,
                                                  torch.device("cpu"))
        for i in range(len(adv)):
            if tied[i] or m[i] == 0:
                ties += 1
                continue
            assert bool(m[i] > 0) == bool(escaped[i]), (i, float(m[i]), bool(escaped[i]))
            agree += 1
    assert agree > 0
    print(f"3. margin vs argmax     OK  sign agrees with the real selector on {agree} cases, "
          f"{ties} ties excluded")


def check_loud_patch_is_found():
    """The finder covers a high-frequency patch; this is what the screen measured."""
    covers = metrics.centred_coverage(*LOC, SIZE, WINDOW)
    adv = loud_patch(images(16), *LOC)
    chosen = finder_attack.finder_scores("tv", adv).argmax(1)
    assert covers[chosen].all(), "tv failed to cover an obviously high-frequency patch"
    m = finder_attack.evasion_loss("tv", *LOC, SIZE, WINDOW, torch.device("cpu"))(adv)
    assert (m < 0).all(), "margin should be negative when the patch is covered"
    print("4. loud patch found     OK  covered on 16/16, and the margin is negative there")


def check_gradient_reaches_patch():
    """The objective pushes pixels inside the patch, away from ties."""
    adv = loud_patch(images(4), *LOC).requires_grad_(True)
    m = finder_attack.evasion_loss("tv", *LOC, SIZE, WINDOW, torch.device("cpu"))(adv)
    g = torch.autograd.grad(m.sum(), adv)[0]
    inside = g[:, :, LOC[0]:LOC[0] + SIZE, LOC[1]:LOC[1] + SIZE].abs().sum()
    assert torch.isfinite(g).all() and inside > 0, float(inside)
    print(f"5. gradient path        OK  {float(inside):.3e} of gradient mass inside the patch")


def check_hook_matches_external():
    """The pre-hook installs the same mask the replay computes outside the model."""
    blocks = [torch.nn.Module() for _ in range(2)]
    model = torch.nn.Module()

    class Stub(torch.nn.Module):
        def forward(self, x):
            return x.mean((1, 2, 3))

    model.forward_stub = Stub()
    x = loud_patch(images(6), *LOC)

    scores = finder_attack.finder_scores("tv", x)
    want = rsa.centred_token_mask(scores.argmax(1), WINDOW)

    installed = {}

    class Handle:
        def set_mask(self, mask):
            installed["mask"] = mask

    hook = finder_attack.install_finder(model.forward_stub, Handle(), "tv", WINDOW)
    model.forward_stub(x)
    hook.remove()
    assert torch.equal(installed["mask"], want), "the hook and the external path disagree"
    assert installed["mask"][:, 0].sum() == 0, "CLS must never be masked"
    assert (installed["mask"][:, 1:].sum(1) == WINDOW * WINDOW).all()
    print(f"6. hook null            OK  hook reproduces the external mask on 6 images, "
          f"CLS untouched, {WINDOW * WINDOW} tokens each")


def check_smooth_init_is_smooth():
    """The smooth initialisation really lowers patch total variation."""
    x = images(4)
    loud = loud_patch(x, *LOC)
    smooth = finder_attack.smooth_init(x, *LOC, SIZE)
    covers = metrics.centred_coverage(*LOC, SIZE, WINDOW)
    anchor = covers.nonzero().flatten()[0]
    s_loud = finder_attack.finder_scores("tv", loud)[:, anchor]
    s_smooth = finder_attack.finder_scores("tv", smooth)[:, anchor]
    assert (s_smooth < s_loud).all(), (s_smooth, s_loud)
    print("7. smooth init          OK  lowers the covering anchor's tv below the loud patch")


def main():
    check_score_shapes()
    check_tile_order()
    check_margin_matches_argmax()
    check_loud_patch_is_found()
    check_gradient_reaches_patch()
    check_hook_matches_external()
    check_smooth_init_is_smooth()
    print("\nall finder checks passed")


if __name__ == "__main__":
    main()
