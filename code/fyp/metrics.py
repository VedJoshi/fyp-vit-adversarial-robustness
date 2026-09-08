"""Attention diagnostics: entropy, dominance, patch attention mass, RSA's score.

All functions take tensors produced by `hooks.AttentionCapture`.

Shape conventions throughout:
    logits, weights : (B, H, N, N)   N = 197 for DeiT-S, CLS at index 0
    v               : (B, H, N, head_dim)
    returns         : (B, ...) with heads/queries reduced unless stated
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from . import config


# --------------------------------------------------------------------------
# Locating a patch in token space
# --------------------------------------------------------------------------
def patch_token_indices(
    top: int,
    left: int,
    size: int,
    grid: int = config.GRID,
    patch_size: int = config.PATCH_SIZE,
    include_cls_offset: bool = True,
) -> list[int]:
    """Token indices covered by a pixel-space patch at (top, left) of side `size`.

    A patch that is not grid-aligned straddles more tokens than its size suggests: a
    32 px patch is 2 tokens wide when aligned and 3 when not.

    With `include_cls_offset` the returned indices are into the full 197-token sequence
    (CLS at 0), which is how the attention matrices are indexed.
    """
    r0, r1 = top // patch_size, (top + size - 1) // patch_size
    c0, c1 = left // patch_size, (left + size - 1) // patch_size
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, grid - 1), min(c1, grid - 1)

    off = 1 if include_cls_offset else 0
    return [off + r * grid + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def token_grid_position(token: int, grid: int = config.GRID, include_cls_offset: bool = True) -> tuple[int, int]:
    """Inverse of the above for a single token -> (row, col) in the token grid.

    Raises ValueError for a token that has no grid position: CLS under
    `include_cls_offset`, and any index outside the `grid * grid` image tokens.
    """
    j = token - (1 if include_cls_offset else 0)
    if not 0 <= j < grid * grid:
        which = "CLS" if include_cls_offset and token == config.CLS_INDEX else "out of range"
        raise ValueError(
            f"token {token} has no grid position ({which}); image tokens are "
            f"{1 if include_cls_offset else 0}..{grid * grid - (0 if include_cls_offset else 1)}"
        )
    return divmod(j, grid)


def _key_sets(
    n_tokens: int,
    patch_tokens: Sequence[int],
    exclude_cls: bool,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks (in_patch, out_of_patch) over the key axis."""
    in_p = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    in_p[torch.as_tensor(list(patch_tokens), device=device)] = True
    out_p = ~in_p
    if exclude_cls:
        out_p[config.CLS_INDEX] = False
        # A patch token can never be CLS, but guard anyway.
        in_p[config.CLS_INDEX] = False
    return in_p, out_p


# --------------------------------------------------------------------------
# Entropy
# --------------------------------------------------------------------------
def attention_entropy(weights: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """H_i = -sum_j a_ij log a_ij, per query, then reduced over heads and queries.

    Entropy is permutation-invariant over the key axis: [0.7,0.1,0.1,0.1] and
    [0.1,0.7,0.1,0.1] score identically. It measures how spread out attention is, not
    where it went. `dominance` is position-aware.

    reduce: "mean" over heads and queries, "cls" for the CLS query only, or "none".
    """
    p = weights.clamp_min(1e-12)
    h = -(p * p.log()).sum(-1)          # (B, H, N)
    if reduce == "none":
        return h
    if reduce == "cls":
        return h[:, :, config.CLS_INDEX].mean(1)     # (B,)
    return h.mean(dim=(1, 2))                        # (B,)


def max_entropy(n_tokens: int = config.N_TOKENS) -> float:
    """log N - the entropy of uniform attention."""
    return math.log(n_tokens)


# --------------------------------------------------------------------------
# Dominance
# --------------------------------------------------------------------------
def _lme(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-mean-exp over the masked entries of the last axis.

    LME_S(z) = log( (1/|S|) sum_{j in S} e^{z_j} ) = logsumexp(z_S) - log|S|

    A smooth soft-maximum: dominated by the largest entries, differentiable
    everywhere.
    """
    neg_inf = torch.finfo(z.dtype).min
    masked = z.masked_fill(~mask, neg_inf)
    return torch.logsumexp(masked, dim=-1) - math.log(int(mask.sum()))


def dominance(
    logits: torch.Tensor,
    patch_tokens: Sequence[int],
    reduce: str = "mean",
    exclude_cls: bool = True,
) -> torch.Tensor:
    """D_i = LME_{j in P} B_ij - LME_{j not in P} B_ij, reduced over heads/queries.

    Reading:  D << 0  legitimate tokens dominate (healthy)
              D ~= 0  patch and non-patch equally competitive
              D >  0  the patch is the preferred attention target (hijacked)

    Because the softmax denominator is common to both terms it cancels exactly, so
    this is identical to the log-ratio of post-softmax attention mass::

        D_i = log[ mean_{j in P} a_ij / mean_{j not in P} a_ij ]

    i.e. "how many times more attention the patch receives than an average
    non-patch token". Computing it from the *logits* is a purely numerical choice:
    at the logit gaps a trained ViT reaches (see `logit_gap`), the post-softmax
    values underflow to exactly 0 or 1 in float32 and carry no gradient, while the
    logits still do.

    `exclude_cls` keeps CLS out of the comparison set, so the baseline is an average
    image token; CLS structurally attracts a large share of attention.
    """
    n = logits.shape[-1]
    in_p, out_p = _key_sets(n, patch_tokens, exclude_cls, logits.device)

    d = _lme(logits, in_p) - _lme(logits, out_p)     # (B, H, N)
    if reduce == "none":
        return d
    if reduce == "cls":
        return d[:, :, config.CLS_INDEX].mean(1)     # (B,)
    return d.mean(dim=(1, 2))                        # (B,)


def patch_attention_mass(
    weights: torch.Tensor,
    patch_tokens: Sequence[int],
    reduce: str = "mean",
    exclude_cls_query: bool = False,
) -> torch.Tensor:
    """S_P - total post-softmax attention flowing *into* the patch tokens.

    The Decoys paper's received-attention score (their Eq. 5), the quantity ARMRO
    thresholds on, and what Patch-Fool's attention term maximises.
    """
    idx = torch.as_tensor(list(patch_tokens), device=weights.device)
    s = weights[..., idx].sum(-1)                    # (B, H, N)
    if exclude_cls_query:
        s = s[:, :, 1:]
    if reduce == "none":
        return s
    if reduce == "cls":
        return weights[:, :, config.CLS_INDEX][..., idx].sum(-1).mean(1)
    return s.mean(dim=(1, 2))


# --------------------------------------------------------------------------
# Logit gaps
# --------------------------------------------------------------------------
#: Past this gap, exp(-gap) is exactly 0 in float32, so every non-maximal softmax entry
#: underflows and the attention row becomes exactly one-hot with exactly zero gradient.
#: Jain & Dutta (CVPR 2024) report gaps of 250-1000 by block 12 of a trained ViT-B/16.
#:
#: The value is where exp(-gap) falls below half the smallest positive subnormal and
#: rounds to zero, `-log(2^-150) = 150 * log 2 = 103.972`. At gap = 103 the result is
#: still the smallest subnormal 1.4e-45, not zero; a threshold of 103 therefore labels
#: rows in (103, 103.972] as underflowed while their smallest probability is nonzero.
#: This is a derivation from the float32 format, not a number Jain & Dutta state. The
#: measured boundary of `torch.exp` on the pinned build (torch 2.7.0) is 103.9720802,
#: 3e-6 above the format value, so the constant is conservative to that precision.
FLOAT32_UNDERFLOW_GAP = 150.0 * math.log(2.0)          # 103.97208...


def logit_gap(logits: torch.Tensor, reduce: str = "max") -> torch.Tensor:
    """Gap between the largest and second-largest pre-softmax logit, per query.

    Past `FLOAT32_UNDERFLOW_GAP` the softmax is saturated in float32 and the
    non-maximal entries carry no gradient.
    """
    top2 = logits.topk(2, dim=-1).values
    gap = top2[..., 0] - top2[..., 1]                # (B, H, N)
    if reduce == "none":
        return gap
    if reduce == "max":
        return gap.amax(dim=(1, 2))
    return gap.mean(dim=(1, 2))


def logit_span(logits: torch.Tensor, reduce: str = "none") -> torch.Tensor:
    """Gap between the largest and smallest pre-softmax logit, per query.

    The span is the first gap in a row to cross the underflow threshold, since the
    smallest entry is the first to vanish. `logit_gap` is the last.
    """
    span = logits.amax(dim=-1) - logits.amin(dim=-1)     # (B, H, N)
    if reduce == "none":
        return span
    if reduce == "max":
        return span.amax(dim=(1, 2))
    return span.mean(dim=(1, 2))


def underflow_fraction(logits: torch.Tensor, mode: str = "complete") -> torch.Tensor:
    """Fraction of attention rows past the float32 underflow threshold.

    `mode="complete"` uses `logit_gap`, largest minus second largest, which is Jain &
    Dutta's quantity. Crossing it means *every* non-maximal entry has underflowed and
    the row is exactly one-hot with exactly zero gradient.

    `mode="partial"` uses `logit_span`, largest minus smallest. Crossing it means the
    *smallest* entry has underflowed while the row is still not one-hot. This is the
    weaker event and it happens first, so `complete` at 0% does not by itself establish
    that no entry in any row has underflowed. Reporting one and describing the other is
    how "0% underflow" overstates the measurement.
    """
    if mode == "complete":
        gap = logit_gap(logits, reduce="none")
    elif mode == "partial":
        gap = logit_span(logits, reduce="none")
    else:
        raise ValueError(f"unknown mode {mode!r}; pick 'complete' or 'partial'")
    return (gap > FLOAT32_UNDERFLOW_GAP).float().mean(dim=(1, 2))


# --------------------------------------------------------------------------
# RSA's anomaly score
# --------------------------------------------------------------------------
#: Which vector the value tokens are centred on before the norm is taken. Algorithm 1
#: writes a single `mu` and does not say whether it is formed per head.
#:
#: ``"global"``    every head's tokens are measured against one vector: the mean over
#:                 heads of the per-head token means. The score is
#:                 mean_h ||v_i^h - mu_bar||, mu_bar = MEAN_h(MEAN_i(v_i^h)).
#: ``"per-head"``  each head's tokens are measured against that head's own mean. The
#:                 score is mean_h ||v_i^h - mu^h||, mu^h = MEAN_i(v_i^h).
#:
#: ``"global"`` is what the authors' code does, at
#: `pycls/models/vision_transformer.py:89-93` of wagner-group/robust-self-attention:
#: it forms the per-head mean, averages that over heads, and broadcasts the single
#: result across every head. The per-head mean is still what replaces a masked value
#: at line 101, so their score and their replacement are centred differently.
#:
#: The two frames are not a constant apart. Under ``"global"`` each head carries the
#: offset (mu^h - mu_bar), so the norm mixes a token's own deviation with a fixed
#: per-head vector, the token ranking changes, and the argmax selects a different
#: window. ``"per-head"`` is what this project ran before the reference code was
#: available; it is kept so those results stay reproducible.
SCORE_FRAMES = ("global", "per-head")
DEFAULT_SCORE_FRAME = "global"


def rsa_token_scores(
    v: torch.Tensor,
    exclude_cls: bool = True,
    frame: str = DEFAULT_SCORE_FRAME,
) -> torch.Tensor:
    """RSA's per-token anomaly score, ||v_i - mu||_2, with mu a mean image-token value.

    From RSA Algorithm 1::

        mu   <- MEAN_{i in N}(v_i)          # image tokens only, CLS excluded
        s(w) <- sum_{i in w} ||v_i - mu||_2

    The score is an isotropic distance and so is direction-blind, while adversarial
    harm depends on the component of (v_p - mu) along the margin-reducing direction
    (`direction_decomposition`).

    `frame` picks which mean plays the part of `mu`; see `SCORE_FRAMES`. The norms are
    averaged across heads either way. Returns (B, N_img).
    """
    vv = v[:, :, 1:, :] if exclude_cls else v        # (B, H, N_img, head_dim)
    mu = score_frame_mean(v, exclude_cls=exclude_cls, frame=frame)
    scores = (vv - mu).norm(dim=-1)                  # (B, H, N_img)
    return scores.mean(1)                            # average over heads -> (B, N_img)


def score_frame_mean(
    v: torch.Tensor,
    exclude_cls: bool = True,
    frame: str = DEFAULT_SCORE_FRAME,
) -> torch.Tensor:
    """The `mu` that `rsa_token_scores` measures deviations against, under `frame`.

    Broadcastable against `(B, H, N, head_dim)`: shape `(B, 1, 1, head_dim)` under
    ``"global"`` and `(B, H, 1, head_dim)` under ``"per-head"``.

    Anything that interprets what RSA charges a token for - `direction_decomposition`
    and the mechanism scripts above all - has to centre on the same vector the ranking
    used. Centring per head while the score is ranked globally measures a different
    deviation: the two frames are not a constant apart, because under ``"global"`` each
    head carries the offset `(mu^h - mu_bar)`.
    """
    if frame not in SCORE_FRAMES:
        raise ValueError(f"unknown frame {frame!r}; pick from {SCORE_FRAMES}")
    vv = v[:, :, 1:, :] if exclude_cls else v
    mu = vv.mean(dim=2, keepdim=True)                # (B, H, 1, head_dim)
    if frame == "global":
        mu = mu.mean(dim=1, keepdim=True)            # (B, 1, 1, head_dim), all heads
    return mu


def rsa_window_scores(
    token_scores: torch.Tensor,
    window: int,
    grid: int = config.GRID,
    padded: bool = False,
) -> torch.Tensor:
    """Average-pool token scores into sliding-window scores.

    RSA: "Computing the window anomaly scores can be implemented efficiently as an
    average pooling operation on the 2D grid of token anomaly scores."

    Their prose says "average" and Algorithm 1 says "sum". For one window shape the two
    agree up to a constant, so the argmax is identical; once shapes vary they do not.

    `padded` decides the candidate set, which is what the paper leaves open when it says
    "the set of windows covering unique sets of ViT patches":

    False   only windows lying wholly inside the grid. Returns (B, G-w+1, G-w+1), and
            the index is the window's top-left corner. The number of candidates
            containing a token is then 1 at a corner, 3 on an edge and 9 in the
            interior at w=3, so a corner token can never be the centre of one.
    True    every window centred on a token, truncated at the grid edge. Returns
            (B, G, G); index `i` is a window starting at grid coordinate `i - (w-1)//2`.
            The average is over the tokens actually covered, not over the padding.
    """
    B = token_scores.shape[0]
    g = token_scores.view(B, 1, grid, grid)
    if not padded:
        return torch.nn.functional.avg_pool2d(g, kernel_size=window, stride=1).squeeze(1)
    # Asymmetric padding keeps the output at exactly `grid` for even windows too.
    lo, hi = (window - 1) // 2, window // 2
    pad = (lo, hi, lo, hi)
    total = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(g, pad, value=0.0), kernel_size=window, stride=1
    )
    covered = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(torch.ones_like(g), pad, value=0.0),
        kernel_size=window,
        stride=1,
    )
    return (total / covered).squeeze(1)


def rsa_window_size_released(patch_px: int, patch_size: int = config.PATCH_SIZE) -> int:
    """The window side the authors' released code computes, transcribed.

    `pycls/models/vision_transformer.py:82-87` of wagner-group/robust-self-attention::

        max_patches = adv_patch_size // vit_patch_size + 1
        if adv_patch_size % vit_patch_size > 1:
            max_patches += 1

    This is not `ceil(p/16) + 1`. The two agree whenever `p % 16 != 1`, which covers
    10, 20, 30, 40 and 50 px (both give 2, 3, 3, 4, 5), and differ at p = 1 (mod 16):
    at 17 px the released formula gives 2 and `ceil+1` gives 3. No RSA patch size is in
    that class, so the headline results are unaffected either way.
    """
    side = patch_px // patch_size + 1
    if patch_px % patch_size > 1:
        side += 1
    return min(side, config.GRID)


def rsa_window_size(
    patch_px: int,
    patch_size: int = config.PATCH_SIZE,
    rule: str = "ceil+1",
) -> int:
    """Tokens per side of the window RSA masks for a patch of `patch_px` pixels.

    A patch of p pixels straddles at most ceil(p/16) + 1 tokens per side when not
    grid-aligned, and covers at least ceil(p/16). The paper gives neither as a formula.
    RSA takes the patch size as a known hyperparameter; overestimating it costs clean
    accuracy (their Table 2: 93.16 at 0px down to 83.98 at 50px).

    `rule` names which reading the side comes from. `released` is the authors' own
    formula (`rsa_window_size_released`), `ceil` is the tight side, and every other rule
    uses `ceil+1` and differs in the candidate set or the masking step instead. See
    `rsa.WINDOW_RULES`. `released` and `ceil+1` coincide at all five of RSA's patch
    sizes and part company only when `patch_px % patch_size == 1`.
    """
    if rule == "released":
        return rsa_window_size_released(patch_px, patch_size)
    tight = math.ceil(patch_px / patch_size)
    side = tight if rule == "ceil" else tight + 1
    return min(side, config.GRID)


def rsa_argmax_window(window_scores: torch.Tensor) -> torch.Tensor:
    """The (row, col) of the top-scoring window per image. Returns (B, 2).

    RSA's non-differentiable step, applied once per layer.
    """
    B, G, _ = window_scores.shape
    flat = window_scores.view(B, -1).argmax(-1)
    return torch.stack([flat // G, flat % G], dim=-1)


# --------------------------------------------------------------------------
# Value-space direction decomposition
# --------------------------------------------------------------------------
def direction_decomposition(
    delta: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Split a value-space deviation into its margin-reducing and orthogonal parts.

    With ``delta = v_p - mu`` and ``u`` the unit vector along which moving reduces the
    classification margin::

        delta_parallel = <delta, u> u
        delta_perp     = delta - delta_par

    RSA penalises ``||delta||^2 = ||delta_par||^2 + ||delta_perp||^2``; to first order
    harm depends only on ``delta_par``. The attacker's stealth problem is::

        maximise <delta, u>   subject to   ||delta|| <= tau

    solved by putting the whole budget in ``u``.

    Returns `aligned_fraction` = ||delta_par||^2 / ||delta||^2 in [0, 1]. A random
    direction gives 1/head_dim.
    """
    u = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    proj = (delta * u).sum(-1, keepdim=True)          # <delta, u>
    par = proj * u
    perp = delta - par

    sq = delta.pow(2).sum(-1).clamp_min(1e-12)
    return {
        "norm": delta.norm(dim=-1),
        "parallel": proj.squeeze(-1),
        "parallel_norm": par.norm(dim=-1),
        "perp_norm": perp.norm(dim=-1),
        "aligned_fraction": par.pow(2).sum(-1) / sq,
    }


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Classification margin: true-class logit minus the best other class.

    Positive means correctly classified. Its gradient defines the direction `u` in
    `direction_decomposition`.
    """
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y[:, None], float("-inf"))
    return true - other.amax(1)
