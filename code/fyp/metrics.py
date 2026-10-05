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


def patch_token_indices(
    top: int,
    left: int,
    size: int,
    grid: int = config.GRID,
    patch_size: int = config.PATCH_SIZE,
    include_cls_offset: bool = True,
) -> list[int]:
    """Indices of tokens touched by the pixel patch, including partial overlaps.

    include_cls_offset reserves index 0 for CLS.
    """
    r0, r1 = top // patch_size, (top + size - 1) // patch_size
    c0, c1 = left // patch_size, (left + size - 1) // patch_size
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, grid - 1), min(c1, grid - 1)

    off = 1 if include_cls_offset else 0
    return [off + r * grid + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


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
        in_p[config.CLS_INDEX] = False
    return in_p, out_p


def attention_entropy(weights: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """Attention entropy per query, reduced by mean, CLS query, or not at all."""
    p = weights.clamp_min(1e-12)
    h = -(p * p.log()).sum(-1)          # (B, H, N)
    if reduce == "none":
        return h
    if reduce == "cls":
        return h[:, :, config.CLS_INDEX].mean(1)     # (B,)
    return h.mean(dim=(1, 2))                        # (B,)


def _lme(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-mean-exp over masked entries of the last axis."""
    neg_inf = torch.finfo(z.dtype).min
    masked = z.masked_fill(~mask, neg_inf)
    return torch.logsumexp(masked, dim=-1) - math.log(int(mask.sum()))


def dominance(
    logits: torch.Tensor,
    patch_tokens: Sequence[int],
    reduce: str = "mean",
    exclude_cls: bool = True,
) -> torch.Tensor:
    """Log-ratio of mean attention to patch versus non-patch tokens.

    Compute from logits to avoid softmax underflow. Positive means the patch
    receives more attention per token; exclude_cls keeps CLS out of the baseline.
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
    """Total post-softmax attention received by patch tokens."""
    idx = torch.as_tensor(list(patch_tokens), device=weights.device)
    s = weights[..., idx].sum(-1)                    # (B, H, N)
    if exclude_cls_query:
        s = s[:, :, 1:]
    if reduce == "none":
        return s
    if reduce == "cls":
        return weights[:, :, config.CLS_INDEX][..., idx].sum(-1).mean(1)
    return s.mean(dim=(1, 2))


# exp(-gap) rounds below the smallest float32 subnormal.
FLOAT32_UNDERFLOW_GAP = 150.0 * math.log(2.0)          # 103.97208...


def logit_gap(logits: torch.Tensor, reduce: str = "max") -> torch.Tensor:
    """Largest minus second-largest logit per query, before reduction."""
    top2 = logits.topk(2, dim=-1).values
    gap = top2[..., 0] - top2[..., 1]                # (B, H, N)
    if reduce == "none":
        return gap
    if reduce == "max":
        return gap.amax(dim=(1, 2))
    return gap.mean(dim=(1, 2))


def logit_span(logits: torch.Tensor, reduce: str = "none") -> torch.Tensor:
    """Largest minus smallest logit per query, before reduction."""
    span = logits.amax(dim=-1) - logits.amin(dim=-1)     # (B, H, N)
    if reduce == "none":
        return span
    if reduce == "max":
        return span.amax(dim=(1, 2))
    return span.mean(dim=(1, 2))


def underflow_fraction(logits: torch.Tensor, mode: str = "complete") -> torch.Tensor:
    """Fraction of rows exceeding the float32 underflow gap.

    complete compares the top two logits; partial compares the largest and smallest.
    """
    if mode == "complete":
        gap = logit_gap(logits, reduce="none")
    elif mode == "partial":
        gap = logit_span(logits, reduce="none")
    else:
        raise ValueError(f"unknown mode {mode!r}; pick 'complete' or 'partial'")
    return (gap > FLOAT32_UNDERFLOW_GAP).float().mean(dim=(1, 2))


# Global averages over heads and image tokens; per-head averages only tokens.
SCORE_FRAMES = ("global", "per-head")
DEFAULT_SCORE_FRAME = "global"


def rsa_token_scores(
    v: torch.Tensor,
    exclude_cls: bool = True,
    frame: str = DEFAULT_SCORE_FRAME,
    mu: torch.Tensor | None = None,
) -> torch.Tensor:
    """Head-averaged L2 distance from the chosen mean; returns (B, N_img).

    CLS is excluded by default. frame selects a global or per-head mean.
    Passing mu substitutes a reference computed elsewhere, which scores one
    image's tokens against another's mean; frame is then unused.
    """
    vv = v[:, :, 1:, :] if exclude_cls else v        # (B, H, N_img, head_dim)
    if mu is None:
        mu = score_frame_mean(v, exclude_cls=exclude_cls, frame=frame)
    scores = (vv - mu).norm(dim=-1)                  # (B, H, N_img)
    return scores.mean(1)                            # average over heads -> (B, N_img)


def score_frame_mean(
    v: torch.Tensor,
    exclude_cls: bool = True,
    frame: str = DEFAULT_SCORE_FRAME,
) -> torch.Tensor:
    """Mean used by RSA scoring, broadcastable to (B, H, N, head_dim).

    Global shape: (B, 1, 1, head_dim); per-head: (B, H, 1, head_dim).
    """
    if frame not in SCORE_FRAMES:
        raise ValueError(f"unknown frame {frame!r}; pick from {SCORE_FRAMES}")
    vv = v[:, :, 1:, :] if exclude_cls else v
    mu = vv.mean(dim=2, keepdim=True)                # (B, H, 1, head_dim)
    if frame == "global":
        mu = mu.mean(dim=1, keepdim=True)            # (B, 1, 1, head_dim), all heads
    return mu


# Anchor ranking: RSA's value anomaly, or the distance to the nearest admissible
# neighbour with the token's surrounding 3x3 block excluded.
SELECT_RULES = ("anomaly", "knnfar1")
DEFAULT_SELECT_RULE = "anomaly"


def head_mean_distances(v: torch.Tensor) -> torch.Tensor:
    """(B, 196, 196) mean over heads of the per-head token distance matrix.

    v is (B, H, 197, D) with CLS at index 0, which is dropped: CLS is not a
    candidate anchor and is not part of the image's own token population.
    """
    x = v[:, :, 1:, :].float()
    n, h = x.shape[0], x.shape[1]
    out = x.new_zeros(n, config.N_IMAGE_TOKENS, config.N_IMAGE_TOKENS)
    for j in range(h):
        out = out + torch.cdist(x[:, j], x[:, j])
    return out / h


def exclusion_mask(radius: int, grid: int = config.GRID) -> torch.Tensor:
    """(196, 196) True where j is too close to i to count as a neighbour.

    radius 0 excludes only self; radius 1 also excludes the surrounding 3x3.
    """
    idx = torch.arange(grid * grid)
    r, c = idx // grid, idx % grid
    return ((r[:, None] - r[None, :]).abs().le(radius)
            & (c[:, None] - c[None, :]).abs().le(radius))


def kth_distance(dist: torch.Tensor, k: int, radius: int) -> torch.Tensor:
    """(B, 196) distance to the k-th nearest admissible token."""
    d = dist.masked_fill(exclusion_mask(radius).to(dist.device), float("inf"))
    return d.topk(k, dim=-1, largest=False).values[..., k - 1]


def select_scores(
    v: torch.Tensor,
    frame: str = DEFAULT_SCORE_FRAME,
    rule: str = DEFAULT_SELECT_RULE,
) -> torch.Tensor:
    """Anchor ranking, (B, N_img). The defence and its evasion objective share this.

    High is anomalous under either rule, so the argmax is the anchor in both.
    """
    if rule == "anomaly":
        return rsa_token_scores(v, frame=frame)
    if rule == "knnfar1":
        return kth_distance(head_mean_distances(v), 1, 1)
    raise ValueError(f"unknown select rule {rule!r}; pick from {SELECT_RULES}")


def rsa_window_scores(
    token_scores: torch.Tensor,
    window: int,
    grid: int = config.GRID,
    padded: bool = False,
    pool: str = "mean",
) -> torch.Tensor:
    """Mean or max token score in each sliding window.

    Unpadded output (B, G-w+1, G-w+1) is indexed by top-left corner.
    Padded output (B, G, G) is indexed by anchor token; edge windows are clipped
    and their means exclude padding.
    """
    if pool not in ("mean", "max"):
        raise ValueError(f"pool must be 'mean' or 'max', got {pool!r}")
    B = token_scores.shape[0]
    g = token_scores.view(B, 1, grid, grid)
    if pool == "max":
        if not padded:
            return torch.nn.functional.max_pool2d(
                g, kernel_size=window, stride=1).squeeze(1)
        lo, hi = (window - 1) // 2, window // 2
        return torch.nn.functional.max_pool2d(
            torch.nn.functional.pad(g, (lo, hi, lo, hi), value=float("-inf")),
            kernel_size=window, stride=1).squeeze(1)
    if not padded:
        return torch.nn.functional.avg_pool2d(g, kernel_size=window, stride=1).squeeze(1)
    lo, hi = (window - 1) // 2, window // 2
    pad = (lo, hi, lo, hi)
    # Divide by covered area so padding does not dilute edge scores.
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
    """Released RSA window formula: p//s + 1, plus 1 when p%s > 1.

    Unlike ceil(p/s)+1, this gives one fewer token when p%s == 1.
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
    """Window side in tokens: released formula, ceil(p/s), or ceil(p/s)+1.

    Centred-rule sizing is handled separately by RSAConfig.for_patch.
    """
    if rule == "released":
        return rsa_window_size_released(patch_px, patch_size)
    tight = math.ceil(patch_px / patch_size)
    side = tight if rule == "ceil" else tight + 1
    return min(side, config.GRID)


def centred_coverage(top: int,
                     left: int,
                     size: int,
                     window: int,
                     grid: int = config.GRID,
                     patch_size: int = config.PATCH_SIZE) -> torch.Tensor:
    """Boolean (grid*grid,) mask of anchors whose centred window covers the patch.

    Includes the inward border shift. Covering anchors can lie outside the
    patch footprint; indices exclude CLS.
    """
    lo = (window - 1) // 2
    r0 = (top // patch_size, (top + size - 1) // patch_size)
    c0 = (left // patch_size, (left + size - 1) // patch_size)
    tok = torch.arange(grid * grid)
    br = (tok // grid - lo).clamp(0, max(grid - window, 0))
    bc = (tok % grid - lo).clamp(0, max(grid - window, 0))
    return ((br <= r0[0]) & (r0[1] < br + window)
            & (bc <= c0[0]) & (c0[1] < bc + window))


def centred_block_side(patch_px: int,
                       stride: int | None = None,
                       patch_size: int = config.PATCH_SIZE,
                       grid: int = config.GRID) -> int:
    """Window side needed to cover the patch from any token in its footprint.

    A footprint spanning k cells per axis needs side 2*k-1, capped at the grid.
    stride=None checks arbitrary offsets; otherwise check offsets reachable
    on that stride. A 20 px patch needs side 3 at stride 20, but 5 off-grid.
    """
    if patch_px <= 0:
        return 0
    offsets = (range(patch_size) if stride is None
               else {(n * stride) % patch_size for n in range(patch_size)})
    k = max((t + patch_px - 1) // patch_size - t // patch_size + 1 for t in offsets)
    return min(2 * k - 1, grid)


def rsa_argmax_window(window_scores: torch.Tensor) -> torch.Tensor:
    """Top-scoring window corner per image, shape (B, 2); ties use scan order."""
    B, G, _ = window_scores.shape
    flat = window_scores.view(B, -1).argmax(-1)
    return torch.stack([flat // G, flat % G], dim=-1)


def rsa_tied_windows(window_scores: torch.Tensor) -> torch.Tensor:
    """Count windows tied at the maximum score per image: (B, G, G) -> (B,)."""
    flat = window_scores.flatten(1)
    return (flat == flat.amax(1, keepdim=True)).sum(1)


def direction_decomposition(
    delta: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Split delta into components parallel and orthogonal to direction.

    Returns norms, signed projection, and the fraction of squared norm aligned.
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
    """True-class logit minus the largest competing logit."""
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y[:, None], float("-inf"))
    return true - other.amax(1)
