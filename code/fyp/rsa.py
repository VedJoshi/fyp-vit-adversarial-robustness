"""RSA attention and masking variants (Mu & Wagner, 2021).

Scores value anomalies, selects tokens, and replaces their values with the
image-token mean. CLS is never masked. Selection is discrete; gradients
flow through the values and attention weights.

Use enable(model, cfg) as a context manager to install and restore attention.
The intended and released backward modes preserve the replacement values
but give different gradients through their mean.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn as nn

from . import config, metrics, models

# Masked keys: zero drops their weight; softmax redistributes it; uniform sets 1/N.
RENORMALISATIONS = ("zero", "softmax", "uniform")
DEFAULT_RENORMALISATION = "uniform"

SCORE_FRAMES = metrics.SCORE_FRAMES
DEFAULT_SCORE_FRAME = metrics.DEFAULT_SCORE_FRAME

# ceil+1/released/ceil: mean-scored windows, differing in side formula.
# padded: clipped edge windows; multi: mean-ranked variable shapes.
# token: best mean window containing the peak; max: max-scored windows.
# topk: individual tokens; centred: fixed block around the peak, shifted at edges.
WINDOW_RULES = ("ceil+1", "padded", "ceil", "multi", "token", "topk", "released", "max",
                "centred")
DEFAULT_WINDOW_RULE = "ceil+1"

WINDOW_POOL = {r: ("max" if r == "max" else "mean") for r in WINDOW_RULES}

# Released scatter repeats each source N times, scaling only the mean-gradient path.
BACKWARDS = ("intended", "released")
DEFAULT_BACKWARD = "intended"

# Force exact patch tokens or a maximum-coverage candidate window.
ORACLE_MODES = ("extent", "window")


# Centred sizing depends on the allowed offsets; off-grid use needs explicit sizing.
LOCATION_STRIDE = 20

@dataclass(frozen=True)
class RSAConfig:
    """Defense settings; patch_px is in pixels, window in tokens. Zero disables masking."""

    patch_px: int
    window: int
    renormalise: str = DEFAULT_RENORMALISATION
    window_rule: str = DEFAULT_WINDOW_RULE
    score_frame: str = DEFAULT_SCORE_FRAME
    backward: str = DEFAULT_BACKWARD
    # Blocks after k reuse block k's mask; None reselects at every block.
    freeze_mask_after: int | None = None
    # Mask the union of n_windows blocks around greedily chosen anchors, suppressing
    # anchors within Chebyshev distance < nms_radius (defaults to window).
    n_windows: int = 1
    nms_radius: int | None = None

    def __post_init__(self):
        if self.renormalise not in RENORMALISATIONS:
            raise ValueError(
                f"unknown renormalise {self.renormalise!r}; pick from {RENORMALISATIONS}"
            )
        if self.window_rule not in WINDOW_RULES:
            raise ValueError(
                f"unknown window_rule {self.window_rule!r}; pick from {WINDOW_RULES}"
            )
        if self.score_frame not in SCORE_FRAMES:
            raise ValueError(
                f"unknown score_frame {self.score_frame!r}; pick from {SCORE_FRAMES}"
            )
        if self.backward not in BACKWARDS:
            raise ValueError(
                f"unknown backward {self.backward!r}; pick from {BACKWARDS}"
            )
        if not 0 <= self.window <= config.GRID:
            raise ValueError(f"window {self.window} outside [0, {config.GRID}]")
        if self.freeze_mask_after is not None and self.freeze_mask_after < 0:
            raise ValueError(
                f"freeze_mask_after {self.freeze_mask_after} is negative; "
                f"pass a block index or None"
            )
        if self.n_windows < 1:
            raise ValueError(f"n_windows {self.n_windows} is below 1")
        if self.n_windows > 1 and self.window_rule != "centred":
            raise ValueError(
                f"n_windows > 1 is implemented for the centred rule only; "
                f"got {self.window_rule!r}"
            )
        if self.nms_radius is not None and self.nms_radius < 1:
            raise ValueError(f"nms_radius {self.nms_radius} is below 1")
        # The first anchor always exists; only the ones after it can be suppressed away.
        if (self.n_windows - 1) * (2 * self.radius - 1) ** 2 >= config.GRID ** 2:
            raise ValueError(
                f"n_windows={self.n_windows} at radius {self.radius} can suppress the "
                f"whole {config.GRID}x{config.GRID} grid before {self.n_windows} anchors "
                f"are chosen"
            )

    @property
    def radius(self) -> int:
        """Chebyshev suppression radius for multi-window selection."""
        return self.window if self.nms_radius is None else self.nms_radius

    @classmethod
    def for_patch(cls, patch_px: int,
                  renormalise: str = DEFAULT_RENORMALISATION,
                  window_rule: str = DEFAULT_WINDOW_RULE,
                  score_frame: str = DEFAULT_SCORE_FRAME,
                  backward: str = DEFAULT_BACKWARD,
                  freeze_mask_after: int | None = None) -> "RSAConfig":
        """Size the mask for patch_px; centred masks use the stride-20 grid guarantee."""
        if patch_px <= 0:
            window = 0
        elif window_rule == "centred":
            window = metrics.centred_block_side(patch_px, stride=LOCATION_STRIDE)
        else:
            window = metrics.rsa_window_size(patch_px, rule=window_rule)
        return cls(patch_px=patch_px, window=window, renormalise=renormalise,
                   window_rule=window_rule, score_frame=score_frame, backward=backward,
                   freeze_mask_after=freeze_mask_after)

    @classmethod
    def released(cls, patch_px: int) -> "RSAConfig":
        """Match the released window formula, scoring, attention weights and derivative."""
        return cls.for_patch(patch_px, renormalise="uniform", window_rule="released",
                             score_frame="global", backward="released")

    @property
    def masked_tokens(self) -> int:
        """Nominal mask area; padded, multi and overlapping windows mask fewer."""
        return self.n_windows * self.window * self.window

    @property
    def allowed_masked_counts(self) -> frozenset[int]:
        """Every per-layer masked-token count this rule can legitimately produce."""
        w, g = self.window, config.GRID
        if w == 0:
            return frozenset({0})
        if self.window_rule == "padded":
            sides = {min(w, g) - d for d in range(w // 2 + 1)}
            sides = {s for s in sides if s > 0}
            return frozenset(a * b for a in sides for b in sides)
        if self.window_rule == "multi":
            c = max(w - 1, 1)
            return frozenset(a * b for a in (c, c + 1) for b in (c, c + 1))
        if self.n_windows > 1:
            # Blocks shift inward at the borders, so separated anchors can still overlap.
            return frozenset(range(w * w, self.n_windows * w * w + 1))
        return frozenset({w * w})

    def as_dict(self) -> dict:
        return {
            "patch_px": self.patch_px,
            "window": self.window,
            "renormalise": self.renormalise,
            "window_rule": self.window_rule,
            "score_frame": self.score_frame,
            "backward": self.backward,
            "freeze_mask_after": self.freeze_mask_after,
            "n_windows": self.n_windows,
            "nms_radius": self.radius,
            "masked_tokens": self.masked_tokens,
        }


def nms_anchors(scores: torch.Tensor, k: int, radius: int,
                grid: int = config.GRID) -> torch.Tensor:
    """Greedily take k anchors, suppressing anchors within Chebyshev distance < radius.

    Ties go to the lower token index, matching argmax. Returns (B, k).
    """
    s = scores.clone()
    ar = torch.arange(grid, device=scores.device)
    out = []
    for _ in range(k):
        tok = s.argmax(1)                                              # (B,)
        out.append(tok)
        near_r = ((ar[None, :] - (tok // grid)[:, None]).abs() < radius)
        near_c = ((ar[None, :] - (tok % grid)[:, None]).abs() < radius)
        s = s.masked_fill((near_r[:, :, None] & near_c[:, None, :]).flatten(1),
                          float("-inf"))
    return torch.stack(out, dim=1)


class _ScaleGrad(torch.autograd.Function):
    """Reproduce the repeated-scatter gradient without allocating repeated sources.

    Forward is identity; backward scales only the replacement-mean path.
    """

    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


class RSAAttention(nn.Module):
    """Reuse timm attention parameters with RSA masking.

    Detached diagnostics: last_window is (B, 2); last_mask is (B, N).
    """

    def __init__(self, attn: nn.Module, cfg: RSAConfig,
                 layer_idx: int = 0, shared: dict | None = None):
        super().__init__()
        self.qkv = attn.qkv
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scale = attn.scale
        self.fused_attn = False

        self.cfg = cfg
        self.layer_idx = layer_idx
        # Shared only within this installation; the selected block refreshes it each pass.
        self.shared = {} if shared is None else shared
        self.last_window: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None
        self.oracle_mask: torch.Tensor | None = None

    @torch.no_grad()
    def select(self, v: torch.Tensor) -> torch.Tensor:
        """Return a detached (B, N) key mask, with CLS always False.

        Oracle masks bypass scoring. With freeze_mask_after=k, later blocks reuse
        block k's mask, refreshed on every forward pass.
        """
        B = v.shape[0]
        if self.oracle_mask is not None:
            mask = self.oracle_mask.to(v.device).expand(B, -1)
            self.last_window = None
            self.last_mask = mask
            return mask

        frozen = self.cfg.freeze_mask_after
        if frozen is not None and self.layer_idx > frozen:
            mask = self.shared.get("mask")
            if mask is None:
                raise RuntimeError(
                    f"freeze_mask_after={frozen} but block {frozen} wrote no mask before "
                    f"block {self.layer_idx} ran; is block {frozen} installed?"
                )
            if mask.shape[0] != B:
                raise RuntimeError(
                    f"frozen mask has batch {mask.shape[0]}, this pass has {B}"
                )
            self.last_window = self.shared.get("window")
            self.last_mask = mask
            return mask

        w, g, rule = self.cfg.window, config.GRID, self.cfg.window_rule
        scores = metrics.rsa_token_scores(v, frame=self.cfg.score_frame)  # (B, N_img)

        if rule == "topk":
            k = min(w * w, scores.shape[1])
            idx = scores.topk(k, dim=1).indices                               # (B, k)
            flat = torch.zeros_like(scores, dtype=torch.bool)
            flat.scatter_(1, idx, True)
            grid = flat.view(B, g, g)
            rc = torch.stack([idx[:, 0] // g, idx[:, 0] % g], dim=-1)
        elif self.cfg.n_windows > 1:
            anchors = nms_anchors(scores, self.cfg.n_windows, self.cfg.radius, g)
            lo = (w - 1) // 2
            r0 = (anchors // g - lo).clamp(0, g - w)                   # (B, k)
            c0 = (anchors % g - lo).clamp(0, g - w)
            ar = torch.arange(g, device=v.device)
            rows = (ar >= r0[..., None]) & (ar < (r0 + w)[..., None])  # (B, k, G)
            cols = (ar >= c0[..., None]) & (ar < (c0 + w)[..., None])
            grid = (rows[:, :, :, None] & cols[:, :, None, :]).any(1)  # (B, G, G)
            rc = torch.stack([r0[:, 0], c0[:, 0]], dim=-1)             # highest anchor
        else:
            r0, c0, h, wd = self._window_corner(scores, w, g, rule)
            ar = torch.arange(g, device=v.device)
            rows = (ar[None, :] >= r0[:, None]) & (ar[None, :] < (r0 + h)[:, None])
            cols = (ar[None, :] >= c0[:, None]) & (ar[None, :] < (c0 + wd)[:, None])
            grid = rows[:, :, None] & cols[:, None, :]                        # (B, G, G)
            rc = torch.stack([r0, c0], dim=-1)

        cls_col = torch.zeros(B, 1, dtype=torch.bool, device=v.device)
        mask = torch.cat([cls_col, grid.reshape(B, -1)], dim=1)               # (B, N)

        self.last_window = rc
        self.last_mask = mask
        if frozen is not None and self.layer_idx == frozen:
            # Refresh on every forward pass, including each attack iterate.
            self.shared["mask"], self.shared["window"] = mask, rc
        return mask

    def _window_corner(self, scores, w, g, rule):
        """The selected window as (row0, col0, height, width), each (B,) or scalar."""
        B = scores.shape[0]

        if rule == "padded":
            lo = (w - 1) // 2
            ws = metrics.rsa_window_scores(scores, w, padded=True)             # (B, G, G)
            rc = metrics.rsa_argmax_window(ws)
            r0 = (rc[:, 0] - lo).clamp(min=0)
            c0 = (rc[:, 1] - lo).clamp(min=0)
            r1 = (rc[:, 0] - lo + w).clamp(max=g)
            c1 = (rc[:, 1] - lo + w).clamp(max=g)
            return r0, c0, r1 - r0, c1 - c0

        if rule == "multi":
            c = max(w - 1, 1)
            best_s = torch.full((B,), float("-inf"), device=scores.device)
            best_r = torch.zeros(B, dtype=torch.long, device=scores.device)
            best_c = torch.zeros(B, dtype=torch.long, device=scores.device)
            best_h = torch.zeros(B, dtype=torch.long, device=scores.device)
            best_w = torch.zeros(B, dtype=torch.long, device=scores.device)
            for h in (c, c + 1):
                for wd in (c, c + 1):
                    if h > g or wd > g:
                        continue
                    pooled = torch.nn.functional.avg_pool2d(
                        scores.view(B, 1, g, g), kernel_size=(h, wd), stride=1
                    ).squeeze(1)
                    cols = pooled.shape[-1]
                    val, flat = pooled.flatten(1).max(1)
                    take = val > best_s
                    best_r = torch.where(take, flat // cols, best_r)
                    best_c = torch.where(take, flat % cols, best_c)
                    best_h = torch.where(take, torch.full_like(best_h, h), best_h)
                    best_w = torch.where(take, torch.full_like(best_w, wd), best_w)
                    best_s = torch.where(take, val, best_s)
            return best_r, best_c, best_h, best_w

        if rule == "centred":
            # An even window has no centre token; the anchor takes the upper-left
            # of the two central positions.
            lo = (w - 1) // 2
            tok = scores.argmax(1)
            r0 = (tok // g - lo).clamp(0, g - w)
            c0 = (tok % g - lo).clamp(0, g - w)
            return r0, c0, w, w

        if rule == "token":
            ws = metrics.rsa_window_scores(scores, w)                          # (B, n, n)
            n = ws.shape[-1]
            tok = scores.argmax(1)
            tr, tc = tok // g, tok % g
            ar = torch.arange(n, device=scores.device)
            ok_r = (ar[None, :] <= tr[:, None]) & (ar[None, :] + w > tr[:, None])
            ok_c = (ar[None, :] <= tc[:, None]) & (ar[None, :] + w > tc[:, None])
            allowed = ok_r[:, :, None] & ok_c[:, None, :]
            masked = ws.masked_fill(~allowed, float("-inf"))
            rc = metrics.rsa_argmax_window(masked)
            return rc[:, 0], rc[:, 1], w, w

        ws = metrics.rsa_window_scores(scores, w, pool=WINDOW_POOL[rule])
        rc = metrics.rsa_argmax_window(ws)
        return rc[:, 0], rc[:, 1], w, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        logits = (q * self.scale) @ k.transpose(-2, -1)       # (B, H, N, N)

        if self.cfg.window > 0:
            mask = self.select(v)
            keys = mask[:, None, None, :]                     # (B, 1, 1, N) over keys
            values = mask[:, None, :, None]                   # (B, 1, N, 1) over tokens

            mu = v[:, :, 1:, :].mean(dim=2, keepdim=True)     # (B, H, 1, head_dim)
            if self.cfg.backward == "released":
                mu = _ScaleGrad.apply(mu, float(config.N_TOKENS))
            v = torch.where(values, mu.expand_as(v), v)

            # Match upstream ordering: dropout precedes the masked-column overwrite.
            if self.cfg.renormalise == "softmax":
                attn = self.attn_drop(logits.masked_fill(keys, float("-inf")).softmax(dim=-1))
            elif self.cfg.renormalise == "zero":
                attn = self.attn_drop(logits.softmax(dim=-1)).masked_fill(keys, 0.0)
            else:  # "uniform" - the paper's literal alpha_{.,i} <- 1/N, CLS counted
                attn = self.attn_drop(logits.softmax(dim=-1)).masked_fill(
                    keys, 1.0 / config.N_TOKENS)
        else:
            self.last_window = None
            self.last_mask = None
            attn = self.attn_drop(logits.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))

    def extra_repr(self) -> str:
        return (f"window={self.cfg.window}, renormalise={self.cfg.renormalise!r}, "
                f"score_frame={self.cfg.score_frame!r}, backward={self.cfg.backward!r}")


def _candidate_windows(rule: str, window: int,
                       grid: int = config.GRID) -> list[tuple[int, int, int, int]]:
    """Candidate (row, col, height, width) windows for the rule; topk has none."""
    if rule == "topk":
        raise ValueError("window_rule 'topk' masks single tokens and has no window set")
    if rule == "padded":
        lo = (window - 1) // 2
        out = []
        for i in range(grid):
            for j in range(grid):
                r0, c0 = max(i - lo, 0), max(j - lo, 0)
                r1, c1 = min(i - lo + window, grid), min(j - lo + window, grid)
                out.append((r0, c0, r1 - r0, c1 - c0))
        return out
    if rule == "multi":
        c = max(window - 1, 1)
        shapes = [(h, w) for h in (c, c + 1) for w in (c, c + 1) if h <= grid and w <= grid]
    else:
        shapes = [(window, window)]
    return [(r, c, h, w) for h, w in shapes
            for r in range(grid - h + 1) for c in range(grid - w + 1)]


def oracle_window(top: int, left: int, size: int, window: int,
                  window_rule: str = DEFAULT_WINDOW_RULE,
                  grid: int = config.GRID) -> tuple[int, int, int, int]:
    """Candidate window covering the most patch tokens.

    Returns (row, col, height, width). Ties prefer lowest row, then column,
    then smallest area. Full coverage is not guaranteed if the window is too small.
    """
    idx = metrics.patch_token_indices(top, left, size, grid=grid,
                                      include_cls_offset=False)
    cells = [(i // grid, i % grid) for i in idx]

    def rank(win: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        r, c, h, w = win
        covered = sum(r <= pr < r + h and c <= pc < c + w for pr, pc in cells)
        return (-covered, r, c, h * w)

    return min(_candidate_windows(window_rule, window, grid), key=rank)


class RSAHandle:
    """Live installation of RSA on a model. Reverts on `remove()` or on block exit."""

    def __init__(self, blocks, originals: dict[int, nn.Module], cfg: RSAConfig, n_fused: int):
        self.blocks = blocks
        self.originals = originals
        self.cfg = cfg
        self.n_fused_disabled = n_fused
        self.active = True

    @property
    def layers(self) -> list[int]:
        return sorted(self.originals)

    @property
    def modules(self) -> list[RSAAttention]:
        """The installed `RSAAttention` modules, in layer order."""
        return [self.blocks[i].attn for i in self.layers]

    def set_config(self, cfg: RSAConfig | None = None, **changes) -> RSAConfig:
        """Update every installed layer and clear the previous frozen mask."""
        cfg = cfg if cfg is not None else replace(self.cfg, **changes)
        for m in self.modules:
            m.cfg = cfg
            # Do not reuse a mask selected under the previous config.
            m.shared.clear()
        self.cfg = cfg
        return cfg

    def set_oracle(self, top: int, left: int, size: int, mode: str = "extent",
                   device: torch.device | None = None) -> torch.Tensor:
        """Install a (1, N) mask derived from the known patch location.

        extent masks exactly the patch tokens; window chooses a maximum-coverage
        candidate. Both retain the normal value-replacement step.
        """
        if mode not in ORACLE_MODES:
            raise ValueError(f"unknown oracle mode {mode!r}; pick from {ORACLE_MODES}")
        if mode == "extent":
            idx = metrics.patch_token_indices(top, left, size)
        else:
            r, c, h, w = oracle_window(top, left, size, self.cfg.window,
                                       self.cfg.window_rule)
            idx = [1 + rr * config.GRID + cc
                   for rr in range(r, r + h) for cc in range(c, c + w)]
        mask = torch.zeros(1, config.N_TOKENS, dtype=torch.bool, device=device)
        mask[0, idx] = True
        for m in self.modules:
            m.oracle_mask = mask
        return mask

    def clear_oracle(self) -> None:
        """Restore score-based window selection on every layer."""
        for m in self.modules:
            m.oracle_mask = None

    def windows(self) -> dict[int, torch.Tensor]:
        """The (B, 2) window each layer selected on the most recent forward pass."""
        return {i: m.last_window for i, m in zip(self.layers, self.modules)}

    def masks(self) -> dict[int, torch.Tensor]:
        """The (B, N) boolean key mask each layer applied on the most recent pass."""
        return {i: m.last_mask for i, m in zip(self.layers, self.modules)}

    def remove(self) -> None:
        if not self.active:
            return
        for i, attn in self.originals.items():
            self.blocks[i].attn = attn
        self.active = False

    def __enter__(self) -> "RSAHandle":
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def __repr__(self) -> str:
        state = "active" if self.active else "removed"
        return (f"RSAHandle({state}, {len(self.originals)} layers, "
                f"window={self.cfg.window}, renormalise={self.cfg.renormalise!r})")


def _blocks(model: nn.Module):
    """The transformer blocks, through `NormalizedModel` or a bare timm model."""
    if hasattr(model, "blocks"):
        return model.blocks
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model.blocks
    raise AttributeError("could not find `.blocks` on this model")


def enable(
    model: nn.Module,
    cfg: RSAConfig,
    layers: Sequence[int] | None = None,
) -> RSAHandle:
    """Install RSA in place; the returned context manager restores attention.

    layers defaults to all blocks. Partial installation is for diagnostics.
    """
    blocks = _blocks(model)
    chosen = list(range(len(blocks))) if layers is None else list(layers)
    if cfg.freeze_mask_after is not None and cfg.freeze_mask_after not in chosen:
        raise ValueError(
            f"freeze_mask_after={cfg.freeze_mask_after} but that block is not installed; "
            f"the blocks after it would have no mask to reuse"
        )

    n_fused = models.disable_fused_attention(model)
    originals: dict[int, nn.Module] = {}
    shared: dict = {}
    for i in chosen:
        attn = blocks[i].attn
        if isinstance(attn, RSAAttention):
            raise RuntimeError(f"layer {i} already has RSA installed; remove the old handle first")
        originals[i] = attn
        blocks[i].attn = RSAAttention(attn, cfg, layer_idx=i, shared=shared).to(
            next(attn.parameters()).device)

    return RSAHandle(blocks, originals, cfg, n_fused)


def is_enabled(model: nn.Module) -> bool:
    """Whether any block currently carries an `RSAAttention`."""
    return any(isinstance(b.attn, RSAAttention) for b in _blocks(model))


def verify(model: nn.Module, x: torch.Tensor, patch_px: int = 30, verbose: bool = True) -> dict:
    """Check identity, mask sizes, CLS exclusion, layer selection and input gradients.

    Also report the difference between intended and released input gradients.
    """
    if is_enabled(model):
        raise RuntimeError("remove the existing RSA handle before verifying")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        base = model(x).clone()

    with enable(model, RSAConfig.for_patch(0)):
        with torch.no_grad():
            identity_err = (model(x) - base).abs().max().item()

    cfg = RSAConfig.for_patch(patch_px)
    with enable(model, cfg) as handle:
        with torch.no_grad():
            model(x)
        masks = handle.masks()
        counts = sorted({int(c) for m in masks.values() for c in m.sum(1)})
        cls_masked = any(bool(m[:, config.CLS_INDEX].any()) for m in masks.values())
        wins = torch.stack([handle.windows()[i] for i in handle.layers])
        all_same = bool(all(torch.equal(wins[0], wins[i]) for i in range(len(wins))))

    grads = {}
    for mode in BACKWARDS:
        xg = x.clone().detach().requires_grad_(True)
        with enable(model, replace(cfg, backward=mode)):
            (grads[mode],) = torch.autograd.grad(model(xg).sum(), xg)
    grad = grads[cfg.backward]
    grad_sum = grad.abs().sum().item()

    gi, gr = grads["intended"].flatten(), grads["released"].flatten()
    backward_cos = float(torch.nn.functional.cosine_similarity(gi, gr, dim=0))
    backward_sign_disagreement = float((gi.sign() != gr.sign()).float().mean())

    if was_training:
        model.train()

    out = {
        "patch_px": patch_px,
        "window": cfg.window,
        "identity_max_abs_err": identity_err,
        "masked_token_counts": counts,
        "expected_masked_tokens": sorted(cfg.allowed_masked_counts),
        "cls_ever_masked": cls_masked,
        "all_layers_same_window": all_same,
        "n_distinct_windows": len({tuple(r) for r in wins.reshape(-1, 2).tolist()}),
        "input_grad_abs_sum": grad_sum,
        "backward": cfg.backward,
        "backward_grad_cosine": backward_cos,
        "backward_sign_disagreement": backward_sign_disagreement,
    }

    if verbose:
        print(f"rsa.verify(patch_px={patch_px}) on input {tuple(x.shape)}")
        print(f"  1. window 0 == undefended : {identity_err:.3e}   "
              f"{'OK' if identity_err == 0.0 else 'FAIL'}")
        print(f"  2. tokens masked per layer: {counts}   allowed {sorted(cfg.allowed_masked_counts)}")
        print(f"  3. CLS masked anywhere    : {cls_masked}")
        print(f"  4. distinct windows       : {out['n_distinct_windows']} over "
              f"{len(wins)} layers x {x.shape[0]} images")
        print(f"  5. |d(sum logits)/dx|     : {grad_sum:.3e}   (backward={cfg.backward!r})")
        print(f"  6. intended vs released dx: cosine {backward_cos:.4f}, "
              f"{100 * backward_sign_disagreement:.1f}% of signs disagree")

    assert identity_err == 0.0, f"window 0 does not reproduce the undefended logits ({identity_err:.3e})"
    allowed = cfg.allowed_masked_counts
    assert set(counts) <= allowed, f"masked token counts {counts} not all in {sorted(allowed)}"
    assert not cls_masked, "CLS was masked"
    assert not all_same, "every layer selected the same window"
    assert grad_sum > 0, "no gradient reaches the input through the defense"
    return out
