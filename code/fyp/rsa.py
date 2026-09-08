"""RSA - Robust Self-Attention (Mu & Wagner, ICML 2021 UDL Workshop).

RSA replaces the weighted mean at the heart of self-attention with a robust
aggregation: at every layer it scores each value token by its distance from the mean
image-token value, finds the highest-scoring sliding window, and neutralises that
window before attention is applied. Their Algorithm 1, per attention layer::

    mu     <- MEAN_{i in N}(v_i)                 image tokens only, CLS excluded
    s(w)   <- SUM_{i in w} ||v_i - mu||_2        for every window w in W
    w*     <- ARGMAX_W s(w)
    for i in w*:  v_i <- mu,  alpha_{.,i} <- 1/N
    z_i    <- SUM_j alpha_{i,j} v_j

The scoring half lives in `metrics`: `rsa_token_scores`, `rsa_window_scores`,
`rsa_window_size`, `rsa_argmax_window`. This module supplies the masking, the
per-layer replacement and the wiring.

`enable(model, cfg)` swaps every block's `attn` for an `RSAAttention` holding the same
parameters, so the model object is unchanged from the outside and `attacks.*`,
`diagnostics.*` and `metrics.*` operate on it as they do on the undefended model::

    with rsa.enable(clf, rsa.RSAConfig.for_patch(30)) as handle:
        adv = attacks.patch_autopgd(clf, x, y, top=0, left=0, size=30)
        acc = attacks.accuracy(clf, adv, y)

Six properties of the implementation:

1. **Per layer, independently.** Every block scores its own value tensor and masks its
   own argmax window. The window is not computed once and reused.
2. **CLS is excluded** from `mu` and from the scores, and is never masked. It stays
   available as a key to every query.
3. **Masked tokens are neutralised as keys and as values**, so they carry nothing
   distinctive to any other token. They remain queries: their own output rows are
   still computed from the surviving keys.
4. **Four knobs are config flags.** The paper fixes none of the first three: the
   window rule (`WINDOW_RULES`), what happens to a masked key's attention weight
   (`RENORMALISATIONS`), and which mean the anomaly score is taken against
   (`SCORE_FRAMES`). Each default is now the authors' released code rather than a
   reading of the paper; the alternatives are kept so earlier runs stay reproducible
   and so the spread across them can be measured. The fourth, `BACKWARDS`, picks which
   *derivative* the masking step carries: the released code's repeated-index `scatter`
   and this module's `torch.where` produce bit-identical forward tensors and different
   gradients, so an attack is not the authors' attack unless it is also selected.
   `RSAConfig.released(patch_px)` sets all four to the released reading at once.
5. **The argmax is not differentiable.** The selection runs under `no_grad` and the
   window index enters the graph only as a boolean mask. Gradients flow through the
   surviving logits and values. Rung L4 of the attack ladder is what has to get
   through the selection.
6. **Fused attention stays off.** `RSAAttention` computes the attention matrix
   explicitly, and `enable` calls `models.disable_fused_attention` so that
   `hooks.AttentionCapture` keeps working on the defended model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn as nn

from . import config, metrics, models

#: How the attention weights of a masked key are handled. The paper sets
#: `alpha_{.,i} <- 1/N` and is silent on what happens to the rest of the row.
#:
#: ``"softmax"``  the pre-softmax logits of masked keys are set to -inf, so softmax
#:                renormalises the surviving entries and every row sums to 1.
#: ``"zero"``     the post-softmax weights of masked keys are set to 0, so rows sum to
#:                less than 1 and the block output loses that share of its magnitude.
#: ``"uniform"``  the post-softmax weights of masked keys are set to `1/N_TOKENS`,
#:                the paper's literal rule with CLS counted; rows sum to neither 1 nor
#:                less than 1.
#:
#: Under ``"softmax"`` and ``"zero"`` the masked values carry zero weight, so
#: `v_i <- mu` has no effect on the output; it is applied in every mode regardless.
#:
#: ``"uniform"`` is the default because it is what the authors' code does. At
#: `pycls/models/vision_transformer.py:107` of wagner-group/robust-self-attention the
#: masked key's column is scattered with `1 / context_length`, `context_length` being
#: `num_patches ** 2 + 1` = 197, and the rows are not renormalised afterwards. The
#: constant here is `config.N_TOKENS` to match, CLS included.
#:
#: Clean accuracy does not separate the three modes well enough to have settled this.
#: Measured on 512 images, ``"zero"`` lands within 0.78 points of their Table 2 at
#: every assumed patch size against 1.18 for ``"uniform"``, which is why ``"zero"``
#: was the default before the reference code was available; ``"softmax"`` is 5.67 out,
#: because renormalising the surviving row to sum to 1 removes most of the
#: clean-accuracy cost the paper reports, giving 89.65% at 50px against their 83.98%.
#: A 0.4-point difference in Table 2 fidelity is not evidence against the mode the
#: authors actually ran. Record: `results/m5_rsa_sanity.json`.
RENORMALISATIONS = ("zero", "softmax", "uniform")
DEFAULT_RENORMALISATION = "uniform"

#: Which mean the anomaly score is taken against. Defined in `metrics` beside the
#: scoring function and named here so the defense's knobs sit together.
SCORE_FRAMES = metrics.SCORE_FRAMES
DEFAULT_SCORE_FRAME = metrics.DEFAULT_SCORE_FRAME

#: Which set of windows the argmax ranges over, and what gets masked. The paper says
#: the candidate set is "the set of windows covering unique sets of ViT patches" and
#: gives the side as no formula at all, so more than one reading is consistent with it.
#:
#: ``"ceil+1"``  side ceil(p/16)+1, only windows wholly inside the grid. The first
#:               reading implemented, and the one that matches their Table 2.
#: ``"padded"``  same side, but the candidate set is every window *centred* on a token,
#:               truncated at the grid edge. A clipped window still covers a unique set
#:               of ViT patches, so the phrase admits it.
#: ``"ceil"``    side ceil(p/16), wholly inside. The tight side; does not always contain
#:               a patch that straddles a cell border.
#: ``"multi"``   every shape in {c, c+1} x {c, c+1}, c = ceil(p/16), wholly inside,
#:               ranked by average. Shapes now differ in size, so "average" and "sum"
#:               stop agreeing and the prose's "average" is the reading taken.
#: ``"token"``   the highest-scoring token, then the ceil+1 window containing it with
#:               the highest average. From "select the image token with the greatest
#:               overall anomaly score".
#: ``"topk"``    no window: the k highest-scoring tokens masked individually, k the
#:               ceil+1 window's area. From the paper's "Single suspicious token"
#:               paragraph. Liu et al.'s Attention-Mask (ICML 2023) masks single
#:               tokens the same way, but its threat model is a pixel-budgeted
#:               attacker whose perturbation scatters across tokens, so it is a
#:               separate defense rather than a reimplementation of RSA.
#:
#: ``"ceil+1"`` is the default because it is what the authors' code computes. At
#: `pycls/models/vision_transformer.py:82-87` of wagner-group/robust-self-attention
#: the side is `p // 16 + 1`, plus one more when `p % 16 > 1`, which is the exact
#: worst case ceil((p + 15) / 16). That agrees with ceil(p/16) + 1 at every patch size
#: except p = 1 (mod 16), and the paper uses none of those: at 10, 20, 30, 40 and 50px
#: both give 2, 3, 3, 4, 5. Line 94 pools with `stride=1` and no padding, so the
#: candidate set is the unpadded one and ``"padded"`` is ruled out too.
#:
#: The other five readings are kept because the spread across them is a result in its
#: own right. The unpadded candidate set gives a corner token one candidate window
#: against an interior token's nine, so a corner patch can never be centred. On one
#: batch at 20px, with the patch re-optimised under each reading, ``"padded"`` takes a
#: corner location from 6.25% to 50.00% robust, costs about one image at the interior
#: and leaves clean accuracy at 93.75%. See `../../01_gate_a/gate-a-diagnosis.md`.
#: ``"released"``  side `p // 16 + 1 (+1 if p % 16 > 1)`, the authors' own formula,
#:                 transcribed at `metrics.rsa_window_size_released`. Identical to
#:                 ``"ceil+1"`` at 10, 20, 30, 40 and 50 px and at every other size
#:                 where `p % 16 != 1`; it exists so an exact-released control does not
#:                 have to rely on that coincidence holding.
WINDOW_RULES = ("ceil+1", "padded", "ceil", "multi", "token", "topk", "released")
DEFAULT_WINDOW_RULE = "ceil+1"

#: Which derivative the masking step carries. The forward output is identical either
#: way; only the gradient the attacker differentiates through differs.
#:
#: ``"intended"``  the natural derivative of the operator Algorithm 1 describes. The
#:                 replacement is `torch.where(masked, mu, v)`, so a masked position
#:                 passes its output gradient to `mu` once.
#: ``"released"``  the derivative the authors' released code actually has. At
#:                 `pycls/models/vision_transformer.py:101` of
#:                 wagner-group/robust-self-attention the replacement is
#:                 `v.scatter(-2, idx.repeat(1, H, context_length, D), v_mean)` with
#:                 `v_mean = v_mean_head.repeat(1, 1, context_length, 1)`. The index
#:                 tensor names the same target row `context_length` times, so the
#:                 forward merely writes that row once, but autograd routes the output
#:                 gradient back to every one of the 197 repeated source elements. The
#:                 gradient reaching `mu` is therefore `config.N_TOKENS` times the
#:                 intended one; the gradient reaching the unmasked values is the same.
#:
#: This is not a cosmetic difference. `attacks.patch_autopgd` takes `grad.sign()`, so
#: the two settings send the attack in materially different directions: on random
#: tensors at RSA's five patch sizes the input-gradient cosine is 0.21-0.45 and 34-38%
#: of gradient signs disagree, while the forward tensors are bit-exact. Every result
#: recorded before 8 September 2026 was produced under ``"intended"``, which is why
#: that is still the default; ``"released"`` is what an "attacked the authors' released
#: implementation" claim requires. `scripts/check_reference_fidelity.py` proves the
#: two implementations agree forward and that ``"released"`` matches the literal
#: scatter loop's gradient.
BACKWARDS = ("intended", "released")
DEFAULT_BACKWARD = "intended"


@dataclass(frozen=True)
class RSAConfig:
    """The defense's hyperparameters.

    `patch_px` is the patch size RSA assumes, which their threat model takes as known.
    `window` is the resulting window side in tokens; `window = 0` disables masking, so
    the wrapper reduces to ordinary attention and reproduces the undefended logits.
    That is the `0px` column of their Table 2.
    """

    patch_px: int
    window: int
    renormalise: str = DEFAULT_RENORMALISATION
    window_rule: str = DEFAULT_WINDOW_RULE
    score_frame: str = DEFAULT_SCORE_FRAME
    backward: str = DEFAULT_BACKWARD

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

    @classmethod
    def for_patch(cls, patch_px: int,
                  renormalise: str = DEFAULT_RENORMALISATION,
                  window_rule: str = DEFAULT_WINDOW_RULE,
                  score_frame: str = DEFAULT_SCORE_FRAME,
                  backward: str = DEFAULT_BACKWARD) -> "RSAConfig":
        """The config RSA uses against a `patch_px`-pixel patch.

        `patch_px = 0` gives `window = 0`, which is no masking. Otherwise the window is
        `metrics.rsa_window_size` under `window_rule`.
        """
        window = 0 if patch_px <= 0 else metrics.rsa_window_size(patch_px, rule=window_rule)
        return cls(patch_px=patch_px, window=window, renormalise=renormalise,
                   window_rule=window_rule, score_frame=score_frame, backward=backward)

    @classmethod
    def released(cls, patch_px: int) -> "RSAConfig":
        """The config that reproduces the authors' released code, forward and backward.

        Their window formula, their `1/197` renormalisation, their globally-centred
        score, and their repeated-index `scatter` derivative. This is the setting an
        "attacked RSA as released" claim needs; `for_patch` alone gives the operator
        Algorithm 1 describes, which is forward-identical and backward-different.
        """
        return cls.for_patch(patch_px, renormalise="uniform", window_rule="released",
                             score_frame="global", backward="released")

    @property
    def masked_tokens(self) -> int:
        """The nominal number of image tokens the mask removes per layer.

        Exact for `ceil+1`, `ceil`, `token` and `topk`. `padded` masks fewer at the grid
        edge and `multi` masks whatever the winning shape holds, so use
        `allowed_masked_counts` to check an observed count.
        """
        return self.window * self.window

    @property
    def allowed_masked_counts(self) -> frozenset[int]:
        """Every per-layer masked-token count this rule can legitimately produce."""
        w, g = self.window, config.GRID
        if w == 0:
            return frozenset({0})
        if self.window_rule == "padded":
            # A window centred at the edge is truncated, so each side runs over the
            # lengths a w-wide interval can have once clipped to [0, g).
            sides = {min(w, g) - d for d in range(w // 2 + 1)}
            sides = {s for s in sides if s > 0}
            return frozenset(a * b for a in sides for b in sides)
        if self.window_rule == "multi":
            c = max(w - 1, 1)
            return frozenset(a * b for a in (c, c + 1) for b in (c, c + 1))
        return frozenset({w * w})

    def as_dict(self) -> dict:
        return {
            "patch_px": self.patch_px,
            "window": self.window,
            "renormalise": self.renormalise,
            "window_rule": self.window_rule,
            "score_frame": self.score_frame,
            "backward": self.backward,
            "masked_tokens": self.masked_tokens,
        }


class _ScaleGrad(torch.autograd.Function):
    """Identity forward; multiplies the gradient by `scale` on the way back.

    Reproduces what the released `scatter` does to the gradient of the replacement
    mean without paying for `window^2` scatters of a (B, H, 197, 64) tensor per layer.
    `check_reference_fidelity.py` pins it against the literal loop.
    """

    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


class RSAAttention(nn.Module):
    """One block's attention with RSA's robust aggregation in place of the mean.

    Holds the wrapped `Attention`'s own submodules by reference, so it introduces no
    parameters and `attn.qkv` remains the same Linear that `hooks.AttentionCapture`
    hooks. Its forward is timm's explicit attention path with the masking inserted
    between the logits and the softmax.

    `last_window` is the (B, 2) grid position selected on the most recent forward
    pass, and `last_mask` the (B, N) boolean key mask, both detached.
    """

    def __init__(self, attn: nn.Module, cfg: RSAConfig):
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
        self.last_window: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None

    @torch.no_grad()
    def select(self, v: torch.Tensor) -> torch.Tensor:
        """The window to mask, as a (B, N) boolean key mask with CLS always False.

        Runs under `no_grad`: the argmax is the defense's non-differentiable step and
        nothing downstream of it needs a gradient with respect to the selection.
        """
        B = v.shape[0]
        w, g, rule = self.cfg.window, config.GRID, self.cfg.window_rule
        scores = metrics.rsa_token_scores(v, frame=self.cfg.score_frame)  # (B, N_img)

        if rule == "topk":
            # No window. The k most anomalous tokens are masked where they fall.
            k = min(w * w, scores.shape[1])
            idx = scores.topk(k, dim=1).indices                               # (B, k)
            flat = torch.zeros_like(scores, dtype=torch.bool)
            flat.scatter_(1, idx, True)
            grid = flat.view(B, g, g)
            rc = torch.stack([idx[:, 0] // g, idx[:, 0] % g], dim=-1)
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
        return mask

    def _window_corner(self, scores, w, g, rule):
        """The selected window as (row0, col0, height, width), each (B,) or scalar."""
        B = scores.shape[0]

        if rule == "padded":
            # Index i in the padded output is a window starting at grid coord i - lo.
            lo = (w - 1) // 2
            ws = metrics.rsa_window_scores(scores, w, padded=True)             # (B, G, G)
            rc = metrics.rsa_argmax_window(ws)
            r0 = (rc[:, 0] - lo).clamp(min=0)
            c0 = (rc[:, 1] - lo).clamp(min=0)
            r1 = (rc[:, 0] - lo + w).clamp(max=g)
            c1 = (rc[:, 1] - lo + w).clamp(max=g)
            return r0, c0, r1 - r0, c1 - c0

        if rule == "multi":
            # Shapes differ in size, so "average" and "sum" no longer agree and the
            # ranking is by average across all of them. Each image picks its own shape.
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

        if rule == "token":
            # The top token, then the best window containing it.
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

        # "ceil+1" and "ceil": one shape, windows wholly inside the grid.
        ws = metrics.rsa_window_scores(scores, w)
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

            # Dropout before the overwrite, as upstream does at
            # `vision_transformer.py:96-107`: softmax, attn_drop, then scatter. At
            # evaluation p = 0 and the order cannot matter; in training mode it can,
            # because dropping a masked column would otherwise scale 1/197 by 1/(1-p).
            # The "softmax" mode masks pre-softmax and has no upstream counterpart.
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
        """Change the window or the renormalisation on every installed layer."""
        cfg = cfg if cfg is not None else replace(self.cfg, **changes)
        for m in self.modules:
            m.cfg = cfg
        self.cfg = cfg
        return cfg

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
    """Install RSA on `model` in place and return the handle that reverts it.

    `layers` defaults to every block. RSA is specified as a drop-in replacement for
    all of them, so a partial installation is a diagnostic rather than the defense.
    """
    blocks = _blocks(model)
    chosen = list(range(len(blocks))) if layers is None else list(layers)

    n_fused = models.disable_fused_attention(model)
    originals: dict[int, nn.Module] = {}
    for i in chosen:
        attn = blocks[i].attn
        if isinstance(attn, RSAAttention):
            raise RuntimeError(f"layer {i} already has RSA installed; remove the old handle first")
        originals[i] = attn
        blocks[i].attn = RSAAttention(attn, cfg).to(next(attn.parameters()).device)

    return RSAHandle(blocks, originals, cfg, n_fused)


def is_enabled(model: nn.Module) -> bool:
    """Whether any block currently carries an `RSAAttention`."""
    return any(isinstance(b.attn, RSAAttention) for b in _blocks(model))


def verify(model: nn.Module, x: torch.Tensor, patch_px: int = 30, verbose: bool = True) -> dict:
    """Check the installed defense against the properties it is specified to have.

    Five asserted checks and one reported measurement, in the manner of
    `hooks.verify_capture`:

    1. **Identity at window 0** - with masking off the wrapper reproduces the undefended
       logits exactly, so the re-implemented attention matches timm's arithmetic.
    2. **Mask size** - every layer masks a count the window rule allows. That is
       `window^2` for most rules; `padded` masks fewer at the grid edge and `multi`
       masks whatever its winning shape holds.
    3. **CLS** - CLS is masked at no layer.
    4. **Per-layer independence** - the twelve layers do not all select the same window.
    5. **Gradient** - a gradient reaches the input through the defense.
    6. **Which gradient** - the input gradients under `backward="intended"` and
       `backward="released"` are compared and their cosine and sign disagreement are
       returned. This is reported, not asserted: both are legitimate settings and the
       point is that a run has to say which one it used. See `BACKWARDS`.

    Raises AssertionError naming the property that failed. Returns the measurements.
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
