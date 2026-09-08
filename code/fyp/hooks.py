"""Attention capture for timm Vision Transformers.

The hook is on `qkv` rather than on the attention matrix: the pre-softmax logits are
an intermediate value inside `Attention.forward`, not the output of any submodule, so
no hook can observe them directly. `qkv` is a Linear layer and everything else is a
deterministic function of its output, so that one tensor is captured and q, k, v, the
logits and the weights are recomputed as timm does:

    qkv  -> (B, N, 3, H, head_dim) -> permute -> q, k, v   each (B, H, N, head_dim)
    q, k -> q_norm(q), k_norm(k)                           (Identity for DeiT)
    logits = (q * scale) @ k^T                             (B, H, N, N)
    weights = softmax(logits, dim=-1)

`verify_capture` checks that reconstruction against timm's own tensors.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from . import config
from .models import disable_fused_attention


def _blocks(model: nn.Module):
    """Reach the transformer blocks through NormalizedModel or a bare timm model."""
    if hasattr(model, "blocks"):
        return model.blocks
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model.blocks
    raise AttributeError("could not find `.blocks` on this model")


@dataclass
class LayerAttention:
    """One layer's captured tensors. All shapes are (B, H, N, *)."""

    layer: int
    q: torch.Tensor | None = None          # (B, H, N, head_dim)
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    logits: torch.Tensor | None = None     # (B, H, N, N)  pre-softmax, = QK^T/sqrt(d_k)
    weights: torch.Tensor | None = None    # (B, H, N, N)  post-softmax, rows sum to 1

    # Set only when the capture runs with `detach=False`. See `grad` below.
    raw: torch.Tensor | None = field(default=None, repr=False)
    _num_heads: int = 0
    _head_dim: int = 0

    def mem_mb(self) -> float:
        total = 0
        for t in (self.q, self.k, self.v, self.logits, self.weights):
            if t is not None:
                total += t.numel() * t.element_size()
        return total / 1e6

    def grad(self, name: str) -> torch.Tensor:
        """d(loss)/d(q | k | v) after a backward pass. Requires `detach=False`.

        Not `rec.v.grad`: a forward hook on the `qkv` Linear runs after that module
        and receives its output, and the q/k/v built from it are a separate reshape on
        a parallel branch of the autograd graph. timm reshapes the same output
        independently and the loss depends on timm's copy, so `.grad` on the captured
        tensors is always None.

        Instead the gradient is retained on the qkv output, which is in the graph, and
        sliced here with the identical reshape. reshape/permute/unbind are views, so
        the mapping is exact.

        For models with a non-identity `q_norm`/`k_norm` the q and k gradients are
        taken before that norm; DeiT and ViT use Identity. The v gradient is exact.
        """
        if self.raw is None:
            raise RuntimeError("no graph captured - construct AttentionCapture(detach=False)")
        if self.raw.grad is None:
            raise RuntimeError("no gradient yet - call .backward() on a scalar first")
        B, N, _ = self.raw.shape
        g = self.raw.grad.reshape(B, N, 3, self._num_heads, self._head_dim).permute(2, 0, 3, 1, 4)
        return {"q": g[0], "k": g[1], "v": g[2]}[name]


class AttentionCapture:
    """Context manager that records per-layer attention internals.

    Memory: the logits are (B, H, N, N), ~0.93 MB per image per layer for DeiT-S, so
    a full 12-layer capture costs ~11 MB per image per quantity. `layers` and `store`
    restrict what is kept; `to_cpu` moves it off the GPU.

    Usage::

        with AttentionCapture(clf, store=("logits", "weights")) as cap:
            with torch.no_grad():
                logits_out = clf(x)
        cap[5].weights        # layer 5 attention, (B, H, N, N)

    `detach=True` (the default) detaches the captured tensors. `detach=False` keeps
    them in the graph, which `LayerAttention.grad` requires.
    """

    ALL = ("q", "k", "v", "logits", "weights")

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[int] | None = None,
        store: Iterable[str] = ("logits", "weights"),
        detach: bool = True,
        to_cpu: bool = False,
    ):
        self.model = model
        self.blocks = _blocks(model)
        self.layers = list(range(len(self.blocks))) if layers is None else list(layers)

        store = tuple(store)
        unknown = set(store) - set(self.ALL)
        if unknown:
            raise ValueError(f"unknown quantities {sorted(unknown)}; pick from {self.ALL}")
        # logits and weights both need q and k; ask for them internally.
        self.store = store
        self._need_qk = bool({"logits", "weights", "q", "k"} & set(store))

        self.detach = detach
        self.to_cpu = to_cpu

        self.data: dict[int, LayerAttention] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        # Without this the attention matrix is never materialised. See models.py.
        self.n_fused_disabled = disable_fused_attention(model)

    # -- plumbing ---------------------------------------------------------
    def _make_hook(self, layer: int, attn: nn.Module):
        def hook(_module, _inp, out):
            B, N, _ = out.shape
            qkv = out.reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn.q_norm(q), attn.k_norm(k)

            rec = LayerAttention(
                layer=layer, _num_heads=attn.num_heads, _head_dim=attn.head_dim
            )
            if not self.detach and out.requires_grad:
                # `out` is the real graph node; ours are a parallel branch. See
                # LayerAttention.grad for why this is the tensor to retain.
                out.retain_grad()
                rec.raw = out

            logits = None
            if {"logits", "weights"} & set(self.store):
                logits = (q * attn.scale) @ k.transpose(-2, -1)

            def keep(t):
                if t is None:
                    return None
                if self.detach:
                    t = t.detach()
                    if self.to_cpu:
                        t = t.cpu()
                # When detach=False we leave the tensor attached so it can be used
                # inside a differentiable attack objective. Gradients *w.r.t.* it
                # come from `LayerAttention.grad`, not from `t.grad`.
                return t

            if "q" in self.store:
                rec.q = keep(q)
            if "k" in self.store:
                rec.k = keep(k)
            if "v" in self.store:
                rec.v = keep(v)
            if "logits" in self.store:
                rec.logits = keep(logits)
            if "weights" in self.store:
                rec.weights = keep(logits.softmax(dim=-1))

            self.data[layer] = rec

        return hook

    def __enter__(self) -> "AttentionCapture":
        self.data.clear()
        for i in self.layers:
            attn = self.blocks[i].attn
            self._handles.append(attn.qkv.register_forward_hook(self._make_hook(i, attn)))
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # -- access -----------------------------------------------------------
    def __getitem__(self, layer: int) -> LayerAttention:
        return self.data[layer]

    def __len__(self) -> int:
        return len(self.data)

    def stack(self, name: str) -> torch.Tensor:
        """Stack one quantity across layers -> (L, B, H, N, *), layers in order."""
        return torch.stack([getattr(self.data[i], name) for i in sorted(self.data)])

    def total_mem_mb(self) -> float:
        return sum(rec.mem_mb() for rec in self.data.values())

    def summary(self) -> str:
        if not self.data:
            return "AttentionCapture: nothing captured yet (run a forward pass inside the block)"
        first = self.data[min(self.data)]
        shape = None
        for name in ("logits", "weights", "q"):
            t = getattr(first, name)
            if t is not None:
                shape = tuple(t.shape)
                break
        return (
            f"AttentionCapture: {len(self.data)} layers, storing {self.store}, "
            f"first-layer shape {shape}, {self.total_mem_mb():.1f} MB held, "
            f"fused_attn disabled on {self.n_fused_disabled} module(s)"
        )


@contextmanager
def _true_attention(model: nn.Module, layer: int):
    """Capture timm's own post-softmax attention, straight off the `attn_drop` input."""
    truth = {}

    def hook(_m, inp, _out):
        truth["attn"] = inp[0].detach()

    h = _blocks(model)[layer].attn.attn_drop.register_forward_hook(hook)
    try:
        yield truth
    finally:
        h.remove()


def verify_capture(
    model: nn.Module,
    x: torch.Tensor,
    layer: int = 0,
    atol: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """Check the capture against timm's own arithmetic.

    Three independent checks:

    1. **Softmax check** - the recomputed post-softmax weights match the tensor
       timm actually feeds to `attn_drop`. Confirms the qkv reshape, the head
       split and the scale factor are all right.
    2. **Row-sum check** - every attention row sums to 1. Catches a wrong softmax
       axis, which is otherwise invisible.
    3. **Output check** - reconstructing the block's attention output from the
       captured weights and values, `(attn @ v) -> transpose -> reshape -> proj`,
       reproduces the real `attn` submodule output.

    Raises AssertionError with the observed error if any check fails.
    """
    model.eval()
    blocks = _blocks(model)
    attn_mod = blocks[layer].attn

    real_out = {}

    def out_hook(_m, _i, out):
        real_out["y"] = out.detach()

    h = attn_mod.register_forward_hook(out_hook)
    try:
        with AttentionCapture(model, layers=[layer], store=("v", "logits", "weights")) as cap:
            with _true_attention(model, layer) as truth:
                with torch.no_grad():
                    model(x)
    finally:
        h.remove()

    rec = cap[layer]
    results = {}

    # 1. softmax check
    err_soft = (rec.weights - truth["attn"]).abs().max().item()
    results["softmax_max_abs_err"] = err_soft

    # 2. row sums
    err_rows = (rec.weights.sum(-1) - 1.0).abs().max().item()
    results["rowsum_max_abs_err"] = err_rows

    # 3. end-to-end output reconstruction
    B, H, N, _ = rec.weights.shape
    y = (rec.weights @ rec.v).transpose(1, 2).reshape(B, N, -1)
    y = attn_mod.proj(y)
    err_out = (y - real_out["y"]).abs().max().item()
    results["output_max_abs_err"] = err_out

    results["logit_gap_max"] = (
        rec.logits.max(dim=-1).values - rec.logits.topk(2, dim=-1).values[..., 1]
    ).max().item()

    if verbose:
        print(f"verify_capture(layer={layer}) on input {tuple(x.shape)}")
        print(f"  1. softmax match vs timm : {err_soft:.3e}   {'OK' if err_soft < atol else 'FAIL'}")
        print(f"  2. attention rows sum to 1: {err_rows:.3e}   {'OK' if err_rows < atol else 'FAIL'}")
        print(f"  3. block output rebuilt  : {err_out:.3e}   {'OK' if err_out < atol else 'FAIL'}")
        print(f"  (largest pre-softmax logit gap seen at this layer: {results['logit_gap_max']:.1f})")

    assert err_soft < atol, f"recomputed attention != timm's (max err {err_soft:.3e})"
    assert err_rows < atol, f"attention rows do not sum to 1 (max err {err_rows:.3e})"
    assert err_out < atol, f"reconstructed block output != real output (max err {err_out:.3e})"
    return results


def verify_all_layers(model: nn.Module, x: torch.Tensor, atol: float = 1e-4) -> dict:
    """Run `verify_capture` on every block.

    Returns the per-layer error dicts and the worst value of each error across layers.
    """
    n = len(_blocks(model))
    per_layer = [verify_capture(model, x, layer=i, atol=atol, verbose=False) for i in range(n)]
    worst = {
        k: max(r[k] for r in per_layer)
        for k in ("softmax_max_abs_err", "rowsum_max_abs_err", "output_max_abs_err")
    }
    print(f"all {n} layers verified (worst end-to-end reconstruction error "
          f"{worst['output_max_abs_err']:.3e})")
    return {"n_layers": n, "atol": atol, "worst": worst, "per_layer": per_layer}
