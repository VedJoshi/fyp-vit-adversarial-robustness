"""Model loading, RSA-style head slicing, and the [0,1]-space wrapper.

Normalisation is applied inside `NormalizedModel` rather than in the dataset
transform, so attacks operate on [0,1] pixels. `slice_head` adapts the 1000-way
classifier to a class subset without training.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import timm
import torch
import torch.nn as nn

from . import config


class NormalizedModel(nn.Module):
    """Wraps a timm model so it accepts images in [0, 1].

    Applies the ImageNet mean/std the backbone was trained with, so an epsilon in
    [0,1] space is the same size on every channel.
    """

    def __init__(self, model: nn.Module, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x in [0,1]
        return self.model((x - self.mean) / self.std)

    @property
    def blocks(self):
        """The wrapped model's transformer blocks."""
        return self.model.blocks


def slice_head(model: nn.Module, class_indices: Sequence[int]) -> nn.Module:
    """Cut a 1000-way classifier down to the given ImageNet class indices.

    RSA section 4.3:

        "We adapt a pre-trained network, trained on the full ImageNet dataset, to
         our 100-class subset by replacing the original 1000-way linear layer L
         with a constructed 100-way linear layer L'. The weight matrix of L'
         consists of the submatrix of L's weight matrix corresponding to the 100
         classes of interest. The bias vector is similarly cut down to size, and
         all other weights are left unchanged."

    There is no training step. Output logit ordering follows `class_indices` as
    given; `data.py` supplies it sorted.
    """
    old = model.get_classifier()
    if not isinstance(old, nn.Linear):
        raise TypeError(f"expected a Linear classifier, got {type(old).__name__}")
    if old.out_features == len(class_indices):
        return model  # already sliced

    idx = torch.as_tensor(list(class_indices), dtype=torch.long)
    new = nn.Linear(old.in_features, len(idx))
    with torch.no_grad():
        new.weight.copy_(old.weight[idx])
        if old.bias is not None:
            new.bias.copy_(old.bias[idx])
        else:
            new.bias.zero_()
    model.reset_classifier(num_classes=len(idx))
    model.get_classifier().load_state_dict(new.state_dict())
    return model


def disable_fused_attention(model: nn.Module) -> int:
    """Force timm onto its explicit attention path. Returns how many blocks changed.

    timm >= 1.0 uses `F.scaled_dot_product_attention` when `attn.fused_attn` is True.
    That kernel does not materialise the attention matrix, so the attention weights and
    pre-softmax logits do not exist as tensors and hooks capture nothing. `load_model`
    and `hooks.AttentionCapture` both call this.
    """
    n = 0
    for module in model.modules():
        if hasattr(module, "fused_attn") and module.fused_attn:
            module.fused_attn = False
            n += 1
    return n


@contextmanager
def attention_scaling(model: nn.Module, lam: float) -> Iterator[int]:
    """Temporarily multiply every block's pre-softmax attention logits by `lam`.

    This is Jain & Dutta's Eq. 2, `softmax(lam * QK^T / sqrt(d)) V`, and the mechanism
    behind their Attention-Aware Scaling: a small `lam` flattens the softmax, which
    restores gradient flow when attention has saturated hard enough that the non-maximal
    entries underflow to zero. Their Table 1 has `lam = 1e-2` giving their strongest
    attack and `lam = 10` their weakest, so scaling *down* is what strengthens.

    timm computes `q = q * self.scale` and then `q @ k.transpose(-2, -1)`, so scaling
    `attn.scale` scales the logits and nothing else. `lam = 1.0` is the identity.

    Used to wrap an *attack*, not an evaluation: the attack differentiates through the
    scaled network and the adversarial example it returns is then scored on the
    unscaled one. Scoring inside the block would measure a different model.

    Raises if any block still has `fused_attn` set, because
    `F.scaled_dot_product_attention` is called without a `scale` argument and computes
    its own `1/sqrt(E)`. `attn.scale` is ignored on that path, so this would silently
    do nothing - the same failure mode `disable_fused_attention` exists to prevent.

    Yields the number of blocks scaled, and restores every original scale on exit,
    including when the body raises.
    """
    attns = [b.attn for b in _blocks(model)]

    fused = [i for i, a in enumerate(attns) if getattr(a, "fused_attn", False)]
    if fused:
        raise RuntimeError(
            f"fused attention is on at blocks {fused}; `attn.scale` is ignored on that "
            "path, so scaling it would do nothing. Call disable_fused_attention first."
        )

    originals = [a.scale for a in attns]
    try:
        for a in attns:
            a.scale = a.scale * lam
        yield len(attns)
    finally:
        for a, s0 in zip(attns, originals):
            a.scale = s0


def _blocks(model: nn.Module):
    """The transformer blocks, through NormalizedModel or a bare timm model."""
    if hasattr(model, "blocks"):
        return model.blocks
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model.blocks
    raise AttributeError("could not find `.blocks` on this model")


#: Checkpoint containers pycls and torch write around a state dict, tried in order.
_STATE_KEYS = ("model_state", "state_dict", "model", "net")


def _unwrap_state_dict(obj) -> dict:
    """Returns the tensor state dict inside a checkpoint container, keys unprefixed.

    Strips a `module.` prefix left by DataParallel and a `model.` prefix left by
    pycls wrappers.
    """
    state = obj
    for key in _STATE_KEYS:
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError(f"no state dict found in checkpoint (got {type(state).__name__})")

    out = {}
    for k, v in state.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def load_rsa_checkpoint(
    path,
    model_name: str = "vit_small_patch16_224",
    device: torch.device | None = None,
    strict: bool = True,
) -> tuple[NormalizedModel, dict]:
    """Load RSA's released ImageNet-100 ViT-small into the timm architecture.

    `path` is their published checkpoint. Their `VisionTransformer` names its
    submodules `patch_embed`, `cls_token`, `pos_embed`, `blocks.<i>.norm1`,
    `blocks.<i>.attn.{qkv,proj}`, `blocks.<i>.norm2`, `blocks.<i>.mlp.{fc1,fc2}`,
    `norm` and `head`, which is timm's naming, so the state dict transfers without a
    key map.

    The checkpoint is already 100-way, so `slice_head` is not applied and no class
    subset is passed: the head's output order is the sorted 100 wnids of
    `data.rsa_class_wnids`.

    Returns `(clf, info)` in the same shape as `load_model`. `info` gains
    `checkpoint`, `missing_keys` and `unexpected_keys`; with `strict=True` a
    non-empty either raises.
    """
    device = device or config.get_device()
    path = Path(path)

    model = timm.create_model(model_name, pretrained=False, num_classes=config.N_CLASSES)
    cfg = timm.data.resolve_data_config({}, model=model)

    # The checkpoint is a third-party download, so it is unpickled under
    # `weights_only=True`, which admits tensors and plain containers and refuses
    # arbitrary objects. A pycls checkpoint that also stores its config object will
    # fail here; the message says so rather than falling back to a full unpickle.
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"{path.name} did not load under weights_only=True: {exc}\n"
            "The file contains more than tensors and plain containers. Inspect it "
            "before loading it any other way; a full unpickle executes whatever the "
            "file contains."
        ) from exc
    state = _unwrap_state_dict(blob)
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"state dict does not match {model_name}.\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}\n"
            f"Pass strict=False to load anyway and inspect info['missing_keys']."
        )

    model = model.eval().to(device)
    n_switched = disable_fused_attention(model)

    attn0 = model.blocks[0].attn
    info = {
        "model_name": model_name,
        "checkpoint": str(path),
        "mean": cfg["mean"],
        "std": cfg["std"],
        "depth": len(model.blocks),
        "num_heads": attn0.num_heads,
        "head_dim": attn0.head_dim,
        "scale": attn0.scale,
        "embed_dim": model.embed_dim,
        "num_classes": model.get_classifier().out_features,
        "n_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "fused_attn_disabled": n_switched,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "device": str(device),
    }

    clf = NormalizedModel(model, cfg["mean"], cfg["std"]).eval().to(device)
    return clf, info


def load_model(
    class_indices: Sequence[int] | None = None,
    model_name: str = config.MODEL_NAME,
    device: torch.device | None = None,
    pretrained: bool = True,
) -> tuple[NormalizedModel, dict]:
    """Load the victim model, optionally sliced to a class subset.

    Returns `(clf, info)`: `clf` takes [0,1] images, `info` records architecture
    facts (heads, depth, head_dim, scale, mean/std).
    """
    device = device or config.get_device()

    model = timm.create_model(model_name, pretrained=pretrained)
    cfg = timm.data.resolve_data_config({}, model=model)

    if class_indices is not None:
        model = slice_head(model, class_indices)

    model = model.eval().to(device)
    n_switched = disable_fused_attention(model)

    attn0 = model.blocks[0].attn
    info = {
        "model_name": model_name,
        "mean": cfg["mean"],
        "std": cfg["std"],
        "depth": len(model.blocks),
        "num_heads": attn0.num_heads,
        "head_dim": attn0.head_dim,
        "scale": attn0.scale,
        "embed_dim": model.embed_dim,
        "num_classes": model.get_classifier().out_features,
        "n_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "fused_attn_disabled": n_switched,
        "device": str(device),
    }

    clf = NormalizedModel(model, cfg["mean"], cfg["std"]).eval().to(device)
    return clf, info
