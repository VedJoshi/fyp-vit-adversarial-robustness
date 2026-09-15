"""Random-weight ViT with the real token grid for checks without data or checkpoints."""
import torch
import torch.nn as nn

from fyp import config

H, D = 6, 64
GRID, PX = config.GRID, config.PATCH_SIZE


class _StubAttn(nn.Module):
    """The attributes `rsa.RSAAttention` copies out of a timm Attention."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(H * D, 3 * H * D)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(H * D, H * D)
        self.proj_drop = nn.Dropout(0.0)
        self.num_heads = H
        self.head_dim = D
        self.scale = D ** -0.5

    def forward(self, t):
        b, n, c = t.shape
        qkv = self.qkv(t).reshape(b, n, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop(((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(b, n, c)))


class _StubBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(H * D)
        self.attn = _StubAttn()

    def forward(self, t):
        return t + self.attn(self.norm(t))


class _StubViT(nn.Module):
    """Image in, logits out, with the 14x14 token grid the defense indexes."""

    def __init__(self, depth=2, n_classes=10):
        super().__init__()
        self.embed = nn.Conv2d(3, H * D, kernel_size=PX, stride=PX)
        self.cls = nn.Parameter(torch.randn(1, 1, H * D) * 0.02)
        self.blocks = nn.ModuleList(_StubBlock() for _ in range(depth))
        self.head = nn.Linear(H * D, n_classes)

    def forward(self, x):
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        for b in self.blocks:
            t = b(t)
        return self.head(t[:, 0])


class _Recorder(nn.Module):
    """Wraps a model and keeps the first input it is called with."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.first = None

    def forward(self, x):
        if self.first is None:
            self.first = x.detach().clone()
        return self.inner(x)
