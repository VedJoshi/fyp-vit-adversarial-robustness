"""Differential test of `fyp.rsa` against RSA's released implementation.

    python code/scripts/check_reference_fidelity.py

The reference block in `reference_trim` is transcribed line for line from
`pycls/models/vision_transformer.py:81-107` of wagner-group/robust-self-attention,
identified by RSA author Norman Mu as the released implementation. Nothing here
reads the paper: it compares this harness's arithmetic against the code the
authors ran.

What it pins, at all five of RSA's patch sizes:

* the window side, against `p // 16 + 1 (+1 if p % 16 > 1)` at their line 84, for
  `rule="released"` at every size from 1 to 64 and not only where `ceil+1` happens to
  agree with it;
* the anomaly score under `score_frame="global"`, against their lines 88-93, which
  centre every head on the head-averaged mean value vector;
* the set of tokens the argmax masks, against their lines 94-107;
* the replacement value and the masked key's attention weight, 1/197;
* **the gradient.** Forward equality is not reproduction. Their line 101 scatters with
  an index that names the same row 197 times, so autograd sends the output gradient to
  all 197 repeated source elements, while `torch.where` sends it once. The two forward
  tensors are bit-exact and the input gradients are not: on random tensors the cosine
  is 0.21-0.45 and a third of the signs disagree. `patch_autopgd` takes `grad.sign()`,
  so an attack differentiating the wrong one is not the authors' attack. The check
  pins `backward="released"` against the literal scatter loop and asserts that
  `backward="intended"` really does differ, so neither can silently become the other.

It also runs `rsa.verify` on a small stub model, which had no caller before.

Run it after any change to `rsa.RSAAttention` or `metrics.rsa_token_scores`. It needs
no model weights and no dataset, so it runs anywhere in a couple of seconds.
"""
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fyp import config, metrics, rsa  # noqa: E402

torch.manual_seed(0)

B, H, D = 4, 6, 64
NP = 14                      # num_patches per side
CTX = NP * NP + 1            # context_length = 197


def reference_trim(v, w, adv_patch_size, vit_patch_size=16):
    """Lines 81-107, transcribed. Returns (v, w, masked_index_set_per_image)."""
    num_patches = NP
    max_patches = adv_patch_size // vit_patch_size + 1
    context_length = num_patches ** 2 + 1
    if adv_patch_size % vit_patch_size > 1:
        max_patches += 1
    v_image = v[:, :, 1:]
    v_mean_head = torch.mean(v_image, dim=2, keepdim=True)
    v_mean_image = torch.mean(v_mean_head, dim=1, keepdim=True)[:, None]
    v_image = v_image.reshape(B, H, num_patches, num_patches, D)
    distances = torch.mean(torch.norm(v_image - v_mean_image, dim=-1), dim=1)
    pooled = F.avg_pool2d(distances, max_patches, stride=1)
    idxs = torch.argmax(pooled.reshape(B, -1), dim=-1)
    col_idxs = idxs // (num_patches - max_patches + 1)
    row_idxs = idxs % (num_patches - max_patches + 1)
    v_mean = v_mean_head.repeat(1, 1, context_length, 1)
    masked = [set() for _ in range(B)]
    for i in range(max_patches):
        for j in range(max_patches):
            outlier_idx = num_patches * (col_idxs + i) + (row_idxs + j) + 1
            for b in range(B):
                masked[b].add(int(outlier_idx[b]))
            outlier_idx = outlier_idx[:, None, None, None]
            v = v.scatter(-2, outlier_idx.repeat(1, H, context_length, D), v_mean)
            w = w.scatter(-1, outlier_idx.repeat(1, H, context_length, 1),
                          1 / context_length)
    return v, w, masked, distances, max_patches


class _StubAttn(nn.Module):
    """The attributes RSAAttention copies out of a timm Attention."""

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
        """timm's explicit attention, so `rsa.verify` has an undefended model to match."""
        b, n, c = t.shape
        qkv = self.qkv(t).reshape(b, n, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop(((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(b, n, c)))


failures = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('   ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


print("defaults")
check("renormalise", rsa.DEFAULT_RENORMALISATION == "uniform", rsa.DEFAULT_RENORMALISATION)
check("window_rule", rsa.DEFAULT_WINDOW_RULE == "ceil+1", rsa.DEFAULT_WINDOW_RULE)
check("score_frame", rsa.DEFAULT_SCORE_FRAME == "global", rsa.DEFAULT_SCORE_FRAME)
check("uniform constant is 197", config.N_TOKENS == 197, str(config.N_TOKENS))

print("\nwindow size against the reference formula, per patch size")
for size in (10, 20, 30, 40, 50):
    ref_w = size // 16 + 1 + (1 if size % 16 > 1 else 0)
    mine = metrics.rsa_window_size(size, rule="ceil+1")
    check(f"{size}px", ref_w == mine, f"ref {ref_w}, ceil+1 {mine}")

print("\nwindow size, rule='released' against the reference formula, 1-64px")
bad = [p for p in range(1, 65)
       if metrics.rsa_window_size(p, rule="released")
       != min(p // 16 + 1 + (1 if p % 16 > 1 else 0), config.GRID)]
check("released matches at every size", not bad, f"differs at {bad}" if bad else "1..64")
diverge = [p for p in range(1, 65)
           if metrics.rsa_window_size(p, rule="released")
           != metrics.rsa_window_size(p, rule="ceil+1")]
check("ceil+1 differs only at p = 1 (mod 16)", diverge == [1, 17, 33, 49], str(diverge))

print("\ntoken scores, frame='global' against reference lines 88-93")
v = torch.randn(B, H, CTX, D)
_, _, _, ref_dist, _ = reference_trim(v.clone(), torch.rand(B, H, CTX, CTX), 20)
mine_scores = metrics.rsa_token_scores(v, frame="global")
check("max abs diff == 0",
      torch.equal(mine_scores, ref_dist.reshape(B, NP * NP)),
      f"{(mine_scores - ref_dist.reshape(B, NP * NP)).abs().max().item():.3e}")

per_head = metrics.rsa_token_scores(v, frame="per-head")
check("per-head frame differs from global",
      not torch.allclose(per_head, mine_scores),
      f"max diff {(per_head - mine_scores).abs().max().item():.3f}")

print("\nselected window and masked token set, 20px")
for size in (10, 20, 30, 40, 50):
    v = torch.randn(B, H, CTX, D)
    w0 = torch.rand(B, H, CTX, CTX).softmax(-1)
    ref_v, ref_w, ref_masked, _, ref_side = reference_trim(v.clone(), w0.clone(), size)

    cfg = rsa.RSAConfig.for_patch(size)
    module = rsa.RSAAttention(_StubAttn(), cfg)
    mask = module.select(v)
    mine_masked = [set(mask[b].nonzero().flatten().tolist()) for b in range(B)]

    check(f"{size}px window side", cfg.window == ref_side, f"{cfg.window} vs {ref_side}")
    check(f"{size}px masked token set", mine_masked == ref_masked,
          "" if mine_masked == ref_masked
          else f"mine {sorted(mine_masked[0])[:6]} ref {sorted(ref_masked[0])[:6]}")

print("\nvalue replacement and attention weight, 20px")
v = torch.randn(B, H, CTX, D)
w0 = torch.rand(B, H, CTX, CTX).softmax(-1)
ref_v, ref_w, ref_masked, _, _ = reference_trim(v.clone(), w0.clone(), 20)

cfg = rsa.RSAConfig.for_patch(20)
module = rsa.RSAAttention(_StubAttn(), cfg)
mask = module.select(v)
values = mask[:, None, :, None]
keys = mask[:, None, None, :]
mu = v[:, :, 1:, :].mean(dim=2, keepdim=True)
mine_v = torch.where(values, mu.expand_as(v), v)
mine_w = w0.masked_fill(keys, 1.0 / config.N_TOKENS)

check("masked values equal the per-head mean",
      torch.allclose(mine_v, ref_v, atol=1e-6),
      f"{(mine_v - ref_v).abs().max().item():.3e}")
check("masked attention column equals 1/197",
      torch.allclose(mine_w, ref_w, atol=1e-7),
      f"{(mine_w - ref_w).abs().max().item():.3e}")

print("\nbackward pass against the reference scatter, per patch size")
print("  the forward tensors match under both settings; only the gradient separates them")
for size in (10, 20, 30, 40, 50):
    v0 = torch.randn(B, H, CTX, D)
    gout = torch.randn(B, H, CTX, D)

    # Literal upstream: `v.scatter(-2, idx.repeat(1, H, context_length, D), v_mean)`,
    # `v_mean` being the per-head mean repeated `context_length` times.
    va = v0.clone().requires_grad_(True)
    ref_v, _, _, _, _ = reference_trim(va, torch.rand(B, H, CTX, CTX), size)
    (ref_v * gout).sum().backward()
    g_ref = va.grad.clone()

    grads, fwd = {}, {}
    for mode in ("released", "intended"):
        vb = v0.clone().requires_grad_(True)
        cfg = rsa.RSAConfig.for_patch(size, window_rule="released", backward=mode)
        module = rsa.RSAAttention(_StubAttn(), cfg)
        mask = module.select(vb)
        mu = vb[:, :, 1:, :].mean(dim=2, keepdim=True)
        if mode == "released":
            mu = rsa._ScaleGrad.apply(mu, float(config.N_TOKENS))
        out = torch.where(mask[:, None, :, None], mu.expand_as(vb), vb)
        (out * gout).sum().backward()
        grads[mode] = vb.grad.clone()
        fwd[mode] = out.detach()

    check(f"{size}px forward, released == reference",
          torch.equal(fwd["released"], ref_v.detach()),
          f"{(fwd['released'] - ref_v.detach()).abs().max().item():.3e}")
    check(f"{size}px forward, intended == released",
          torch.equal(fwd["intended"], fwd["released"]))
    # Not bit-exact and cannot be: the reference sums 197 equal float32 copies of the
    # same gradient per masked token, `_ScaleGrad` multiplies by 197 once. The claim is
    # relative agreement at float32 resolution and identical signs, which is what
    # `patch_autopgd` consumes.
    scale = g_ref.abs().max().clamp_min(1e-12)
    rel = ((grads["released"] - g_ref).abs().max() / scale).item()
    signs_agree = bool((grads["released"].sign() == g_ref.sign()).all())
    check(f"{size}px gradient, backward='released' == reference",
          rel < 1e-6 and signs_agree,
          f"relative {rel:.3e}, signs {'all agree' if signs_agree else 'DIFFER'}")

    cos = F.cosine_similarity(g_ref.flatten(), grads["intended"].flatten(), dim=0).item()
    disagree = (g_ref.sign() != grads["intended"].sign()).float().mean().item()
    check(f"{size}px gradient, backward='intended' != reference",
          not torch.equal(grads["intended"], g_ref),
          f"cos {cos:.4f}, {100 * disagree:.1f}% of signs disagree")

print("\nconfig plumbing")
cfg = rsa.RSAConfig.for_patch(20, score_frame="per-head")
check("score_frame round-trips", cfg.as_dict()["score_frame"] == "per-head")
try:
    rsa.RSAConfig.for_patch(20, score_frame="nonsense")
    check("rejects an unknown frame", False)
except ValueError:
    check("rejects an unknown frame", True)
try:
    rsa.RSAConfig.for_patch(20, backward="nonsense")
    check("rejects an unknown backward", False)
except ValueError:
    check("rejects an unknown backward", True)
rel = rsa.RSAConfig.released(20)
check("RSAConfig.released is the released reading on all four knobs",
      (rel.window_rule, rel.renormalise, rel.score_frame, rel.backward)
      == ("released", "uniform", "global", "released"),
      f"{rel.window_rule}/{rel.renormalise}/{rel.score_frame}/{rel.backward}")

print("\nrsa.verify on a stub 4-block model")


class _StubBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _StubAttn()
        self.norm = nn.LayerNorm(H * D)

    def forward(self, t):
        return t + self.attn(self.norm(t))


class _StubViT(nn.Module):
    def __init__(self, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList(_StubBlock() for _ in range(depth))
        self.head = nn.Linear(H * D, 10)

    def forward(self, t):
        for b in self.blocks:
            t = b(t)
        return self.head(t[:, 0])


try:
    out = rsa.verify(_StubViT(), torch.randn(B, CTX, H * D), patch_px=30, verbose=False)
    check("verify passes", True,
          f"identity {out['identity_max_abs_err']:.1e}, "
          f"{out['n_distinct_windows']} distinct windows, "
          f"masked {out['masked_token_counts']}")
    check("end-to-end dx differs between the two backwards",
          out["backward_sign_disagreement"] > 0,
          f"cosine {out['backward_grad_cosine']:.4f}, "
          f"{100 * out['backward_sign_disagreement']:.1f}% of input-gradient signs "
          f"disagree across 4 stub blocks")
except AssertionError as exc:
    check("verify passes", False, str(exc))

print("\ngradient against central finite differences, float64")
print("  independent of both implementations: which backward is the derivative of the forward")


def fd_rel_err(f, x, g, n_dirs=8, eps=1e-6):
    """Worst relative gap between <g, u> and (f(x+eps u) - f(x-eps u)) / 2 eps."""
    gen = torch.Generator().manual_seed(1)
    worst = 0.0
    for _ in range(n_dirs):
        u = torch.randn(x.shape, generator=gen, dtype=x.dtype)
        with torch.no_grad():
            fd = (f(x + eps * u) - f(x - eps * u)) / (2 * eps)
        worst = max(worst, abs((g * u).sum().item() - fd.item()) / max(abs(fd.item()), 1e-12))
    return worst


# Masking op alone, the selected tokens held fixed as the forward holds them (no_grad).
for size in (10, 20, 30, 40, 50):
    v0 = torch.randn(B, H, CTX, D, dtype=torch.float64)
    gout = torch.randn(B, H, CTX, D, dtype=torch.float64)
    attn_w = torch.rand(B, H, CTX, CTX, dtype=torch.float64)
    mask = rsa.RSAAttention(_StubAttn(), rsa.RSAConfig.for_patch(size, window_rule="released")
                            ).select(v0.float())

    def op(v, mode):
        mu = v[:, :, 1:, :].mean(dim=2, keepdim=True)
        if mode == "released":
            mu = rsa._ScaleGrad.apply(mu, float(config.N_TOKENS))
        return (torch.where(mask[:, None, :, None], mu.expand_as(v), v) * gout).sum()

    def literal(v):
        return (reference_trim(v, attn_w, size)[0] * gout).sum()

    def mean_path(v):
        mu = v[:, :, 1:, :].mean(dim=2, keepdim=True)
        return (torch.where(mask[:, None, :, None], mu.expand_as(v), v.detach()) * gout).sum()

    errs, grads = {}, {}
    for name, f in (("intended", lambda v: op(v, "intended")),
                    ("released", lambda v: op(v, "released")),
                    ("literal scatter", literal),
                    ("mean path", mean_path)):
        v = v0.clone().requires_grad_(True)
        (grads[name],) = torch.autograd.grad(f(v), v)
        errs[name] = fd_rel_err(f, v0, grads[name])
    check(f"{size}px op, intended matches finite differences", errs["intended"] < 1e-6,
          f"rel {errs['intended']:.1e}")
    check(f"{size}px op, released and literal scatter do not",
          errs["released"] > 1e-2 and errs["literal scatter"] > 1e-2,
          f"rel {errs['released']:.1e} / {errs['literal scatter']:.1e}")
    # Literal = true gradient + 196 extra copies of the mean-path gradient, i.e. that path x197.
    expected = grads["intended"] + (config.N_TOKENS - 1) * grads["mean path"]
    gap = ((grads["literal scatter"] - expected).abs().max()
           / expected.abs().max()).item()
    check(f"{size}px op, literal scatter == true gradient with mean path x197", gap < 1e-10,
          f"rel {gap:.1e}")

# End to end: input gradient of a 4-block stub ViT with RSA installed in every block.
net = _StubViT().double().eval()
x0 = torch.randn(B, CTX, H * D, dtype=torch.float64)
cfg = rsa.RSAConfig.released(20)
for mode in rsa.BACKWARDS:
    with rsa.enable(net, replace(cfg, backward=mode)) as handle:
        def f(x):
            return net(x)[:, 0].sum()
        f(x0)
        m0 = {i: m.clone() for i, m in handle.masks().items()}
        x = x0.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(f(x), x)
        err = fd_rel_err(f, x0, g)
        f(x0 + 1e-6 * torch.ones_like(x0))
        same_sel = all(torch.equal(m0[i], m) for i, m in handle.masks().items())
    ok = (err < 1e-6) if mode == "intended" else (err > 1e-2)
    check(f"end to end, backward='{mode}' {'matches' if mode == 'intended' else 'misses'} "
          f"finite differences", ok and same_sel,
          f"rel {err:.1e}{'' if same_sel else ', selection moved under eps'}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
