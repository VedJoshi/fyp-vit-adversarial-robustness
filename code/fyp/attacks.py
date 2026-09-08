"""Attacks, all operating in natural [0,1] pixel space.

Every function takes a model that accepts [0,1] images (a `models.NormalizedModel`)
and returns an image in [0,1].

Two threat models, not comparable to each other:

    L_inf  : S = { d : ||d||_inf <= eps }        dense support, bounded magnitude
    patch  : S = { d : supp(d) subset P }        sparse support, UNBOUNDED inside P

RSA evaluates with eps = 255/255, unbounded within the patch, which is the default
for the patch attacks here.

Three patch attacks, at different strengths:

    patch_pgd      RSA's *training* attack - 10 steps, fixed step size 0.25, plain SGD
    patch_autopgd  RSA's *evaluation* attack - AutoPGD(DLR), 100 steps, adaptive step
                   size, momentum. This is the one Table 1 was produced with.
    patch_fool     Patch-Fool (Fu et al., ICLR 2022) - picks its own patch by attention
                   and optimises attention onto it alongside the classification loss.
                   The attention-aware adversary measurements 2 and 3 need.

The first two take the patch location as an argument and are searched over a grid by
`worst_location_attack`. Patch-Fool selects its own, on the 16 px token grid, so it
returns the tokens it attacked and is not a drop-in for that search.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from . import config, hooks, metrics


# --------------------------------------------------------------------------
# L_inf
# --------------------------------------------------------------------------
def fgsm(model, x: torch.Tensor, y: torch.Tensor, eps: float = 8 / 255) -> torch.Tensor:
    """One signed gradient step.

    Athalye's warning sign 1: FGSM beating multi-step PGD indicates PGD is stuck,
    not that the model is robust.
    """
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def pgd(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float | None = None,
    steps: int | None = None,
    restarts: int = 1,
) -> torch.Tensor:
    """Projected gradient descent inside the L_inf ball, keeping the best per image."""
    steps = steps or config.SCALE.attack_steps
    alpha = alpha if alpha is not None else max(eps / 4, 1 / 255)

    best = x.clone().detach()
    best_loss = torch.full((x.shape[0],), -math.inf, device=x.device)

    for _ in range(max(restarts, 1)):
        adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(model(adv), y, reduction="sum")
            (grad,) = torch.autograd.grad(loss, adv)
            adv = adv.detach() + alpha * grad.sign()
            adv = torch.min(torch.max(adv, x - eps), x + eps).clamp(0, 1)

        with torch.no_grad():
            per = F.cross_entropy(model(adv), y, reduction="none")
        better = per > best_loss
        best[better] = adv[better]
        best_loss[better] = per[better]

    return best.detach()


# --------------------------------------------------------------------------
# Patch threat model
# --------------------------------------------------------------------------
def patch_mask(
    shape: torch.Size,
    top: int,
    left: int,
    size: int,
    device=None,
) -> torch.Tensor:
    """Binary mask (1, 1, H, W) selecting a square patch. Broadcasts over batch and channels."""
    _, _, H, W = shape
    m = torch.zeros(1, 1, H, W, device=device)
    m[..., top:min(top + size, H), left:min(left + size, W)] = 1.0
    return m


def apply_patch(x: torch.Tensor, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x_adv = (1 - m) * x + m * delta."""
    return ((1 - mask) * x + mask * delta).clamp(0, 1)


def patch_pgd(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    top: int,
    left: int,
    size: int,
    steps: int | None = None,
    step_size: float = 0.25,
    eps: float = 1.0,
    aux_loss: Callable[[torch.Tensor], torch.Tensor] | None = None,
    aux_weight: float = 0.0,
) -> torch.Tensor:
    """PGD restricted to a patch. Defaults follow RSA's PatchPGD.

    RSA's training attack: "10 steps of PGD and a fixed step size of 0.25, using
    basic SGD without momentum", with the perturbation unbounded inside the patch
    (`eps = 255/255`). Their evaluation attack is PatchAutoPGD - AutoPGD(DLR), 100
    steps, adaptive step size - which is not implemented here.

    `aux_loss` takes the adversarial image and returns a scalar maximised alongside
    cross-entropy, for an Attention-Fool-style term. In Patch-Fool's ablation, summing
    an attention term onto CE without per-layer gradient surgery made their attack
    three times worse than omitting it.
    """
    steps = steps or config.SCALE.attack_steps
    mask = patch_mask(x.shape, top, left, size, x.device)

    delta = torch.rand_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        adv = apply_patch(x, delta, mask)
        loss = F.cross_entropy(model(adv), y, reduction="sum")
        if aux_loss is not None and aux_weight:
            loss = loss + aux_weight * aux_loss(adv)
        (grad,) = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + step_size * grad.sign()).clamp(0, 1)
        if eps < 1.0:  # only meaningful if you bound the patch contents
            delta = torch.min(torch.max(delta, x - eps), x + eps).clamp(0, 1)

    return apply_patch(x, delta.detach(), mask)


# --------------------------------------------------------------------------
# PatchAutoPGD - RSA's evaluation attack
# --------------------------------------------------------------------------
def dlr_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Difference-of-Logits Ratio, per example. Croce & Hein (ICML 2020) eq. 4.

        DLR(x, y) = -(z_y - max_{i != y} z_i) / (z_p1 - z_p3)

    with `z_p1 >= z_p2 >= z_p3` the three largest logits. Both shift and rescale
    invariant, so a defense that scales its logits cannot inflate robust accuracy
    against it the way it can against cross-entropy.

    Returned negated, so an attack *maximises* it. Needs at least 3 classes.
    """
    if logits.shape[1] < 3:
        raise ValueError(f"DLR needs at least 3 classes, got {logits.shape[1]}")

    z = logits.sort(dim=1).values                     # ascending
    u = torch.arange(logits.shape[0], device=logits.device)
    y_is_top = (logits.argmax(1) == y).float()
    best_other = z[:, -2] * y_is_top + z[:, -1] * (1.0 - y_is_top)
    return -(logits[u, y] - best_other) / (z[:, -1] - z[:, -3] + 1e-12)


def ce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example cross-entropy, for swapping into `patch_autopgd`."""
    return F.cross_entropy(logits, y, reduction="none")


LOSSES = {"dlr": dlr_loss, "ce": ce_loss}


def _check_oscillation(loss_steps: torch.Tensor, j: int, k: int, rho: float) -> torch.Tensor:
    """APGD condition 1: the step size halves when progress stalls.

    Counts how many of the last `k` steps increased the loss and returns 1.0 where
    that count is at most `rho * k`.

    At the first checkpoint `j - k` is negative, so the last comparison in the window
    indexes the tail of `loss_steps`, which is still the zeros it was allocated with,
    and reduces to `loss_steps[0] > 0`. That is what the fra31 reference does, and it
    is reproduced here rather than corrected: the first checkpoint decides whether the
    step size halves for the remaining 78 steps, and a Table 1 reproduction has to
    take the same branch the published numbers took. Clipping the window to the steps
    that actually ran changes the threshold from `k * rho` to `j * rho` and flips the
    decision whenever the increase count falls between the two.
    """
    inc = sum((loss_steps[j - c] > loss_steps[j - c - 1]).float() for c in range(k))
    return (inc <= k * rho).float()


def patch_autopgd(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    top: int,
    left: int,
    size: int,
    steps: int | None = None,
    eps: float = 1.0,
    loss: str | Callable = "dlr",
    restarts: int = 1,
    step_size_factor: float = 2.0,
    momentum: float = 0.75,
    rho: float = 0.75,
    init: str = "uniform",
    aux_loss: Callable[[torch.Tensor], torch.Tensor] | None = None,
    aux_weight: float = 0.0,
) -> torch.Tensor:
    """AutoPGD restricted to a patch. RSA's evaluation attack.

    RSA reports Table 1 under "PatchAutoPGD", AutoPGD with the DLR loss at 100 steps
    with an adaptive step size and SGD with momentum, unbounded inside the patch
    (`eps = 255/255`). Plain `patch_pgd` under-attacks against that, which would
    inflate robust accuracy and let a Gate A fidelity failure read as a pass.

    Three mechanisms from Croce & Hein (ICML 2020), all per example:

    1. **Momentum.** `x_{k+1} = P(x_k + a(z_{k+1} - x_k) + (1-a)(x_k - x_{k-1}))`,
       with `z_{k+1} = P(x_k + eta * sign(grad))`, `a = 0.75`, and `a = 1` on the
       first step where there is no previous point.
    2. **Adaptive step size**, starting at `step_size_factor * eps` and halving at
       checkpoints when either progress has stalled (`_check_oscillation`) or the
       best loss has not improved since the previous checkpoint and the step size
       was not halved there.
    3. **Restart from the best point** whenever the step size halves.

    Checkpoints follow the reference schedule: the first interval is `0.22 * steps`,
    shrinking by `0.03 * steps` each time down to a floor of `0.06 * steps`.

    The threat model is the patch, not an eps-ball: the perturbation is confined to
    the patch by `patch_mask` and, at the default `eps = 1.0`, bounded only by the
    `[0,1]` pixel range. `eps < 1.0` additionally clips to an eps-ball around `x`.

    `init` selects the starting patch contents. `"uniform"` draws them uniformly from
    `[0,1]`, which is the natural choice when the contents are unbounded and is the
    default. `"boundary"` reproduces AutoPGD's own initialisation: a uniform direction
    divided by its largest absolute coordinate, so exactly one coordinate per image
    lands on the eps-ball boundary and the rest stay inside it. RSA says only that it
    uses standard AutoPGD parameters and does not state which of the two it used, so
    the choice is a parameter and the run record carries it.

    `loss` is a key of `LOSSES` or a callable `(logits, y) -> per-example tensor`.
    `aux_loss` takes the adversarial image and returns a per-example tensor added to
    the objective, for an RSA-aware stealth term. In Patch-Fool's ablation, summing an
    attention term onto the classification loss without per-layer gradient surgery made
    their attack three times worse than omitting it.

    Returns, per example, the point that misclassified if the attack found one, and
    otherwise the highest-loss point seen. The reference returns its initialisation
    for examples it never flips; robust accuracy is identical either way, since both
    are classified correctly, but the highest-loss point is the stronger candidate for
    `worst_location_attack`, which ranks locations by loss.

    The unmodified image is itself inside the threat model - the patch may hold the
    original pixels - so an image the model already misclassifies is not robust no
    matter what this returns. That case belongs to the caller: score with
    `robust_correct`, not with `accuracy`.
    """
    steps = steps or config.SCALE.attack_steps
    loss_fn = LOSSES[loss] if isinstance(loss, str) else loss
    mask = patch_mask(x.shape, top, left, size, x.device)

    n_iter_2 = max(int(0.22 * steps), 1)
    n_iter_min = max(int(0.06 * steps), 1)
    size_decr = max(int(0.03 * steps), 1)

    B = x.shape[0]
    bcast = (-1, 1, 1, 1)

    def project(z: torch.Tensor) -> torch.Tensor:
        if eps < 1.0:
            z = torch.min(torch.max(z, x - eps), x + eps)
        return apply_patch(x, z, mask)

    def evaluate(adv: torch.Tensor):
        adv = adv.clone().detach().requires_grad_(True)
        logits = model(adv)
        per = loss_fn(logits, y)
        if aux_loss is not None and aux_weight:
            per = per + aux_weight * aux_loss(adv)
        (grad,) = torch.autograd.grad(per.sum(), adv)
        return per.detach(), grad.detach(), logits.detach()

    any_flip = torch.zeros(B, dtype=torch.bool, device=x.device)
    best_flip = x.clone()
    best_loss_pt = x.clone()
    best_loss_overall = torch.full((B,), -math.inf, device=x.device)

    for _ in range(max(restarts, 1)):
        if init == "uniform":
            adv = project(torch.rand_like(x))
        elif init == "boundary":
            t = 2 * torch.rand_like(x) - 1
            t = t / t.flatten(1).abs().amax(1).view(*bcast).clamp_min(1e-12)
            adv = project(x + eps * t)
        else:
            raise ValueError(f"unknown init {init!r}; pick 'uniform' or 'boundary'")

        step = torch.full((B, 1, 1, 1), step_size_factor * eps, device=x.device)
        per, grad, logits = evaluate(adv)

        x_best, grad_best, loss_best = adv.clone(), grad.clone(), per.clone()
        flipped = logits.argmax(1) != y
        x_flip = adv.clone()

        adv_old = adv.clone()
        loss_steps = torch.zeros(steps, B, device=x.device)
        k = n_iter_2
        counter = 0
        # Ones, not zeros: the reference initialises `reduced_last_check` to true so
        # that `1 - reduced_last` is zero at the first checkpoint and the
        # no-improvement condition cannot fire there. Only oscillation can halve the
        # first step size.
        reduced_last = torch.ones(B, device=x.device)
        loss_best_last = loss_best.clone()

        for i in range(steps):
            a = momentum if i > 0 else 1.0
            z = project(adv + step * grad.sign())
            nxt = project(adv + a * (z - adv) + (1 - a) * (adv - adv_old))
            adv_old, adv = adv, nxt

            per, grad, logits = evaluate(adv)
            loss_steps[i] = per

            now_wrong = logits.argmax(1) != y
            x_flip[now_wrong] = adv[now_wrong]
            flipped |= now_wrong

            better = per > loss_best
            x_best[better] = adv[better]
            grad_best[better] = grad[better]
            loss_best[better] = per[better]

            counter += 1
            if counter == k:
                halve = torch.max(
                    _check_oscillation(loss_steps, i, k, rho),
                    (1.0 - reduced_last) * (loss_best_last >= loss_best).float(),
                )
                reduced_last = halve.clone()
                loss_best_last = loss_best.clone()

                sel = halve > 0
                if sel.any():
                    step[sel] = step[sel] / 2.0
                    # `adv_old` is deliberately left alone, so the momentum term on the
                    # next step is taken against the pre-restart point. The reference
                    # does the same.
                    adv = torch.where(sel.view(*bcast), x_best, adv)
                    grad = torch.where(sel.view(*bcast), grad_best, grad)

                counter = 0
                k = max(k - size_decr, n_iter_min)

        newly = flipped & ~any_flip
        best_flip[newly] = x_flip[newly]
        any_flip |= flipped

        improved = loss_best > best_loss_overall
        best_loss_pt[improved] = x_best[improved]
        best_loss_overall[improved] = loss_best[improved]

    return torch.where(any_flip.view(*bcast), best_flip, best_loss_pt).detach()


# --------------------------------------------------------------------------
# Patch-Fool - the attention-aware attack
# --------------------------------------------------------------------------
def _pcgrad(a: torch.Tensor, c: torch.Tensor, normalise: bool = False) -> torch.Tensor:
    """Project the attention gradient off the cross-entropy gradient where they conflict.

    Both arguments are flattened per example, `(B, D)`. Rows whose cosine similarity is
    negative are the ones pulling against each other; those get `a` replaced by its
    component orthogonal to `c`, and every other row is returned untouched.

    `normalise` selects the divisor, and the two are not the same operation:

        False  a - (<a,c> / ||c||) c      the `GATECH-EIC/Patch-Fool` reference
        True   a - (<a,c> / ||c||^2) c    the projection from Yu et al. (NeurIPS 2020)

    Only the second is a projection. The first leaves `<a', c> = <a,c>(1 - ||c||)`,
    which is zero only when the cross-entropy gradient happens to be a unit vector, so
    it over-corrects below that norm and under-corrects above it. The reference form is
    the default because it is what produced Patch-Fool's published numbers; the run
    record carries the choice.
    """
    sim = F.cosine_similarity(a, c, dim=1)
    conflict = sim < 0
    if not bool(conflict.any()):
        return a

    a_c, c_c = a[conflict], c[conflict]
    denom = c_c.norm(dim=-1).clamp_min(1e-12)
    scale = (a_c * c_c).sum(dim=-1) / (denom.pow(2) if normalise else denom)

    out = a.clone()
    out[conflict] = a_c - scale.view(-1, 1) * c_c
    return out


def token_patch_mask(
    tokens: torch.Tensor,
    image_size: int = config.IMAGE_SIZE,
    patch_size: int = config.PATCH_SIZE,
) -> torch.Tensor:
    """Image-token indices `(B, k)` -> per-image binary mask `(B, 1, H, W)`.

    Indices are into the 196 image tokens, CLS excluded, which is what
    `select_patches_by_attention` returns. Unlike `patch_mask` the region is different
    for every image in the batch, so the mask carries a batch dimension and cannot be
    broadcast from a single square.
    """
    grid = image_size // patch_size
    flat = torch.zeros(tokens.shape[0], grid * grid, device=tokens.device)
    flat.scatter_(1, tokens.long(), 1.0)
    m = flat.view(-1, 1, grid, grid)
    return m.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)


def select_patches_by_attention(
    model,
    x: torch.Tensor,
    layer: int = 4,
    num_patch: int = 1,
) -> torch.Tensor:
    """Rank image tokens by the attention they *receive*, and take the top `num_patch`.

    Averages the attention matrix over heads and then over query positions, so the
    score for token j is the mean weight every token places on j at this layer. CLS is
    dropped from the candidates but kept as a query, matching the reference.

    Returns `(B, num_patch)` indices into the 196 image tokens, CLS excluded.
    """
    with hooks.AttentionCapture(model, layers=[layer], store=("weights",)) as cap:
        with torch.no_grad():
            model(x)
        received = cap[layer].weights.mean(dim=1).mean(dim=-2)[:, 1:]

    return received.argsort(dim=1, descending=True)[:, :num_patch]


def patch_fool(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 250,
    num_patch: int = 1,
    select_layer: int = 4,
    lr: float = 0.22,
    atten_loss_weight: float = 0.002,
    lr_step: int = 10,
    gamma: float = 0.95,
    patches: torch.Tensor | None = None,
    pcgrad: str = "reference",
    attn_layers: Sequence[int] | None = None,
    keep_best: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch-Fool (Fu et al., ICLR 2022), checked against `GATECH-EIC/Patch-Fool`.

    The attack every other patch attack here is measured against. `patch_pgd` and
    `patch_autopgd` optimise pixels against the classification loss and are indifferent
    to what attention does; Patch-Fool adds a term that *drags attention onto the patch
    it is perturbing*, which is what makes it the right adversary for measurements 2
    and 3. Those two measure how much of the deviation RSA charges for actually reduces
    the margin, and an attack that ignores attention cannot answer that.

    Three parts, all from the reference:

    1. **The patch is chosen, not given.** `select_patches_by_attention` takes the token
       receiving the most attention at `select_layer`. The patch is therefore one
       16x16 token cell, grid-aligned - a different threat model from RSA's
       arbitrarily-placed 10-50 px squares, and not comparable to a Table 1 number.
    2. **Two objectives.** Cross-entropy, plus `-log a[i, p]` averaged over every query
       row `i` at each layer in `attn_layers`, which is maximised when all attention
       lands on the chosen patch `p`.
    3. **Per-layer gradient surgery.** Each layer's attention gradient is taken
       separately, projected off the cross-entropy gradient where the two conflict
       (`_pcgrad`), and only then accumulated at `atten_loss_weight`. Summing the terms
       into one scalar instead - which is what `aux_loss` on `patch_pgd` does - made
       their attack three times worse in their own ablation.

    Optimisation follows the reference: Adam at `lr = 0.22` on the patch contents,
    `StepLR(step_size=10, gamma=0.95)`, 250 iterations, ascending by way of
    `delta.grad = -grad`. `attn_layers` defaults to layers `1 .. L//2 - 1`, which is
    1-5 for a 12-block model - the reference skips layer 0 and stops at half depth.

    Two deviations, both deliberate:

    - **The patch replaces pixels; it does not add to them.** The reference computes
      `X + delta * mask` and clamps only `delta` to the valid pixel range, so the
      composite can leave it and the resulting image is not renderable. Here the patch
      content *is* `delta`, clamped to `[0,1]` by `apply_patch`, which is the threat
      model the rest of this module uses and RSA's `eps = 255/255`. The reachable set
      inside the patch is unchanged; only the parameterisation is.
    - **`keep_best` returns the first misclassifying iterate** rather than the last one,
      free of charge because the logits are already computed each step. The reference
      returns the final iterate. Set it False to reproduce that exactly.

    Returns `(adv, tokens)` - the adversarial images, and the `(B, num_patch)` token
    indices that were attacked, which measurements 2 and 3 need to know where to look.
    `patches` overrides the selection with indices of that shape.

    Cost is the reason this is not the default attack: one forward and
    `1 + len(attn_layers)` backward passes per iteration, six on a 12-block model, over
    250 iterations.
    """
    if pcgrad not in ("reference", "normalised"):
        raise ValueError(f"pcgrad must be 'reference' or 'normalised', got {pcgrad!r}")

    B = x.shape[0]
    tokens = (select_patches_by_attention(model, x, layer=select_layer, num_patch=num_patch)
              if patches is None else patches.to(x.device).long())
    mask = token_patch_mask(tokens, image_size=x.shape[-1])

    if attn_layers is None:
        n_blocks = len(hooks._blocks(model))
        attn_layers = [i for i in range(n_blocks // 2) if i != 0]
    attn_layers = list(attn_layers)

    # The reference targets the single highest-attention patch even when it perturbs
    # several, and indexes the attention matrix, so CLS is added back on.
    target = tokens[:, 0] + 1

    delta = torch.rand_like(x).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=gamma)

    best = x.clone()
    any_flip = torch.zeros(B, dtype=torch.bool, device=x.device)

    for _ in range(steps):
        adv = apply_patch(x, delta, mask)

        with hooks.AttentionCapture(
            model, layers=attn_layers, store=("weights",), detach=False
        ) as cap:
            logits = model(adv)

        ce = F.cross_entropy(logits, y, reduction="sum")
        (grad,) = torch.autograd.grad(ce, delta, retain_graph=True)
        ce_flat = grad.view(B, -1).detach().clone()

        for layer in attn_layers:
            w = cap[layer].weights.mean(dim=1)                    # (B, N, N), head mean
            nll = -torch.log(w.clamp_min(1e-12))
            idx = target.view(B, 1, 1).expand(B, w.shape[1], 1)
            attn_loss = nll.gather(2, idx).mean()

            (a_grad,) = torch.autograd.grad(attn_loss, delta, retain_graph=True)
            projected = _pcgrad(a_grad.view(B, -1), ce_flat, normalise=(pcgrad == "normalised"))
            grad = grad + atten_loss_weight * projected.view_as(grad)

        if keep_best:
            with torch.no_grad():
                newly = (logits.argmax(1) != y) & ~any_flip
                best[newly] = adv[newly].detach()
                any_flip |= newly

        opt.zero_grad()
        delta.grad = -grad          # Adam descends; the objective is being maximised.
        opt.step()
        sched.step()
        with torch.no_grad():
            delta.clamp_(0, 1)

    final = apply_patch(x, delta.detach(), mask)
    if keep_best:
        final = torch.where(any_flip.view(-1, 1, 1, 1), best, final)
    return final.detach(), tokens

def patch_locations(size: int, stride: int = 20, image_size: int = config.IMAGE_SIZE) -> list[tuple[int, int]]:
    """The evaluation grid of patch positions, as RSA uses it.

    RSA: "evaluated at evenly-spaced patch locations along a grid with stride 20 ...
    10px and 20px patches are evaluated at 121 locations, 30px and 40px at 100,
    50px at 81."

    Reverse-engineered from their counts and matching all five: positions are
    `range(0, image_size - size + 1, stride)` with no extra position at the far edge.
    That gives 11 per side for 10/20 px, 10 for 30/40 px and 9 for 50 px, i.e.
    121 / 100 / 81 locations. Appending a flush-right position gives 121 for 30 px,
    which does not match.

    RSA's adversarial training samples one location uniformly at random per image,
    while evaluation searches this grid for the worst one.
    """
    coords = list(range(0, image_size - size + 1, stride))
    return [(t, l) for t in coords for l in coords]


def worst_location_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    size: int,
    locations: Sequence[tuple[int, int]] | None = None,
    steps: int | None = None,
    attack: Callable = patch_pgd,
    **kw,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Attack at every candidate location, keep the worst case per image.

    An image counts as robust only if it survives the attack at every location, so a
    location that misclassifies ranks above one that does not and the loss orders
    locations only within those two groups. Ranking on loss alone can return a
    correctly-classified point for an image that some other location did flip, which
    reports it as robust.

    The unmodified image is a feasible point of the threat model - the patch is free to
    hold the original pixels - so an image that is already misclassified starts out
    flipped, at `x`, and no location can make it robust. Without that seeding the first
    location overwrites the clean point and an image the model never classified
    correctly is reported robust.

    `attack` is the per-location attack: `patch_pgd` by default, `patch_autopgd` for
    an evaluation faithful to RSA. Cost scales as `n_locations x steps`.
    """
    locations = locations or patch_locations(size)
    best = x.clone()
    best_loss = torch.full((x.shape[0],), -math.inf, device=x.device)
    best_flip = ~accuracy(model, x, y)
    best_loc: list[tuple[int, int]] = [(0, 0)] * x.shape[0]

    for (top, left) in locations:
        adv = attack(model, x, y, top, left, size, steps=steps, **kw)
        with torch.no_grad():
            logits = model(adv)
            per = F.cross_entropy(logits, y, reduction="none")
            flip = logits.argmax(1) != y

        better = (flip & ~best_flip) | ((flip == best_flip) & (per > best_loss))
        best[better] = adv[better]
        best_loss[better] = per[better]
        best_flip |= flip
        for i in better.nonzero(as_tuple=True)[0].tolist():
            best_loc[i] = (top, left)

    return best.detach(), best_loc


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
@torch.no_grad()
def accuracy(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example correctness as a bool tensor. RA and ASR both derive from this."""
    return (model(x).argmax(1) == y)


@torch.no_grad()
def robust_correct(model, adv: torch.Tensor, y: torch.Tensor, clean_correct: torch.Tensor) -> torch.Tensor:
    """Per-example robustness: correct under attack *and* correct before it.

    An image the model already misclassifies is never robust. The unmodified image is
    inside every threat model used here - an eps-ball contains its centre, and a patch
    may hold the pixels that were already there - so the identity perturbation is a
    valid adversarial example and an attack that returned it would succeed. Scoring
    with `accuracy` alone counts such an image robust whenever the attack happens to
    hand back a point the model gets right, which inflates robust accuracy by up to the
    clean error rate.

    `clean_correct` is `accuracy(model, x, y)` on the same examples, in the same order.
    """
    return accuracy(model, adv, y) & clean_correct


def report(clean_correct: torch.Tensor, adv_correct: torch.Tensor) -> dict[str, float]:
    """Robust accuracy and attack success rate - note the different denominators.

        RA  = correct under attack AND correct clean / ALL examples
        ASR = successfully flipped / examples that were INITIALLY CORRECT

    The denominators differ, so the two are not interchangeable.

    `adv_correct` is conditioned on `clean_correct` here rather than trusted as passed,
    so a caller handing in a bare `accuracy(model, adv, y)` still gets the right robust
    accuracy. See `robust_correct` for why a clean error is never robust.
    """
    robust = adv_correct & clean_correct
    n = clean_correct.numel()
    n_correct = int(clean_correct.sum())
    flipped = int((clean_correct & ~robust).sum())
    return {
        "clean_acc": 100.0 * n_correct / n,
        "robust_acc": 100.0 * int(robust.sum()) / n,
        "asr": 100.0 * flipped / max(n_correct, 1),
        "n": n,
    }
