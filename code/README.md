# Code — ViT adversarial patch robustness

Working code for Direction 1. The science is in `../02b_direction1_from_the_ground_up.md`;
this is how to run it.

## Quick start

```bash
pip install torch torchvision timm matplotlib jupyterlab
cd code
jupyter lab notebooks/
```

Run `M1_baseline.ipynb` first. **`config.SCALE` is `FULL`**, which needs ImageNet-100
on disk and runs in minutes; set it to `DEV` in `fyp/config.py` for the one-minute
version, which downloads Imagenette (98 MB) instead.

## The three notebooks

| Notebook | What it does | Runtime (DEV, laptop GPU) |
|---|---|---|
| `M1_baseline.ipynb` | DeiT-S, RSA-style head slicing, clean accuracy, token geometry | ~12 s |
| `M2_attacks.ipynb` | FGSM, PGD, patch-PGD, worst-case-over-locations, **Athalye's warning signs** | ~46 s |
| `M3_instrumentation.ipynb` | Hook verification + **the three measurements** | ~19 s |

At `FULL` the same notebooks are minutes rather than seconds: M2 runs Athalye's warning
signs over 256 images at a 100-step budget, which is six PGD runs per batch and about
18 minutes on a 6 GB laptop GPU.

## The package

| Module | What it holds |
|---|---|
| `fyp/config.py` | paths, model, the `DEV`/`FULL` switch, seeding |
| `fyp/data.py` | ImageNet-100 and Imagenette, RSA's class-slicing contract |
| `fyp/models.py` | timm loading, RSA head slicing, the `[0,1]` wrapper, `fused_attn` off, `attention_scaling` |
| `fyp/hooks.py` | `AttentionCapture` and the three-way capture verification |
| `fyp/metrics.py` | entropy, dominance, logit gap and span, softmax underflow, RSA's anomaly score |
| `fyp/attacks.py` | FGSM, PGD, patch-PGD, PatchAutoPGD, Patch-Fool, RSA's location grid, worst-case-over-locations, `robust_correct` |
| `fyp/rsa.py` | the RSA defense: per-layer masking, the renormalisation flag, the model wiring |
| `fyp/diagnostics.py` | Athalye's warning signs, in both the `L_inf` and the patch threat model |
| `fyp/results.py` | run records to tracked `results/*.json` |
| `scripts/build_imagenet100.py` | builds the `FULL` val split from an archive |
| `scripts/logit_gap_by_model.py` | Measurement 1 across architectures; `--batch` is the sample size |
| `scripts/warning_signs.py` | regenerates the two `FULL` warning-sign records |
| `scripts/validate_patch_fool.py` | Patch-Fool against PatchAutoPGD at a matched location |
| `scripts/mechanism_full.py` | measurements 2 and 3 at `FULL`, driven by Patch-Fool |
| `scripts/aas_efficacy.py` | whether pre-softmax attention scaling helps on this checkpoint |
| `scripts/gate_a.py` | the Table 1 reproduction sweep, resumable |

## Scale

`fyp/config.py` has one switch:

```python
SCALE: Scale = DEV     # Imagenette, 10 classes, 64 images  -> proves the code is correct
SCALE: Scale = FULL    # ImageNet-100, 512 images, RSA's protocol -> produces results
```

**It is set to `FULL`.** Every runtime in the table above is the `DEV` figure.

**DEV numbers are not results.** Imagenette is ten well-separated classes; DeiT-S gets
~100% on it. DEV exists so you can find bugs in a minute instead of an hour.

For `FULL` you need ImageNet-100 on disk. Because RSA's head-slicing recipe involves no
training, **Act 1 needs only the validation split** — about 5000 images, ~650 MB, not the
127k-image training set. Point `$FYP_IMAGENET100` at a directory of wnid folders
(`<root>/val/n01440764/*.JPEG`). The training split is only needed for Act 2's patch
adversarial training, which wants a cluster anyway.

## Four things that bite

**1. `fused_attn` must be off.** timm ≥ 1.0 uses `F.scaled_dot_product_attention`, which
never materialises the attention matrix — the weights and pre-softmax logits do not exist
as tensors, and every hook silently captures nothing. `models.load_model` disables it.

**2. Captured tensors are on a parallel autograd branch.** A hook on `qkv` gets that
Linear's output; the q/k/v we build from it are *our* reshape, while timm reshapes the
same output separately and the loss flows through timm's copy. So `rec.v.grad` is always
`None`. Use `rec.grad("v")`, which slices the retained gradient of the qkv output.

**3. Normalisation lives in the model, not the transform.** Attacks perturb `[0,1]` pixel
space; `NormalizedModel` applies mean/std internally. This is what makes `eps = 8/255`
mean the same thing on every channel.

**4. Score robustness with `attacks.robust_correct`, not `attacks.accuracy`.** An image
the model already misclassifies is not robust: the unmodified image is inside both threat
models — an eps-ball contains its centre, a patch may hold the pixels already there — so
the identity perturbation is a valid adversarial example. `accuracy(model, adv, y)` alone
counts such an image robust whenever the attack hands back a point the model happens to
get right, which inflates robust accuracy by up to the clean error rate (7.03 points here)
and inflates a random-search error rate by the whole of it. `worst_location_attack`,
`gate_a.py` and both `diagnostics` functions all seed from the clean prediction.

## Verify before trusting

```python
from fyp import models, hooks
clf, info = models.load_model()
hooks.verify_all_layers(clf, torch.rand(2, 3, 224, 224).cuda())
```

Checks the reconstruction three ways — recomputed softmax against timm's own tensor,
attention rows summing to 1, and rebuilding the block output from captured weights and
values. The softmax match and the output reconstruction come out at `0.000e+00`; row sums
carry `3.58e-07`, float32 epsilon against the `1e-4` tolerance. `verify_all_layers` returns
the per-layer errors and the worst of each across layers. **Re-run after any timm upgrade.**

Athalye's warning signs are the same kind of check applied to the attack rather than the
hooks, and they run on any model including a defended one:

```python
from itertools import islice
from fyp import diagnostics
res = diagnostics.warning_signs(clf, islice(loader, 8))
diagnostics.print_report(res)
```

On an undefended model all four testable signs pass. A failure locates a defect in the
attack harness. Sign 2 needs a surrogate model and is not implemented, so `all_pass`
covers four signs and `untested` names the fifth.

**Signs 1 and 5 reject a plateau.** They are comparisons, and equality satisfies a
comparison, so an attack that is simply stuck — same robust accuracy at every budget,
same at one step as at a hundred — used to pass both. Equality now passes only where the
attack has saturated at 0% and there is nothing left to improve on. That is Athalye's own
carve-out and it is what lets the check fail against a defense rather than only against a
broken harness.

`warning_signs` is the `L_inf` threat model. `patch_warning_signs` is the same five signs
restated for the patch threat model, driven by whichever patch attack is passed in:

```python
res = diagnostics.patch_warning_signs(clf, islice(loader, 8), attack=attacks.patch_autopgd)
diagnostics.print_patch_report(res)
```

The budget that varies is the **patch size**, not `eps` — a patch attack is already unbounded
in magnitude — so sign 5 sweeps sizes and sign 3's unbounded limit is a patch covering the whole
image, which leaves the attacker a free choice of input.

## Environment notes

Verified on Python 3.13, torch 2.7.0+cu118, timm 1.0.15, RTX 4050 Laptop (6 GB).

- **torchvision is a CPU-only build** (`0.22.0+cpu`) alongside a CUDA torch. Harmless
  here — only `datasets` and `transforms` are used, which are pure Python — but
  `torchvision.io` GPU decode would fail.
- **6 GB VRAM is fine for Act 1** and not enough for Act 2. RSA trains at batch 512;
  attention capture alone costs ~11 MB per image per quantity per 12 layers. Use
  `AttentionCapture(..., to_cpu=True)` when sweeping many images, and plan on cluster
  compute for patch adversarial training.
- `num_workers=0` in the loaders on purpose — worker processes interact badly with
  notebooks on Windows. Raise it on Linux.

## Results are tracked

Each notebook ends with a `fyp.results.save(...)` cell writing `results/<name>.json`, and those
files are committed. Figures and datasets stay ignored. Every record carries a `_meta` block with
the scale, dataset, model, seed, torch version, device and git commit it was produced under, and
an `is_result` flag that is true only at `FULL`.

```python
from fyp import results
results.load("m2_attacks")["_meta"]["is_result"]
```

**Check that flag before quoting a number.** DEV runs on Imagenette and its numbers are not
comparable to RSA's.

## The RSA defense

`fyp/rsa.py` is RSA (Mu & Wagner, ICML 2021 UDL Workshop) as a drop-in replacement for every
block's attention. `enable` swaps the modules in place and returns a handle that reverts them,
so the model object is unchanged from the outside and `attacks.*`, `diagnostics.*` and
`metrics.*` run against the defended model exactly as they run against the undefended one.

```python
from fyp import attacks, rsa

cfg = rsa.RSAConfig.for_patch(30)            # window 3x3, renormalise="uniform", frame="global"
with rsa.enable(clf, cfg) as handle:
    adv = attacks.patch_autopgd(clf, x, y, top=0, left=0, size=30)
    acc = attacks.accuracy(clf, adv, y)
    handle.windows()[6]                      # (B, 2) window layer 6 masked on the last pass
```

The scoring half is in `metrics` — `rsa_token_scores`, `rsa_window_scores`, `rsa_window_size`,
`rsa_argmax_window`. `rsa.py` adds the masking, the per-layer replacement and the wiring.

**`renormalise` is a config flag because the paper is silent.** RSA sets `α_{·,i} ← 1/N` for
masked columns and never says what happens to the rest of the row. The three modes:

| `renormalise` | What it does | Row sums | max \|Δ\| vs Table 2 |
|---|---|---|---|
| `"uniform"` (default) | masked keys' post-softmax weights set to `1/197` — the paper's literal rule, and what the authors' code does | neither | 1.18 |
| `"zero"` | masked keys' post-softmax weights set to 0 | less than 1 | **0.78** |
| `"softmax"` | masked keys' pre-softmax logits set to `-inf` | exactly 1 | 5.67 |

**Settled 7 Sep by the authors' code, not by clean accuracy.**
`pycls/models/vision_transformer.py:107` of `wagner-group/robust-self-attention` scatters
`1 / context_length` into the masked column, `context_length = 14² + 1 = 197`, and does not
renormalise. That is `"uniform"`, and the constant is `config.N_TOKENS`, not `N_IMAGE_TOKENS`.

Until then the default was `"zero"`, chosen on the Table 2 fit below — 0.78 points against
`"uniform"`'s 1.18. **That was the wrong basis for the choice.** Masking nine tokens out of 196
costs about the same on a clean image whether the right nine or the wrong nine are masked, so
clean accuracy has almost no power to discriminate here, and a 0.4-point edge is not evidence
against the mode the authors ran. Measured on 512 ImageNet-100 images against their Table 2
(93.16 / 92.58 / 90.43 / 90.43 / 88.67 / 83.98):

```
zero      92.97  91.80  91.02  91.02  87.89  84.57     <- within 0.78 everywhere
uniform   92.97  91.41  90.82  90.82  88.28  85.16
softmax   92.97  92.38  91.02  91.02  90.43  89.65     <- curve far too flat
RSA Tbl2  93.16  92.58  90.43  90.43  88.67  83.98
sizes        0px   10px   20px   30px   40px   50px
```

Renormalising the surviving row to sum to 1 removes most of the clean-accuracy cost RSA
report — 89.65% at 50px against their 83.98%. The lost attention mass **is** the cost.
`"uniform"` stays close because it differs from `"zero"` only by a `(|w*|/N)·μ` term.
Record: `results/m5_rsa_sanity.json`.

Gate A ran on 3-4 Sep and the mode does not decide Table 1: on one batch at 20px the three modes
land within single digits of each other at both locations tested. The window rule does decide it.
See `## Gate A` below and `../gate-a-diagnosis.md`.

`RSAConfig.for_patch(0)` gives `window = 0`, which disables masking; the wrapper then reproduces
the undefended logits to `0.000e+00`. That is Table 2's `0px` column, and it is the check that
the re-implemented attention is arithmetically identical to timm's.

**Four things that are deliberate.** Each layer scores its own value tensor and masks its own
argmax window, so DeiT-S makes twelve independent decisions. `[CLS]` is excluded from the mean
and from the scores and is never masked. Masked tokens are removed as keys and as values but
remain queries. The argmax runs under `no_grad` and is **not** smoothed — rung L4 of the attack
ladder (BPDA) is what has to get through it.

**Verify before trusting it**, the same way the hooks are verified:

```python
rsa.verify(clf, torch.rand(4, 3, 224, 224).cuda())
```

Five checks — window 0 reproduces the undefended logits at `0.000e+00`, every layer masks
exactly `window²` tokens, CLS is masked at no layer, the twelve layers do not all pick the same
window, and a gradient reaches the input. It raises naming the property that failed.

The gradient check matters for Gate A specifically: the argmax is non-differentiable, but the
network around it is not, and the attacker gets the gradient with the current mask held fixed.
Measured on the patch, that gradient is non-zero at **100% of patch pixels at the same magnitude
as outside the patch**, so a Gate A number is a measurement of the defense rather than of
gradient masking.

## PatchAutoPGD against the reference

`attacks.patch_autopgd` is AutoPGD (Croce & Hein, ICML 2020) restricted to a patch, written
against `fra31/auto-attack`. Gate A reproduces a table the reference implementation produced,
so where the reference has a quirk this follows the quirk rather than improving on it. Two
places where that decision is live, both fixed on 29 Aug after a review:

- **The first checkpoint's oscillation window.** The reference indexes `loss_steps[j - k]`
  with `j - k` negative, which wraps to the zero-filled tail of the buffer and reduces the
  last comparison of the window to `loss_steps[0] > 0`. Clipping the window to the steps that
  actually ran looks like a fix, and it changes the threshold from `k * rho` to `j * rho`,
  which flips the halving decision whenever the increase count lands between the two. The
  first checkpoint sets the step size for the remaining 78 steps, so the reference behaviour
  is reproduced.
- **`reduced_last` starts at ones.** The reference initialises its `reduced_last_check` to
  true so the no-improvement condition cannot fire at the first checkpoint; only oscillation
  can halve the first step size. Starting at zeros halves too early.

Both were measured, and the reference-faithful version is also the stronger attack: on one
32-image batch with a 32px patch at (96,96), 20-step DLR went from 18.75% robust accuracy to
**12.50%** against `patch_pgd`'s 31.25%, and 100 steps reaches 0.00% either way. Record:
`results/m4_patch_autopgd_validation.json`.

`init` is the one parameter RSA does not pin down. `"uniform"` draws patch pixels uniformly
from `[0,1]`, which is the natural choice when the contents are unbounded, and is the default.
`"boundary"` is AutoPGD's own: a uniform direction divided by its largest absolute coordinate,
so one coordinate per image lands on the eps-ball boundary and the rest stay inside. Both were
run; at 100 steps they tie at 0.00%, at 20 steps boundary is slightly stronger (9.38%). The
record and `gate_a.py --init` both carry the choice.

## Patch-Fool against the reference

`attacks.patch_fool` is Patch-Fool (Fu et al., ICLR 2022), written against
`GATECH-EIC/Patch-Fool`. It is the attention-aware adversary, and it exists because
measurements 2 and 3 ask how much of the deviation RSA charges for actually does harm -
a question an attack that ignores attention cannot answer.

Three parts, all from the reference. **The patch is chosen, not given**: the token
receiving the most attention at layer 4, averaged over heads and over query rows, which
makes the patch one grid-aligned 16 px cell rather than an arbitrarily-placed square.
**Two objectives**: cross-entropy plus `-log a[i, p]` averaged over every query row, which
is maximised when all attention lands on the chosen patch. **Per-layer gradient surgery**:
each layer's attention gradient is projected off the cross-entropy gradient where the two
conflict and only then accumulated at weight 0.002. Summing the terms into one scalar
instead made their attack three times worse in their own ablation - which is what
`aux_loss` on `patch_pgd` does, so do not reach for that as a substitute.

Adam at `lr = 0.22` over 250 iterations with `StepLR(10, 0.95)`, ascending by way of
`delta.grad = -grad`. Attention layers default to `1 .. L//2 - 1`, which is 1-5 on a
12-block model: the reference skips layer 0 and stops at half depth.

Two deviations from the reference, both deliberate:

- **The patch replaces pixels; it does not add to them.** The reference computes
  `X + delta * mask` and clamps only `delta` to the valid pixel range, so the composite
  can leave `[0,1]` and the image is not renderable. Here the patch content *is* `delta`,
  clamped by `apply_patch`. The reachable set inside the patch is unchanged; only the
  parameterisation is.
- **`keep_best` returns the first misclassifying iterate**, free of charge because the
  logits are already computed each step. The reference returns the last one. Set it False
  to reproduce that exactly.

One thing in the reference is a bug rather than a quirk, and `_pcgrad` carries both forms.
PCGrad projects `a` off `c` as `a - (<a,c>/||c||^2) c`; the reference divides by `||c||`
once, leaving `<a', c> = <a,c>(1 - ||c||)`, which is zero only when the cross-entropy
gradient happens to be a unit vector. `pcgrad="reference"` is the default because it is
what produced their published numbers, and `pcgrad="normalised"` is the actual projection.

**Validated 29 Aug**, 32 images at `FULL`, 16 px patch, robust accuracy conditioned on the
clean prediction:

| Attack | Patch | Robust acc |
|---|---|---:|
| PatchAutoPGD, 100 steps | token 90, fixed | 9.38% |
| Patch-Fool, 250 steps | token 90, fixed | **3.12%** |
| Patch-Fool, 250 steps | chosen by attention | **0.00%** |

The middle row isolates the objective, since the patch is the same square as the row above
it, and the bottom row adds the selection. Both earn their place, and a 16x16 patch - 0.5%
of the image - takes the undefended model to zero. It costs about 5x PatchAutoPGD per
image: one forward and `1 + len(attn_layers)` backward passes per iteration, six on a
12-block model, over 250 iterations rather than 100. Record:
`results/m4_patch_fool_validation.json`.

## Pre-softmax attention scaling, and why rung L2 came off

`models.attention_scaling(model, lam)` multiplies every block's pre-softmax attention
logits by `lam`, which is Jain & Dutta's Eq. 2, `softmax(lam·QKᵀ/√d)·V`. timm computes
`q = q * self.scale` before the matmul, so scaling `attn.scale` scales the logits and
nothing else. It raises if `fused_attn` is still on, because
`F.scaled_dot_product_attention` is called without a `scale` argument and computes its
own `1/√E` — `attn.scale` is ignored on that path, so the context manager would silently
do nothing. That is the same silent failure `disable_fused_attention` exists to prevent.

**AAS is *Adaptive* Attention Scaling**, not attention-*aware*: their Algorithm 1 learns
one factor per block by ten steps of gradient ascent on an LPIPS feature distance between
the normal and the scaled model, clamped to `[1e-7, 1]` so only scaling down is allowed,
and then attacks the scaled model. `scripts/aas_efficacy.py` implements their **`+Scale`
baseline** — one factor shared across blocks, best of their own set — which is how they
describe that row themselves, and which by their Table 3 carries 1.6 of AAS's 2.1 points.
Full Algorithm 1 is not implemented, and the record says so.

**The result, 256 images at `FULL`, 16 px patch, PatchAutoPGD(DLR) at 100 steps.**
The undefended model is at 91.80% clean. Scaling costs it up to **86.33 points of clean
accuracy**, where Jain & Dutta report at most about one:

| Scaling | Clean acc of the scaled model | Robust acc on the untouched model |
|---|---:|---:|
| `λ = 0.0001` | 5.47% | 91.41% |
| `λ = 0.001` | 5.47% | 91.41% |
| `λ = 0.01` | 5.47% | 91.41% |
| `λ = 0.1` | 15.23% | 91.80% |
| `λ = 1` | 91.80% | 15.62% |
| `λ = 10` | 20.70% | 91.80% |

The unscaled attack leaves **15.62%**. Every scaled variant is *worse*, and the reason is
in the clean column rather than the robust one: **on this checkpoint the scaling destroys
the model.** That is measurement 1 seen from the other side. Their ViT-B/16 carries
pre-softmax gaps of 250–1000, so dividing by 100 still leaves gaps of 2.5–10 and a
functioning, peaked attention. DeiT-S here peaks at 11.19, so dividing by 100 leaves about
0.11 — attention goes essentially uniform and the network stops discriminating. `λ = 10`
breaks it in the other direction, pushing gaps to roughly 110 and past the ~103 underflow
threshold. The checkpoint sits in a narrow well-conditioned band and both directions leave
it.

A 16 px patch is used rather than 32 px because PatchAutoPGD already reaches 0.00% at
32 px, and against a floor no scaling can show an improvement — the flat-ladder reading
the 29 Aug review flagged for the eps sweep. `unscaled` and `λ = 1` are the same
computation and are both run; they agree at `0.00e+00`, which is what shows the context
manager restores state.

## Gate A

`scripts/gate_a.py` drives `attacks.patch_autopgd` over `attacks.patch_locations` against the
defended model and compares against the `ViT-small + RSA` row of RSA's Table 1. An image counts
robust only if it survives every location; the worst-case state is seeded from the clean
predictions, so an image the defense already misclassifies starts out not robust and the reported
number can never exceed clean accuracy.

**The measured figures are withheld from this repository** until the model and class-subset
control closes (see the root README). The sweep, its settings and the control are all here; only
the outputs are held back.

**The three under-specified rules.** RSA's text does not fix the window side, the frame the
anomaly score is centred in, or whether attention rows are renormalised after masking. Each
reading is implemented behind a flag — `--window-rule`, `--score-frame`, `--renorm` — and the
defaults are the values the authors' released implementation uses, verified against their code by
`scripts/check_reference_fidelity.py`. Every sweep pins all three explicitly rather than relying
on a default, so a result file does not change meaning when a default does.

**No third party has implemented RSA's window rule.** Liu et al. (ICML 2023) mask single tokens
the way the `topk` reading does, and were treated here as the one published reimplementation. That
attribution was withdrawn on 8 Sep: Attention-Mask is a separate defense, aimed at a
pixel-budgeted attacker whose perturbation scatters across tokens, so a contiguous window was
never its target. `wagner-group/robust-self-attention` is therefore the sole authority on what RSA
does.

```bash
python scripts/gate_a.py --sizes 20 --n-images 64 --loc-stride 4 --tag pilot20  # ~22 min, laptop
python scripts/gate_a.py --sizes 20 --n-images 64 --tag pilot20full             # ~85 min, laptop
sbatch scripts/slurm/gate_a_reference.sbatch                                    # five sizes, cluster
python scripts/merge_shards.py --tag reference --job <id>                       # merge the shards
```

`--loc-stride` keeps every n-th location, which weakens the worst-case search and raises the
reported robust accuracy, so a pilot **below** a target is decisive and a pilot at or above it is
not. Progress is appended per location to `results/gate_a_<tag>.progress.jsonl` and replayed on
restart, so an interrupted sweep resumes and an interruption costs one location, not the run.

**Laptop budget.** Measured with RSA active: 20.7 s per 100-step attack batch of 32, 1.13x the
undefended 18.3 s, peak memory 2.92 GB, so batch 32 is the ceiling on a 6 GB card. The full
five-size sweep is 48.3 hours there (11.2 h at 10px and 20px, 9.2 h at 30px and 40px, 7.5 h at
50px); as five parallel array tasks it finishes in under a day of wall clock.
Record: `results/m5_rsa_sanity.json`.
