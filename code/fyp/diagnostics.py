"""Athalye's obfuscated-gradient warning signs, accumulated over a batch sample.

Athalye, Carlini and Wagner (ICML 2018), "Obfuscated Gradients Give a False Sense of
Security", list five behaviours that indicate the attack is failing rather than the
model defending:

    1. A one-step attack beats an iterative attack.
    2. A black-box attack beats a white-box attack.
    3. An unbounded attack does not reach 100% success.
    4. Random sampling beats gradient descent.
    5. Increasing the distortion bound does not increase attack success.

On an undefended model all of them hold, so a failure locates a defect in the attack
harness. Sign 2 requires a surrogate model to transfer from and is not implemented, so
`all_pass` covers four signs out of five and `untested` names the one it omits.

Signs 1 and 5 are comparisons, and equality satisfies a comparison. An attack that is
stuck returns the same robust accuracy at every budget and at every step count, which
is exactly the failure these signs exist to catch, so equality passes only where the
attack has saturated at 0% and there is nothing left to improve on.

Every number here is scored with `attacks.robust_correct`: an image the model already
misclassifies counts as not robust, because the unmodified image is a feasible point of
both threat models. Scoring with `attacks.accuracy` instead inflates robust accuracy by
up to the clean error rate, and inflates the random-search error of sign 4 by the whole
of it, since a clean error is misclassified under every draw.

Per-example correctness is accumulated across batches, so the sample size is set by
the batches passed in rather than by what fits on the GPU at once.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch

from . import attacks, config

DEFAULT_SWEEP = (1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255)

LABELS = {
    "1_one_step_vs_iterative": "iterative beats one-step",
    "2_blackbox_vs_whitebox": "white-box beats black-box",
    "3_unbounded_attack": "unbounded attack reaches 0% robust",
    "4_random_vs_gradient": "gradient beats random search",
    "5_monotone_in_budget": "robust accuracy falls as budget rises",
}


def _pct(flags: torch.Tensor) -> float:
    """Percentage of True entries in a bool tensor."""
    return 100.0 * flags.float().mean().item()


def _beats(strong_acc: float, weak_acc: float) -> bool:
    """Sign 1: the stronger attack must leave strictly less robust accuracy.

    Equality passes only once the attack has saturated at 0%, where there is nothing
    left to improve on. Anywhere else, two attacks of very different strength landing
    on the same number is the symptom the sign exists to catch: an attack that is stuck
    rather than a model that is robust.
    """
    if strong_acc > weak_acc:
        return False
    return strong_acc < weak_acc or strong_acc == 0.0


def _falls_with_budget(ladder: list[float]) -> bool:
    """Sign 5: robust accuracy must be non-increasing in budget, and must actually fall.

    Athalye et al. require that increasing the distortion bound increases attack
    success, "except once success has saturated". A ladder that is flat across every
    budget satisfies non-increasing monotonicity and still fails the sign, so the end
    of the ladder must be below its start unless it has reached 0%.
    """
    if not all(a >= b for a, b in zip(ladder, ladder[1:])):
        return False
    return ladder[-1] < ladder[0] or ladder[-1] == 0.0


def warning_signs(
    model,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    eps: float = 8 / 255,
    steps: int | None = None,
    n_random: int = 32,
    eps_sweep: Sequence[float] = DEFAULT_SWEEP,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Run signs 1, 3, 4 and 5 over `batches` and return measurements and verdicts.

    `batches` yields `(x, y)` in [0,1] pixel space, so a DataLoader can be passed
    directly. `steps` defaults to `config.SCALE.attack_steps`.

    `eps` is added to `eps_sweep` if absent and the sweep entry at `eps` supplies
    signs 1 and 4, so each budget is attacked once.

    Signs 1 and 3 and 5 are stated in robust accuracy, correct under attack and correct
    clean, over all examples. Sign 4 compares two error rates over the same denominator
    and the same conditioning: the fraction of all examples that were classified
    correctly and then misclassified under at least one of `n_random` uniform draws from
    the eps-ball, against the fraction classified correctly and then misclassified under
    PGD, which is `clean_acc - pgd_acc`.

    The returned dict has one entry per sign under `signs`, each carrying its
    measured quantities and a `pass` flag, `None` for the unimplemented sign 2, plus
    a top-level `all_pass` over the testable signs.
    """
    steps = steps or config.SCALE.attack_steps
    device = device or config.get_device()
    budgets = sorted(set(eps_sweep) | {eps})

    clean, fgsm, unbounded, random_miss = [], [], [], []
    sweep: dict[float, list[torch.Tensor]] = {e: [] for e in budgets}

    for i, (x, y) in enumerate(batches, 1):
        x, y = x.to(device), y.to(device)

        c = attacks.accuracy(model, x, y)
        clean.append(c)
        fgsm.append(attacks.robust_correct(model, attacks.fgsm(model, x, y, eps), y, c))
        unbounded.append(
            attacks.robust_correct(model, attacks.pgd(model, x, y, eps=1.0, steps=steps), y, c)
        )

        miss = torch.zeros_like(y, dtype=torch.bool)
        for _ in range(n_random):
            noise = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
            miss |= ~attacks.accuracy(model, noise, y)
        random_miss.append(miss & c)

        for e in budgets:
            sweep[e].append(
                attacks.robust_correct(model, attacks.pgd(model, x, y, eps=e, steps=steps), y, c)
            )

        if verbose:
            done = sum(t.numel() for t in clean)
            print(f"  batch {i}: {done} images done")

    clean_c = torch.cat(clean)
    clean_acc = _pct(clean_c)
    fgsm_acc = _pct(torch.cat(fgsm))
    unbounded_acc = _pct(torch.cat(unbounded))
    random_err = _pct(torch.cat(random_miss))
    sweep_acc = {e: _pct(torch.cat(v)) for e, v in sweep.items()}

    pgd_acc = sweep_acc[eps]
    ladder = [sweep_acc[e] for e in budgets]

    signs = {
        "1_one_step_vs_iterative": {
            "fgsm_robust_acc": fgsm_acc,
            "pgd_robust_acc": pgd_acc,
            "pass": _beats(pgd_acc, fgsm_acc),
        },
        "2_blackbox_vs_whitebox": {
            "pass": None,
            "note": "not implemented - needs a surrogate model and a defense to transfer to",
        },
        "3_unbounded_attack": {
            "eps": 1.0,
            "robust_acc": unbounded_acc,
            "pass": unbounded_acc == 0.0,
        },
        "4_random_vs_gradient": {
            "random_search_error": random_err,
            "pgd_error": clean_acc - pgd_acc,
            "n_random_draws": n_random,
            "pass": (clean_acc - pgd_acc) >= random_err,
        },
        "5_monotone_in_budget": {
            "robust_acc_by_eps": {f"{e * 255:.0f}/255": round(sweep_acc[e], 2) for e in budgets},
            "pass": _falls_with_budget(ladder),
        },
    }

    return {
        "n_images": int(clean_c.numel()),
        "attack_steps": steps,
        "eps": eps,
        "clean_acc": clean_acc,
        "signs": signs,
        "all_pass": all(s["pass"] for s in signs.values() if s["pass"] is not None),
        "untested": [k for k, s in signs.items() if s["pass"] is None],
    }


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k} {x}" for k, x in v.items()) + "}"
    return str(v)


def print_report(res: dict) -> None:
    """Print the sample size, then one line per sign with its measurements and verdict."""
    print(
        f"{res['n_images']} images, {res['attack_steps']}-step PGD, "
        f"eps = {res['eps'] * 255:.0f}/255, clean accuracy {res['clean_acc']:.2f}%"
    )
    for key, sign in res["signs"].items():
        verdict = {True: "PASS", False: "FAIL", None: "n/a "}[sign["pass"]]
        detail = ", ".join(
            f"{k} {_fmt(v)}" for k, v in sign.items() if k not in ("pass", "note")
        )
        print(f"  {verdict}  {LABELS[key]:<36s} {detail}")
        if sign.get("note"):
            print(f"        {sign['note']}")
    n_untested = len(res.get("untested", []))
    verdict = "all testable signs pass" if res["all_pass"] else "a sign FAILED"
    print(f"{verdict} ({n_untested} sign not tested)" if n_untested else verdict)


# --------------------------------------------------------------------------
# The same five signs in the patch threat model
# --------------------------------------------------------------------------
PATCH_SWEEP = (10, 20, 30, 40, 50)

PATCH_LABELS = {
    "1_one_step_vs_iterative": "iterative beats one-step",
    "2_blackbox_vs_whitebox": "white-box beats black-box",
    "3_unbounded_attack": "whole-image patch reaches 0% robust",
    "4_random_vs_gradient": "gradient beats random patch contents",
    "5_monotone_in_budget": "robust accuracy falls as patch grows",
}


def centred(size: int, image_size: int = config.IMAGE_SIZE) -> tuple[int, int]:
    """Top-left corner placing a `size`-pixel square at the centre of the image."""
    off = (image_size - size) // 2
    return off, off


def patch_warning_signs(
    model,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    size: int = 10,
    steps: int | None = None,
    attack=attacks.patch_autopgd,
    n_random: int = 32,
    size_sweep: Sequence[int] = PATCH_SWEEP,
    device: torch.device | None = None,
    verbose: bool = True,
    **attack_kw,
) -> dict:
    """Athalye's signs restated for the patch threat model, driven by `attack`.

    The threat model differs from `warning_signs` in what the budget is. A patch
    attack is already unbounded in magnitude (`eps = 255/255`) and bounded in support,
    so the budget that varies is the **patch size**, and the unbounded limit of sign 3
    is a patch covering the whole image, which leaves the attacker a free choice of
    input.

    The four testable signs:

    1. `steps = 1` of `attack` leaves more robust accuracy than `steps` does, or both
       have reached 0%.
    3. A `image_size`-pixel patch leaves 0% robust accuracy.
    4. `n_random` draws of uniform patch contents misclassify no more often than
       `attack` does, over the same denominator and the same conditioning.
    5. Robust accuracy is non-increasing in patch size over `size_sweep`, and lower at
       the largest size than at the smallest unless it has reached 0%.

    Signs 1 and 4 use `size`, which is folded into `size_sweep` so each size is
    attacked once. Every patch is centred (`centred`); the location grid belongs to
    the evaluation in `attacks.worst_location_attack`, not to this check.

    `attack` takes `(model, x, y, top, left, size, steps=...)` and returns an image,
    so `attacks.patch_pgd` and `attacks.patch_autopgd` both fit.
    """
    steps = steps or config.SCALE.attack_steps
    device = device or config.get_device()
    sizes = sorted(set(size_sweep) | {size})
    full = config.IMAGE_SIZE

    clean, one_step, unbounded, random_miss = [], [], [], []
    sweep: dict[int, list[torch.Tensor]] = {s: [] for s in sizes}

    def run(x, y, s, n_steps, clean_correct):
        top, left = centred(s)
        adv = attack(model, x, y, top=top, left=left, size=s, steps=n_steps, **attack_kw)
        return attacks.robust_correct(model, adv, y, clean_correct)

    for i, (x, y) in enumerate(batches, 1):
        x, y = x.to(device), y.to(device)

        c = attacks.accuracy(model, x, y)
        clean.append(c)
        one_step.append(run(x, y, size, 1, c))
        unbounded.append(run(x, y, full, steps, c))

        top, left = centred(size)
        m = attacks.patch_mask(x.shape, top, left, size, x.device)
        miss = torch.zeros_like(y, dtype=torch.bool)
        for _ in range(n_random):
            draw = attacks.apply_patch(x, torch.rand_like(x), m)
            miss |= ~attacks.accuracy(model, draw, y)
        random_miss.append(miss & c)

        for s in sizes:
            sweep[s].append(run(x, y, s, steps, c))

        if verbose:
            done = sum(t.numel() for t in clean)
            print(f"  batch {i}: {done} images done")

    clean_c = torch.cat(clean)
    clean_acc = _pct(clean_c)
    one_step_acc = _pct(torch.cat(one_step))
    unbounded_acc = _pct(torch.cat(unbounded))
    random_err = _pct(torch.cat(random_miss))
    sweep_acc = {s: _pct(torch.cat(v)) for s, v in sweep.items()}

    attack_acc = sweep_acc[size]
    ladder = [sweep_acc[s] for s in sizes]

    signs = {
        "1_one_step_vs_iterative": {
            "one_step_robust_acc": one_step_acc,
            "iterative_robust_acc": attack_acc,
            "pass": _beats(attack_acc, one_step_acc),
        },
        "2_blackbox_vs_whitebox": {
            "pass": None,
            "note": "not implemented - needs a surrogate model and a defense to transfer to",
        },
        "3_unbounded_attack": {
            "patch_px": full,
            "robust_acc": unbounded_acc,
            "pass": unbounded_acc == 0.0,
        },
        "4_random_vs_gradient": {
            "random_search_error": random_err,
            "attack_error": clean_acc - attack_acc,
            "n_random_draws": n_random,
            "pass": (clean_acc - attack_acc) >= random_err,
        },
        "5_monotone_in_budget": {
            "robust_acc_by_size": {f"{s}px": round(sweep_acc[s], 2) for s in sizes},
            "pass": _falls_with_budget(ladder),
        },
    }

    return {
        "n_images": int(clean_c.numel()),
        "attack": getattr(attack, "__name__", str(attack)),
        "attack_steps": steps,
        "patch_px": size,
        "clean_acc": clean_acc,
        "signs": signs,
        "all_pass": all(s["pass"] for s in signs.values() if s["pass"] is not None),
        "untested": [k for k, s in signs.items() if s["pass"] is None],
    }


def print_patch_report(res: dict) -> None:
    """One line per sign, with the attack and patch size the signs were run at."""
    print(
        f"{res['n_images']} images, {res['attack']} at {res['attack_steps']} steps, "
        f"{res['patch_px']}px centred patch, clean accuracy {res['clean_acc']:.2f}%"
    )
    for key, sign in res["signs"].items():
        verdict = {True: "PASS", False: "FAIL", None: "n/a "}[sign["pass"]]
        detail = ", ".join(
            f"{k} {_fmt(v)}" for k, v in sign.items() if k not in ("pass", "note")
        )
        print(f"  {verdict}  {PATCH_LABELS[key]:<40s} {detail}")
        if sign.get("note"):
            print(f"        {sign['note']}")
    n_untested = len(res.get("untested", []))
    verdict = "all testable signs pass" if res["all_pass"] else "a sign FAILED"
    print(f"{verdict} ({n_untested} sign not tested)" if n_untested else verdict)
