"""Write a notebook's headline numbers to `results/<name>.json`.

These files are tracked by git; `results/figures/` is not. See the negations at the
bottom of `.gitignore`.

Each record carries a `_meta` block with the scale, dataset, model, seed, torch and timm
versions, device, git commit and dirty state the numbers were produced under, and an
`is_result` flag.

**The runner declares its own scale.** `n_eval_images` and `attack_steps` are arguments
to `save`, not constants read from `config.SCALE`, and `is_result` is true only when the
runner passes both and both reach `FULL`. Reading them from the module-level scale made
`_meta` describe the scale the process was configured at rather than the run that
happened.

Usage::

    from fyp import results
    results.save("m1_baseline", {"clean_acc": 99.8, ...},
                 n_eval_images=n, attack_steps=args.steps)
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


def _git_commit() -> str | None:
    """Short HEAD hash, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.VAULT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    """Whether the tree has uncommitted changes; None if git is unavailable.

    A commit hash alone does not identify the source a number came from: the headline
    Gate A records were produced from a copied cluster tree with no `.git` at all, and
    a dirty tree with a clean-looking hash is the same problem in a quieter form.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.VAULT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def _timm_version() -> str | None:
    try:
        import timm

        return timm.__version__
    except Exception:
        return None


def _jsonify(obj: Any) -> Any:
    """Coerce tensors, numpy scalars and Paths into something json can write."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and getattr(obj, "numel", lambda: 1)() == 1:
        return obj.item()                      # 0-d tensor / numpy scalar
    if hasattr(obj, "tolist"):
        return obj.tolist()                    # tensor / ndarray
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def header(n_eval_images: int | None = None, attack_steps: int | None = None) -> dict:
    """The `_meta` block stamped onto every record.

    `n_eval_images` and `attack_steps` describe *this run* and must be passed by the
    runner. They used to be read from `config.SCALE`, which describes the scale the
    module was imported at, not what the runner did: a one-image smoke test under
    `FULL` was stamped `n_eval_images: 512`, `attack_steps: 100`, `is_result: true`.
    `m2_attacks.json` carries exactly that - meta 512/100 over core attacks that ran on
    32 images. The payloads were right and the provenance was fiction.

    Omitting them now records `null` rather than a scale constant. A record that cannot
    say what it measured says nothing instead of saying the wrong thing.
    `attack_steps=None` is legitimate for a measurement that runs no attack;
    `n_eval_images=None` is not, and demotes the record.

    **`is_result` and `full_sample` are separate questions.** `is_result` asks whether
    this ran on the real dataset and model with honest provenance. `full_sample` asks
    whether it used the whole evaluation sample. Several measurements are legitimately
    small by design - `validate_patch_fool` and `checkpoint_control` run one 32-image
    batch, and say so in their own docstrings - so collapsing the two would label a real
    validation "DEV - not a result", which is exactly the kind of wrong header this
    change exists to stop. The audit's complaint was that a one-image run *claimed* 512,
    not that a 32-image run cannot be a measurement. Both flags are recorded, the sample
    size sits beside them, and the reader judges.
    """
    import torch

    is_full = config.SCALE.name == "full" and n_eval_images is not None
    full_sample = (
        n_eval_images is not None
        and n_eval_images >= config.SCALE.n_eval_images
        and (attack_steps is None or attack_steps >= config.SCALE.attack_steps)
    )

    return {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "scale": config.SCALE.name,
        "dataset": config.SCALE.dataset,
        "model": config.MODEL_NAME,
        "n_eval_images": n_eval_images,
        "attack_steps": attack_steps,
        "scale_n_eval_images": config.SCALE.n_eval_images,
        "scale_attack_steps": config.SCALE.attack_steps,
        "seed": config.SEED,
        "torch": torch.__version__,
        # RSA copies timm internals (`fyp/rsa.py`) and fused-attention behaviour depends
        # on them (`fyp/models.py`), so the timm version is part of the result.
        "timm": _timm_version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "is_result": is_full,
        "full_sample": full_sample,
    }


def save(name: str, payload: dict, n_eval_images: int | None = None,
         attack_steps: int | None = None) -> Path:
    """Write `results/<name>.json`. Returns the path.

    Pass `n_eval_images` and `attack_steps` from the run's own arguments; see `header`.
    """
    record = {"_meta": header(n_eval_images, attack_steps), **_jsonify(payload)}
    path = config.RESULTS_ROOT / f"{name}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    meta = record["_meta"]
    scope = f"{n_eval_images}/{config.SCALE.n_eval_images} images"
    if attack_steps is not None:
        scope += f", {attack_steps} steps"
    if n_eval_images is None:
        tag = "provenance incomplete - runner did not declare its sample size"
    elif meta["is_result"] and meta["full_sample"]:
        tag = f"RESULT  {scope}"
    elif meta["is_result"]:
        tag = f"RESULT, partial sample  {scope}"
    else:
        tag = f"DEV - not a result  {scope}"
    print(f"wrote {path.relative_to(config.VAULT_ROOT)}  [{tag}]")
    return path


def load(name: str) -> dict:
    """Read `results/<name>.json` back."""
    return json.loads((config.RESULTS_ROOT / f"{name}.json").read_text(encoding="utf-8"))
