"""Write a notebook's headline numbers to `results/<name>.json`.

These files are tracked by git; `results/figures/` is not. See the negations at the
bottom of `.gitignore`.

Each record carries a `_meta` block with the scale, dataset, model, seed, torch version,
device and git commit the numbers were produced under, and an `is_result` flag that is
true only at `config.FULL`.

Usage::

    from fyp import results
    results.save("m1_baseline", {"clean_acc": 99.8, ...})
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


def header() -> dict:
    """The `_meta` block stamped onto every record."""
    import torch

    return {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "scale": config.SCALE.name,
        "dataset": config.SCALE.dataset,
        "model": config.MODEL_NAME,
        "n_eval_images": config.SCALE.n_eval_images,
        "attack_steps": config.SCALE.attack_steps,
        "seed": config.SEED,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "is_result": config.SCALE.name == "full",
    }


def save(name: str, payload: dict) -> Path:
    """Write `results/<name>.json`. Returns the path."""
    record = {"_meta": header(), **_jsonify(payload)}
    path = config.RESULTS_ROOT / f"{name}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    tag = "RESULT" if record["_meta"]["is_result"] else "DEV - not a result"
    print(f"wrote {path.relative_to(config.VAULT_ROOT)}  [{tag}]")
    return path


def load(name: str) -> dict:
    """Read `results/<name>.json` back."""
    return json.loads((config.RESULTS_ROOT / f"{name}.json").read_text(encoding="utf-8"))
