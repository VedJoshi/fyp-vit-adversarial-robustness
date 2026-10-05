"""Save result JSON with run settings and software provenance."""
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
    """Whether tracked or untracked changes exist; None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.VAULT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
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
    """Build metadata using the sample size and steps declared by the runner.

    is_result identifies a FULL-scale run with a declared sample; full_sample
    checks whether its sample and attack steps meet the configured protocol.
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
        "timm": _timm_version(),
        "device": (torch.cuda.get_device_name(0)
                   if config.get_device().type == "cuda" else "cpu"),
        "is_result": is_full,
        "full_sample": full_sample,
    }


def save(name: str, payload: dict, n_eval_images: int | None = None,
         attack_steps: int | None = None) -> Path:
    """Write results/<name>.json with metadata and return its path."""
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
