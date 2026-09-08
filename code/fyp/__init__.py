"""FYP: adversarial patch robustness in Vision Transformers.

    from fyp import config, data, models, hooks, metrics, attacks, rsa, diagnostics, results

`config.describe()` prints the current scale. `results.save(...)` writes a run record.
"""
from . import attacks, config, data, diagnostics, hooks, metrics, models, results, rsa  # noqa: F401

__all__ = ["config", "data", "models", "hooks", "metrics", "attacks", "rsa", "diagnostics", "results"]
