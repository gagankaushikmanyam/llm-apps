"""
Seeding utilities for reproducible experiments.

This module provides a single function `set_seed()` that seeds:
- Python's `random`
- NumPy
- PyTorch (CPU and CUDA if available)

Reproducibility note:
- True determinism can vary by hardware and PyTorch ops.
- This is a strong best-effort approach suitable for demos.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """
    Set seeds for Python, NumPy, and (if installed) PyTorch.

    Args:
        seed: Random seed value.
        deterministic_torch: If True, attempts to make PyTorch deterministic.
            This can reduce performance on GPU but improves reproducibility.

    Raises:
        ValueError: if seed is not a non-negative integer.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch is optional in principle, but required by this repo.
    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # Best-effort determinism settings
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
