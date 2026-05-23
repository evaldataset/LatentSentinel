"""Determinism + seed setup for all Latent Sentinel CLI scripts.

CHECK fix M-9: every GPU script in code/analysis_emnlp/ should import this
module at the top of main() (or earlier) to ensure consistent random state
and deterministic CUDA kernel selection across machines.

Usage:
    from _setup import set_determinism
    set_determinism(seed=42)
"""
from __future__ import annotations
import os
import random
import numpy as np


def set_determinism(seed: int = 42, *, strict: bool = True) -> None:
    """Seed all RNGs and enable deterministic CUDA algorithms.

    Sets:
      - Python random, numpy, torch CPU + CUDA seeds.
      - PYTHONHASHSEED env var (for set/dict ordering reproducibility).
      - CUBLAS_WORKSPACE_CONFIG (required by torch.use_deterministic_algorithms).
      - cudnn benchmark=False, deterministic=True.
      - torch.use_deterministic_algorithms(strict).

    `strict=False` permits some non-deterministic ops to fall back (e.g.
    bf16 matmul); `strict=True` (default) raises if any op is non-det.
    Set strict=False for inference/scoring; True for cache-producing runs.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(strict, warn_only=not strict)
        except Exception:
            # older torch; deterministic-only opt-in
            torch.use_deterministic_algorithms(strict)
    except ImportError:
        pass  # CPU-only env without torch is fine for pure-numpy scripts
