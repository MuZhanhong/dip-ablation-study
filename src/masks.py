"""Masks for image inpainting and held-out validation."""
from __future__ import annotations

import numpy as np


def central_gap_mask(size: int = 64, gap_size: int = 16) -> np.ndarray:
    """Boolean mask: True indicates a missing pixel in the centered square."""
    if not (0 < gap_size <= size):
        raise ValueError("gap_size must be between 1 and size")
    mask = np.zeros((size, size), dtype=bool)
    start = (size - gap_size) // 2
    mask[start : start + gap_size, start : start + gap_size] = True
    return mask


def heldout_mask(
    gap_mask: np.ndarray, fraction: float = 0.05, seed: int = 0
) -> np.ndarray:
    """Sample held-out pixels from visible pixels only; True means held out."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if gap_mask.ndim != 2:
        raise ValueError("gap_mask must be a 2D array")
    visible_indices = np.flatnonzero(~gap_mask)
    count = int(round(fraction * visible_indices.size))
    rng = np.random.default_rng(seed)
    selected = (
        rng.choice(visible_indices, size=count, replace=False)
        if count else np.array([], dtype=int)
    )
    result = np.zeros(gap_mask.shape, dtype=bool)
    result.flat[selected] = True
    return result
