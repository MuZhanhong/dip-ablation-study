"""Simple image inpainting baselines."""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def grey_fill(image: np.ndarray, gap_mask: np.ndarray, fill_value: float = 0.5) -> np.ndarray:
    """Replace missing pixels with a constant grey value."""
    if image.shape != gap_mask.shape:
        raise ValueError("image and gap_mask must have identical shapes")
    result = image.copy()
    result[gap_mask] = fill_value
    return result


def harmonic_fill(image: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
    """Fill the gap by solving the four-neighbour discrete Laplace equation."""
    if image.shape != gap_mask.shape or image.ndim != 2:
        raise ValueError("image and 2D gap_mask must have identical shapes")
    coords = np.argwhere(gap_mask)
    if len(coords) == 0:
        return image.copy()
    h, w = image.shape
    index = -np.ones((h, w), dtype=int)
    for i, (y, x) in enumerate(coords):
        index[y, x] = i
    matrix = lil_matrix((len(coords), len(coords)), dtype=np.float64)
    rhs = np.zeros(len(coords), dtype=np.float64)
    for i, (y, x) in enumerate(coords):
        matrix[i, i] = 4.0
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w:
                j = index[ny, nx]
                if j >= 0:
                    matrix[i, j] = -1.0
                else:
                    rhs[i] += float(image[ny, nx])
    result = image.copy()
    result[gap_mask] = spsolve(matrix.tocsr(), rhs).astype(image.dtype, copy=False)
    return result
