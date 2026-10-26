"""Small, explicit image reconstruction metrics."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean squared error over True entries of mask."""
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if not np.any(mask):
        raise ValueError("mask selects no pixels")
    return float(np.mean((prediction[mask] - target[mask]) ** 2))


def psnr(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR in dB for images with a known data range."""
    mse = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def ssim(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Structural similarity for 2D grayscale images."""
    return float(structural_similarity(target, prediction, data_range=data_range))
