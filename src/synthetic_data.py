"""Deterministic synthetic grayscale image generators."""
from __future__ import annotations

import numpy as np


def gradient_image(size: int = 64) -> np.ndarray:
    """Return a horizontal gradient from 0 to 1, shape (size, size)."""
    row = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.tile(row[None, :], (size, 1))


def blurred_noise_image(
    size: int = 64,
    blur_scale: float = 1.0,
    seed: int = 0,
    base: float = 0.5,
    amplitude: float = 0.1,
) -> np.ndarray:
    """Return periodic Gaussian-smoothed noise around the base value.

    The FFT Gaussian filter uses periodic boundaries. There is no per-image
    min-max normalization.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size)).astype(np.float32)
    if blur_scale > 0:
        fy = np.fft.fftfreq(size)[:, None]
        fx = np.fft.fftfreq(size)[None, :]
        response = np.exp(-2.0 * np.pi**2 * blur_scale**2 * (fx**2 + fy**2))
        noise = np.fft.ifft2(np.fft.fft2(noise) * response).real.astype(np.float32)
    sd = float(noise.std())
    if sd > 0:
        noise = noise / sd
    return (base + amplitude * noise).astype(np.float32)


def stripe_image(
    size: int = 64,
    angle_degrees: float = 30.0,
    period: float = 8.0,
) -> np.ndarray:
    """Return diagonal sinusoidal stripes with values in [0, 1]."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    theta = np.deg2rad(angle_degrees)
    phase = xx * np.cos(theta) + yy * np.sin(theta)
    return (0.5 + 0.5 * np.sin(2.0 * np.pi * phase / period)).astype(np.float32)


def make_image(
    kind: str, *, size: int = 64, seed: int = 0, blur_scale: float = 1.0
) -> np.ndarray:
    """Dispatch to one of the supported synthetic image families."""
    if kind == "gradient":
        return gradient_image(size)
    if kind == "blurred_noise":
        return blurred_noise_image(size=size, blur_scale=blur_scale, seed=seed)
    if kind == "stripes":
        return stripe_image(size=size)
    raise ValueError(f"Unknown image kind: {kind!r}")
