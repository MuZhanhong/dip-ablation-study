"""Synthetic images for the inpainting experiments."""
from __future__ import annotations

import numpy as np


def gradient_image(size: int = 64) -> np.ndarray:
    """Horizontal gradient, from 0 to 1."""
    row = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.tile(row[None, :], (size, 1))


def blurred_noise_image(
    size: int = 64,
    blur_scale: float = 1.0,
    seed: int = 0,
    base: float = 0.5,
    amplitude: float = 0.1,
) -> np.ndarray:
    """Blur noise with a periodic FFT filter, then scale its standard deviation."""
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
    """Diagonal sine-wave stripes."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    theta = np.deg2rad(angle_degrees)
    phase = xx * np.cos(theta) + yy * np.sin(theta)
    return (0.5 + 0.5 * np.sin(2.0 * np.pi * phase / period)).astype(np.float32)


def change_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale every pixel's distance from 0.5 by factor, then clip to [0, 1].

    factor = 1 leaves the image unchanged; factor = 0.5 halves the contrast.
    """
    changed = 0.5 + factor * (np.asarray(image, dtype=np.float32) - 0.5)
    return np.clip(changed, 0.0, 1.0).astype(np.float32)


def make_image(
    kind: str,
    *,
    size: int = 64,
    seed: int = 0,
    blur_scale: float = 1.0,
    stripe_period: float = 8.0,
) -> np.ndarray:
    """Create an image by name."""
    if kind == "gradient":
        return gradient_image(size)
    if kind == "blurred_noise":
        return blurred_noise_image(size=size, blur_scale=blur_scale, seed=seed)
    if kind == "stripes":
        return stripe_image(size=size, period=stripe_period)
    raise ValueError(f"Unknown image kind: {kind!r}")
