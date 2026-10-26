import numpy as np

from baselines.fill import grey_fill, harmonic_fill
from src.masks import central_gap_mask, heldout_mask
from src.synthetic_data import blurred_noise_image, gradient_image, stripe_image


def test_gradient_shape_and_range():
    image = gradient_image(64)
    assert image.shape == (64, 64)
    assert np.isclose(image.min(), 0.0)
    assert np.isclose(image.max(), 1.0)
    assert np.allclose(image[0], image[-1])


def test_blurred_noise_is_reproducible():
    a = blurred_noise_image(size=64, blur_scale=2, seed=7)
    b = blurred_noise_image(size=64, blur_scale=2, seed=7)
    assert np.allclose(a, b)
    assert a.shape == (64, 64)
    assert abs(float(a.mean()) - 0.5) < 0.03


def test_stripes_shape():
    assert stripe_image(64).shape == (64, 64)


def test_masks_do_not_overlap():
    gap = central_gap_mask(64, 16)
    heldout = heldout_mask(gap, fraction=0.05, seed=0)
    assert gap.sum() == 256
    assert not np.any(gap & heldout)
    assert heldout.sum() == round(0.05 * (64 * 64 - 256))


def test_fill_baselines():
    image = gradient_image(16)
    gap = central_gap_mask(16, 4)
    grey = grey_fill(image, gap)
    harmonic = harmonic_fill(image, gap)
    assert np.all(grey[gap] == 0.5)
    assert np.allclose(harmonic[~gap], image[~gap])
    assert np.isfinite(harmonic[gap]).all()
