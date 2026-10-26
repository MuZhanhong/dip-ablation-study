import numpy as np
import pytest
import torch

from src.masks import central_gap_mask, heldout_mask
from src.models import build_model
from src.training import train_inpainting


def test_training_uses_heldout_selection_and_returns_finite_prediction(tmp_path):
    size = 32
    image = np.tile(np.linspace(0, 1, size, dtype=np.float32), (size, 1))
    gap = central_gap_mask(size, 8)
    heldout = heldout_mask(gap, fraction=0.1, seed=0)
    model = build_model(channels=4, levels=2, filter_size=1)
    result = train_inpainting(
        image, gap, heldout, model,
        steps=4, log_every=2, seed=1,
        device="cpu", output_dir=tmp_path,
    )
    assert result.best_step in (2, 4)
    assert np.isfinite(result.best_heldout_mse)
    assert result.best_prediction.shape == image.shape
    assert np.isfinite(result.best_prediction).all()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "best_prediction.npy").exists()


def test_training_rejects_overlapping_masks():
    image = np.zeros((8, 8), dtype=np.float32)
    gap = np.zeros((8, 8), dtype=bool)
    gap[2, 2] = True
    heldout = np.zeros_like(gap)
    heldout[2, 2] = True
    model = build_model(channels=2, levels=1)
    with pytest.raises(ValueError, match="overlap"):
        train_inpainting(image, gap, heldout, model, steps=1, log_every=1)
