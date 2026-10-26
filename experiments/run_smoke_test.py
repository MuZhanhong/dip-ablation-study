"""Quick smoke test for the current data/mask/baseline scaffold."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from baselines.fill import grey_fill, harmonic_fill
from src.masks import central_gap_mask, heldout_mask
from src.synthetic_data import make_image


def main() -> None:
    size, gap_size = 64, 16
    gap = central_gap_mask(size, gap_size)
    heldout = heldout_mask(gap, fraction=0.05, seed=0)
    cases = [
        ("gradient", None),
        ("blurred_noise", 0), ("blurred_noise", 1), ("blurred_noise", 2),
        ("blurred_noise", 4), ("blurred_noise", 8), ("stripes", None),
    ]
    for kind, scale in cases:
        image = make_image(
            kind, size=size, seed=0, blur_scale=1 if scale is None else scale
        )
        grey = grey_fill(image, gap)
        harmonic = harmonic_fill(image, gap)
        assert image.shape == (size, size) and np.isfinite(image).all()
        assert not np.any(gap & heldout)
        assert np.isfinite(grey[gap]).all() and np.isfinite(harmonic[gap]).all()
    print(f"Smoke test passed: generated {len(cases)} images and both fill baselines.")
    print("This does not yet train the eight DIP variants or fully connected baseline.")


if __name__ == "__main__":
    main()
