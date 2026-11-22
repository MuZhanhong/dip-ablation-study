"""Grey and harmonic fills, on exactly the pixels the networks are trained on.

The networks never see the gap or the held-out pixels. So the harmonic fill
here treats both as unknown: it solves the Laplace equation for every gap and
held-out pixel, using only the training pixels as boundary values. Its
held-out error is then comparable with a network's.

No training and no seeds, so one run per image, a few seconds in total.
Output: <output-dir>/<image name>/<baseline>/selected_prediction.npy and
metrics.csv (same columns as a network run, step = 0).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.fill import grey_fill, harmonic_fill
from src.masks import central_gap_mask, heldout_mask
from src.metrics import masked_mse, ssim
from src.synthetic_data import make_image


def metrics_row(prediction, image, gap, heldout):
    """The same numbers a network run logs."""
    train = ~(gap | heldout)
    gap_mse = masked_mse(prediction, image, gap)
    ys, xs = np.where(gap)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return {
        "step": 0.0,
        "train_mse": masked_mse(prediction, image, train),
        "heldout_mse": masked_mse(prediction, image, heldout),
        "gap_mse": gap_mse,
        "psnr": float("inf") if gap_mse == 0 else float(10.0 * np.log10(1.0 / gap_mse)),
        "ssim": ssim(prediction[box], image[box]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments" / "config.yaml"))
    parser.add_argument("--image-type", choices=["gradient", "blurred_noise", "stripes"],
                        default="blurred_noise")
    parser.add_argument("--blur-scale", type=float, default=4.0)
    parser.add_argument("--stripe-period", type=float, default=8.0)
    parser.add_argument("--image-seeds", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "baselines"))
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    size = int(config["data"]["size"])
    mask_cfg = config["mask"]
    gap = central_gap_mask(size, int(mask_cfg["gap_size"]))
    heldout = heldout_mask(gap, float(mask_cfg["heldout_fraction"]), int(mask_cfg["heldout_seed"]))
    unknown = gap | heldout

    for image_seed in (int(part) for part in args.image_seeds.split(",")):
        image = make_image(args.image_type, size=size, seed=image_seed,
                           blur_scale=args.blur_scale, stripe_period=args.stripe_period)
        # Same folder names as run_experiment.py, so results line up with the networks.
        name = args.image_type
        if args.image_type == "blurred_noise":
            name = f"{name}_scale_{args.blur_scale:g}"
        if args.image_type == "stripes" and args.stripe_period != 8.0:
            name = f"{name}_period_{args.stripe_period:g}"
        name = f"{name}_image_seed_{image_seed}"

        fills = {"grey": grey_fill(image, unknown), "harmonic": harmonic_fill(image, unknown)}
        for baseline, prediction in fills.items():
            run_dir = Path(args.output_dir) / name / baseline
            run_dir.mkdir(parents=True, exist_ok=True)
            np.save(run_dir / "selected_prediction.npy", prediction.astype(np.float32))
            row = metrics_row(prediction, image, gap, heldout)
            with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            print(f"{name} | {baseline} | gap MSE={row['gap_mse']:.6g} | "
                  f"held-out MSE={row['heldout_mse']:.6g}")


if __name__ == "__main__":
    main()
