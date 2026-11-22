"""Run the DIP ablation experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

# Allow running this file directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.masks import central_gap_mask, heldout_mask
from src.models import DIPHourglass, ablation_config
from src.synthetic_data import change_contrast, make_image
from src.training import train_inpainting


def list_of_steps(text: str | None) -> tuple[int, ...]:
    """Turn "0,500,5000" into (0, 500, 5000); None or "" into ()."""
    if not text:
        return ()
    return tuple(int(part) for part in text.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments" / "config.yaml"))
    parser.add_argument("--image-type", choices=["gradient", "blurred_noise", "stripes"])
    parser.add_argument("--blur-scale", type=float, help="Only used for blurred_noise")
    parser.add_argument("--ablation-id", type=int, choices=range(1, 9))
    parser.add_argument("--steps", type=int, help="Override configured training steps")
    parser.add_argument("--channels", type=int, help="Override configured channel width")
    parser.add_argument("--seed", type=int, help="Model initialization and input-noise seed")
    parser.add_argument("--image-seed", type=int, help="Synthetic image seed (separate from --seed)")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--device", default=None, help="For example cpu or cuda")
    # Options added for the November sweep. Defaults leave older runs unchanged.
    parser.add_argument("--stripe-period", type=float, default=8.0, help="Only used for stripes")
    parser.add_argument("--contrast", type=float, default=1.0,
                        help="Scale each pixel's distance from 0.5 by this factor")
    parser.add_argument("--save-prediction-every", type=int,
                        help="Save predictions every N steps (default: every logged step)")
    parser.add_argument("--no-model-states", action="store_true",
                        help="Do not save best/final model files (use for unshared networks)")
    parser.add_argument("--save-weights-at", default="",
                        help="Comma-separated steps at which to save weights, 0 = before training")
    parser.add_argument("--measure-kernel-at", default="",
                        help="Comma-separated steps at which to measure the tangent kernel")
    parser.add_argument("--save-features", action="store_true",
                        help="Save the features entering the final layers at the end")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_cfg = config["data"]
    mask_cfg = config["mask"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    size = int(data_cfg["size"])
    channels = int(args.channels or train_cfg["channels"])
    steps = int(args.steps or train_cfg["steps"])
    run_seed = int(config["seed"] if args.seed is None else args.seed)
    image_seed = int(config["seed"] if args.image_seed is None else args.image_seed)

    image_specs: list[tuple[str, float | None]] = []
    if args.image_type:
        if args.image_type == "blurred_noise":
            scales = [args.blur_scale] if args.blur_scale is not None else data_cfg["blur_scales"]
            image_specs = [(args.image_type, float(scale)) for scale in scales]
        else:
            image_specs = [(args.image_type, None)]
    else:
        for kind in data_cfg["image_types"]:
            if kind == "blurred_noise":
                image_specs.extend((kind, float(scale)) for scale in data_cfg["blur_scales"])
            else:
                image_specs.append((kind, None))

    ablation_ids = [args.ablation_id] if args.ablation_id else list(range(1, 9))
    root_out = Path(args.output_dir)
    gap = central_gap_mask(size, int(mask_cfg["gap_size"]))
    validation = heldout_mask(
        gap, float(mask_cfg["heldout_fraction"]), int(mask_cfg["heldout_seed"])
    )

    for kind, scale in image_specs:
        image = make_image(
            kind, size=size, seed=image_seed,
            blur_scale=1.0 if scale is None else scale,
            stripe_period=args.stripe_period,
        )
        if args.contrast != 1.0:
            image = change_contrast(image, args.contrast)
        spec_name = kind if scale is None else f"{kind}_scale_{scale:g}"
        if kind == "stripes" and args.stripe_period != 8.0:
            spec_name = f"{spec_name}_period_{args.stripe_period:g}"
        if args.image_seed is not None:
            spec_name = f"{spec_name}_image_seed_{image_seed}"
        if args.contrast != 1.0:
            spec_name = f"{spec_name}_contrast_{args.contrast:g}"
        for ablation_id in ablation_ids:
            model_config = ablation_config(
                ablation_id, channels=channels, levels=int(train_cfg["levels"])
            )
            # Reset the model seed for each ablation.
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            model = DIPHourglass(model_config)
            run_dir = root_out / spec_name / f"ablation_{ablation_id}"
            selection = eval_cfg[
                "checkpoint_selection_shared" if model_config.weight_sharing
                else "checkpoint_selection_unshared"
            ]
            result = train_inpainting(
                image, gap, validation, model,
                steps=steps,
                learning_rate=float(train_cfg["learning_rate"]),
                log_every=int(train_cfg["log_every"]),
                noise_scale=float(train_cfg["fixed_noise_scale"]),
                seed=run_seed,
                device=args.device,
                output_dir=run_dir,
                selection=selection,
                save_prediction_every=args.save_prediction_every,
                save_model_states=not args.no_model_states,
                save_weights_at=list_of_steps(args.save_weights_at),
                measure_kernel_at=list_of_steps(args.measure_kernel_at),
                save_features=args.save_features,
            )
            selected = next(r for r in result.history if int(r["step"]) == result.selected_step)
            print(
                f"{spec_name} | ablation {ablation_id} | "
                f"selection={result.selection} | step={result.selected_step} | "
                f"gap MSE={selected['gap_mse']:.6g} | "
                f"saved={run_dir}"
            )


if __name__ == "__main__":
    main()
