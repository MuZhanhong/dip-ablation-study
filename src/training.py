"""Fit a DIP network to the observed pixels."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.metrics import masked_mse, psnr, ssim
from src.tangent_kernel import adam_scaling, measure_kernel_with_extras


@dataclass
class TrainingResult:
    """Predictions and metrics saved during training."""

    best_prediction: np.ndarray
    best_step: int
    best_heldout_mse: float
    history: list[dict[str, float]]
    final_prediction: np.ndarray
    final_step: int
    selection: str
    selected_prediction: np.ndarray
    selected_step: int


def features_entering_final_layers(model: nn.Module, noise: torch.Tensor) -> dict[str, np.ndarray]:
    """Record two sets of features at full resolution, without changing the model.

    "after_coarse_levels": the output of the batch norm that follows the inner
    levels (and the last upsampling, if there is one), i.e. what the last
    full-resolution layers receive.
    "before_output_layer": what the final 1x1 output layer receives.
    Each has shape (channels, height, width).
    """
    top_level = model.network[0]
    children = list(top_level.children())
    captured: dict[str, torch.Tensor] = {}

    def keep(name):
        def hook(module, inputs, output):
            captured[name] = output.detach()
        return hook

    handles = [top_level.register_forward_hook(keep("before_output_layer"))]
    inner = [i for i, child in enumerate(children) if isinstance(child, nn.Sequential)]
    if inner:
        after = children[inner[0] + 1:]
        batch_norm = next(m for m in after if isinstance(m, nn.BatchNorm2d))
        handles.append(batch_norm.register_forward_hook(keep("after_coarse_levels")))
    with torch.no_grad():
        model(noise)
    for handle in handles:
        handle.remove()
    return {name: value[0].cpu().numpy().astype(np.float32) for name, value in captured.items()}


def train_inpainting(
    image: np.ndarray,
    gap_mask: np.ndarray,
    heldout_mask: np.ndarray,
    model: nn.Module,
    *,
    steps: int = 3000,
    learning_rate: float = 0.01,
    log_every: int = 50,
    noise_scale: float = 0.1,
    seed: int = 0,
    device: str | torch.device | None = None,
    output_dir: str | Path | None = None,
    selection: str = "heldout",
    save_prediction_every: int | None = None,
    save_model_states: bool = True,
    save_weights_at: tuple[int, ...] = (),
    measure_kernel_at: tuple[int, ...] = (),
    save_features: bool = False,
) -> TrainingResult:
    """Train on observed pixels; keep the best held-out or final prediction.

    Gap pixels are never used for training or checkpoint selection.
    Unshared networks use the final step because their held-out pixels have
    no location-specific training signal.

    Options for the sweep (all off by default, so older runs are unchanged):
    save_prediction_every: save the prediction every this many steps instead
        of at every logged step.
    save_model_states: False skips best/final model files (unshared networks
        have about 100 million weights, so each file is hundreds of MB).
    save_weights_at: steps at which to save the weights; 0 means before training.
    measure_kernel_at: steps at which to measure the tangent kernel; 0 means
        before training. Saved as kernel_step_XXXXX.npy, and from step 1 on
        also the Adam-weighted kernel, kernel_adam_step_XXXXX.npy. The diagonal
        Theta(q, q) along each line and the ties to one far reference pixel per
        line go in kernel_extras_step_XXXXX.npz.
    save_features: save the features entering the final layers at the end.
    The fixed input noise is always saved, so any saved weights can be rerun.
    """
    if selection not in ("heldout", "final"):
        raise ValueError("selection must be 'heldout' or 'final'")
    target = np.asarray(image, dtype=np.float32)
    gap = np.asarray(gap_mask, dtype=bool)
    heldout = np.asarray(heldout_mask, dtype=bool)
    if target.ndim != 2 or target.shape != gap.shape or target.shape != heldout.shape:
        raise ValueError("image and masks must be matching 2D arrays")
    if np.any(gap & heldout):
        raise ValueError("heldout_mask must not overlap gap_mask")
    train_mask = ~(gap | heldout)
    if not np.any(train_mask):
        raise ValueError("No training pixels remain after masking")
    if not np.any(heldout):
        raise ValueError("heldout_mask must contain validation pixels")
    if steps < 1 or log_every < 1:
        raise ValueError("steps and log_every must be positive")
    if not hasattr(model, "config") or not hasattr(model.config, "channels"):
        raise ValueError("model must expose model.config.channels")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    height, width = target.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fixed_noise = torch.randn(
        1, int(model.config.channels), height, width, generator=generator
    ) * float(noise_scale)
    fixed_noise = fixed_noise.to(device)
    target_tensor = torch.from_numpy(target[None, None]).to(device)
    train_tensor = torch.from_numpy(train_mask[None, None]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    out_path = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / "fixed_noise.npy", fixed_noise[0].cpu().numpy())

    def save_extras(step: int) -> None:
        if out_path is None:
            return
        if step in save_weights_at:
            torch.save(model.state_dict(), out_path / f"weights_step_{step:05d}.pt")
            # Adam's running averages of each weight's gradient and squared gradient:
            # needed to work out the Adam-weighted tangent kernel offline.
            torch.save(optimizer.state_dict(), out_path / f"optimizer_step_{step:05d}.pt")
        if step in measure_kernel_at:
            # Adam has no memory before the first step, so only the plain kernel at 0.
            scaling = None if step == 0 else adam_scaling(model, optimizer)
            kernel = measure_kernel_with_extras(model, fixed_noise, scaling=scaling)
            np.save(out_path / f"kernel_step_{step:05d}.npy", kernel["plain"])
            if scaling is not None:
                np.save(out_path / f"kernel_adam_step_{step:05d}.npy", kernel["adam"])
            # Theta(q, q) along each line and the far reference pixel's ties (both
            # plain and, after step 0, Adam): a few kB per step.
            extras = {k: v for k, v in kernel.items() if k not in ("plain", "adam")}
            np.savez(out_path / f"kernel_extras_step_{step:05d}.npz", **extras)

    save_extras(0)

    best_mse = float("inf")
    best_step = 0
    best_prediction: np.ndarray | None = None
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(fixed_noise)
        loss = (((prediction - target_tensor) ** 2) * train_tensor).sum() / train_tensor.sum()
        loss.backward()
        optimizer.step()
        save_extras(step)

        if step % log_every != 0 and step != steps:
            continue

        # BatchNorm should use the same statistics as during fitting.
        with torch.no_grad():
            pred = model(fixed_noise)[0, 0].detach().cpu().numpy().astype(np.float32)
        gap_mse = masked_mse(pred, target, gap)
        ys, xs = np.where(gap)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        row = {
            "step": float(step),
            "train_mse": masked_mse(pred, target, train_mask),
            "heldout_mse": masked_mse(pred, target, heldout),
            "gap_mse": gap_mse,
            "psnr": float("inf") if gap_mse == 0 else float(10.0 * np.log10(1.0 / gap_mse)),
            "ssim": ssim(pred[y0:y1, x0:x1], target[y0:y1, x0:x1]),
        }
        history.append(row)
        if row["heldout_mse"] < best_mse:
            best_mse = row["heldout_mse"]
            best_step = step
            best_prediction = pred.copy()
            if out_path is not None:
                np.save(out_path / "best_prediction.npy", best_prediction)
                if save_model_states:
                    torch.save(model.state_dict(), out_path / "best_model_state.pt")
        save_this_one = (
            save_prediction_every is None
            or step % save_prediction_every == 0
            or step == steps
        )
        if out_path is not None and save_this_one:
            np.save(out_path / f"prediction_step_{step:05d}.npy", pred)

    if best_prediction is None:
        raise RuntimeError("Training ended without producing a checkpoint")

    final_prediction = pred.copy()
    if selection == "heldout":
        selected_step, selected_prediction = best_step, best_prediction
    else:
        selected_step, selected_prediction = step, final_prediction

    if out_path is not None:
        np.save(out_path / "final_prediction.npy", final_prediction)
        if save_model_states:
            torch.save(model.state_dict(), out_path / "final_model_state.pt")
        np.save(out_path / "selected_prediction.npy", selected_prediction)
        if save_features:
            features = features_entering_final_layers(model, fixed_noise)
            np.savez_compressed(out_path / "final_features.npz", **features)
        fields = ["step", "train_mse", "heldout_mse", "gap_mse", "psnr", "ssim"]
        with (out_path / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)

    return TrainingResult(
        best_prediction=best_prediction,
        best_step=best_step,
        best_heldout_mse=best_mse,
        history=history,
        final_prediction=final_prediction,
        final_step=step,
        selection=selection,
        selected_prediction=selected_prediction,
        selected_step=selected_step,
    )

  
