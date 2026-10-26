"""Training loop for masked DIP image reconstruction."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.metrics import masked_mse, psnr, ssim


@dataclass
class TrainingResult:
    """Best-held-out checkpoint and per-checkpoint metric history."""

    best_prediction: np.ndarray
    best_step: int
    best_heldout_mse: float
    history: list[dict[str, float]]


def train_inpainting(
    image: np.ndarray,
    gap_mask: np.ndarray,
    heldout_mask: np.ndarray,
    model: nn.Module,
    *,
    steps: int = 3000,
    learning_rate: float = 0.01,
    log_every: int = 50,
    noise_channels: int = 32,
    noise_scale: float = 0.1,
    seed: int = 0,
    device: str | torch.device | None = None,
    output_dir: str | Path | None = None,
) -> TrainingResult:
    """Fit model to visible non-held-out pixels and select by held-out MSE.

    The target image is used for training only where both masks are False.
    Held-out and gap pixels are excluded from the loss. Gap metrics are
    reported for analysis but never used to select a checkpoint.
    """
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

        if step % log_every != 0 and step != steps:
            continue

        model.eval()
        with torch.no_grad():
            pred = model(fixed_noise)[0, 0].detach().cpu().numpy().astype(np.float32)
        row = {
            "step": float(step),
            "train_mse": masked_mse(pred, target, train_mask),
            "heldout_mse": masked_mse(pred, target, heldout),
            "gap_mse": masked_mse(pred, target, gap),
            "psnr": psnr(pred, target),
            "ssim": ssim(pred, target),
        }
        history.append(row)
        if row["heldout_mse"] < best_mse:
            best_mse = row["heldout_mse"]
            best_step = step
            best_prediction = pred.copy()
            if out_path is not None:
                np.save(out_path / "best_prediction.npy", best_prediction)
                torch.save(model.state_dict(), out_path / "best_model_state.pt")
        if out_path is not None:
            np.save(out_path / f"prediction_step_{step:05d}.npy", pred)

    if best_prediction is None:
        raise RuntimeError("Training ended without producing a checkpoint")

    if out_path is not None:
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
    )
