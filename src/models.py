"""DIP-style hourglass models for the architecture ablation study.

The model has no encoder-to-decoder skip connections. Average pooling forms the
encoder pyramid. In the decoder, upsampling=True uses bilinear interpolation;
upsampling=False uses learned transposed convolutions instead. With
weight_sharing=True, one encoder block and one decoder block are reused at
each pyramid level.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one ablation model."""

    channels: int = 32
    levels: int = 4
    filter_size: int = 3
    upsampling: bool = True
    weight_sharing: bool = True
    output_channels: int = 1

    def __post_init__(self) -> None:
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if self.levels < 1:
            raise ValueError("levels must be >= 1")
        if self.filter_size not in (1, 3):
            raise ValueError("filter_size must be 1 or 3")
        if self.output_channels < 1:
            raise ValueError("output_channels must be >= 1")


class ConvBlock(nn.Module):
    """Two convolution layers with ReLU activations and no normalization."""

    def __init__(self, channels: int, filter_size: int) -> None:
        super().__init__()
        padding = filter_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, filter_size, padding=padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, filter_size, padding=padding),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DIPHourglass(nn.Module):
    """A no-skip hourglass network operating on fixed noise inputs.

    Input and output spatial dimensions must be divisible by 2 ** (levels - 1).
    Output is constrained to [0, 1] with a sigmoid, matching the synthetic
    grayscale target range.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs: object) -> None:
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("Pass either a ModelConfig or keyword configuration, not both")
        self.config = config if config is not None else ModelConfig(**kwargs)
        c = self.config.channels
        k = self.config.filter_size
        self.encoder_shared = ConvBlock(c, k) if self.config.weight_sharing else None
        self.decoder_shared = ConvBlock(c, k) if self.config.weight_sharing else None
        if not self.config.weight_sharing:
            self.encoder_blocks = nn.ModuleList(
                ConvBlock(c, k) for _ in range(self.config.levels)
            )
            self.decoder_blocks = nn.ModuleList(
                ConvBlock(c, k) for _ in range(self.config.levels)
            )
        else:
            self.encoder_blocks = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()

        # Learned 2x upsampling is the non-bilinear ablation.
        self.learned_upsamplers = nn.ModuleList(
            nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)
            for _ in range(max(0, self.config.levels - 1))
        ) if not self.config.upsampling else nn.ModuleList()
        self.output_layer = nn.Conv2d(c, self.config.output_channels, kernel_size=1)

    def _encoder(self, level: int, x: torch.Tensor) -> torch.Tensor:
        block = self.encoder_shared if self.config.weight_sharing else self.encoder_blocks[level]
        assert block is not None
        return block(x)

    def _decoder(self, level: int, x: torch.Tensor) -> torch.Tensor:
        block = self.decoder_shared if self.config.weight_sharing else self.decoder_blocks[level]
        assert block is not None
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input must have shape [batch, channels, height, width]")
        if x.shape[1] != self.config.channels:
            raise ValueError(
                f"Expected {self.config.channels} input channels, got {x.shape[1]}"
            )
        divisor = 2 ** (self.config.levels - 1)
        if x.shape[-2] % divisor or x.shape[-1] % divisor:
            raise ValueError(
                f"Height and width must be divisible by {divisor} for "
                f"{self.config.levels} levels"
            )

        x = self._encoder(0, x)
        for level in range(1, self.config.levels):
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = self._encoder(level, x)

        # Decode back to input resolution without encoder-to-decoder skip paths.
        for decoder_level in range(self.config.levels - 2, -1, -1):
            if self.config.upsampling:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            else:
                x = self.learned_upsamplers[decoder_level](x)
            x = self._decoder(decoder_level, x)

        return torch.sigmoid(self.output_layer(x))


def build_model(
    *,
    channels: int = 32,
    levels: int = 4,
    filter_size: int = 3,
    upsampling: bool = True,
    weight_sharing: bool = True,
    output_channels: int = 1,
) -> DIPHourglass:
    """Build a model from explicit ablation-factor values."""
    return DIPHourglass(
        ModelConfig(
            channels=channels,
            levels=levels,
            filter_size=filter_size,
            upsampling=upsampling,
            weight_sharing=weight_sharing,
            output_channels=output_channels,
        )
    )


def ablation_config(ablation_id: int, *, channels: int = 32, levels: int = 4) -> ModelConfig:
    """Return the configuration for one of the eight preregistered ablations."""
    factors = {
        1: (True, 3, True),
        2: (True, 3, False),
        3: (True, 1, True),
        4: (True, 1, False),
        5: (False, 3, True),
        6: (False, 3, False),
        7: (False, 1, True),
        8: (False, 1, False),
    }
    if ablation_id not in factors:
        raise ValueError("ablation_id must be an integer from 1 to 8")
    upsampling, filter_size, weight_sharing = factors[ablation_id]
    return ModelConfig(
        channels=channels,
        levels=levels,
        filter_size=filter_size,
        upsampling=upsampling,
        weight_sharing=weight_sharing,
    )
