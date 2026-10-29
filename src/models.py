"""DIP-style 17-layer model builder for the architecture ablation study.

The three switches follow the Experiments Week 2 design:
- upsampling=True: average-pool encoder and bilinear decoder; False removes both
  pooling and upsampling, keeping every layer at the input resolution.
- filter_size: use 3x3 or 1x1 filters for the main convolution layers.
- weight_sharing=True: normal convolutions share filters across spatial positions;
  False uses locally connected layers with separate filters at each pixel.
The final 1x1 output convolution is always a normal spatially shared Conv2d.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class LocallyConnected2d(nn.Module):
    """Same-padded stride-1 convolution with separate weights at every pixel."""

    def __init__(self, in_ch: int, out_ch: int, k: int, size: int) -> None:
        super().__init__()
        self.k = k
        self.size = size
        self.out_ch = out_ch
        positions = size * size
        fan_in = in_ch * k * k
        self.weight = nn.Parameter(torch.empty(positions, out_ch, fan_in))
        self.bias = nn.Parameter(torch.empty(positions, out_ch))
        # Match nn.Conv2d's initial distribution so both switch settings start alike.
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patches = F.unfold(x, self.k, padding=self.k // 2)
        out = torch.einsum("bpl,lop->bol", patches, self.weight)
        out = out + self.bias.t()
        return out.view(batch, self.out_ch, self.size, self.size)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one of the eight architecture ablations."""

    channels: int = 32
    levels: int = 4
    filter_size: int = 3
    upsampling: bool = True
    weight_sharing: bool = True
    output_channels: int = 1
    size: int = 64

    def __post_init__(self) -> None:
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if self.levels < 1:
            raise ValueError("levels must be >= 1")
        if self.filter_size not in (1, 3):
            raise ValueError("filter_size must be 1 or 3")
        if self.output_channels < 1:
            raise ValueError("output_channels must be >= 1")
        if self.size < 1 or self.size % (2 ** (self.levels - 1)):
            raise ValueError("size must be positive and divisible by 2 ** (levels - 1)")


def build_net(
    upsample: bool = True,
    k: int = 3,
    shared: bool = True,
    C: int = 32,
    levels: int = 4,
    in_ch: int = 32,
    out_ch: int = 1,
    size: int = 64,
    up_mode: str = "bilinear",
) -> nn.Sequential:
    """Build all eight networks using the tested Week 2 architecture builder."""

    if k not in (1, 3):
        raise ValueError("k must be 1 or 3")
    if levels < 1 or C < 1 or in_ch < 1 or out_ch < 1 or size < 1:
        raise ValueError("channels, levels, and size must be positive")
    if upsample and size % (2 ** (levels - 1)):
        raise ValueError("size must be divisible by 2 ** (levels - 1) when upsampling")

    def conv(cin: int, cout: int, kk: int, spatial_size: int) -> nn.Module:
        if shared:
            return nn.Conv2d(cin, cout, kk, padding=kk // 2)
        return LocallyConnected2d(cin, cout, kk, spatial_size)

    def unit(cin: int, cout: int, kk: int, spatial_size: int) -> list[nn.Module]:
        return [
            conv(cin, cout, kk, spatial_size),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.2),
        ]

    def level(i: int, cin: int, spatial_size: int) -> nn.Sequential:
        inner = spatial_size // 2 if upsample else spatial_size
        layers: list[nn.Module] = [conv(cin, C, k, spatial_size)]
        if upsample:
            layers.append(nn.AvgPool2d(2))
        layers += [nn.BatchNorm2d(C), nn.LeakyReLU(0.2)]
        layers += unit(C, C, k, inner)
        if i < levels - 1:
            layers.append(level(i + 1, C, inner))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode=up_mode))
        layers.append(nn.BatchNorm2d(C))
        layers += unit(C, C, k, spatial_size)
        layers += unit(C, C, 1, spatial_size)
        return nn.Sequential(*layers)

    return nn.Sequential(level(0, in_ch, size), nn.Conv2d(C, out_ch, 1), nn.Sigmoid())


class DIPHourglass(nn.Module):
    """Configuration-compatible wrapper around the tested sequential builder."""

    def __init__(self, config: ModelConfig | None = None, **kwargs: object) -> None:
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("Pass either a ModelConfig or keyword configuration, not both")
        self.config = config if config is not None else ModelConfig(**kwargs)
        self.network = build_net(
            upsample=self.config.upsampling,
            k=self.config.filter_size,
            shared=self.config.weight_sharing,
            C=self.config.channels,
            levels=self.config.levels,
            in_ch=self.config.channels,
            out_ch=self.config.output_channels,
            size=self.config.size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input must have shape [batch, channels, height, width]")
        if x.shape[1] != self.config.channels:
            raise ValueError(
                f"Expected {self.config.channels} input channels, got {x.shape[1]}"
            )
        if x.shape[-2:] != (self.config.size, self.config.size):
            raise ValueError(
                f"Expected spatial shape {(self.config.size, self.config.size)}, "
                f"got {tuple(x.shape[-2:])}"
            )
        return self.network(x)


def build_model(
    *,
    channels: int = 32,
    levels: int = 4,
    filter_size: int = 3,
    upsampling: bool = True,
    weight_sharing: bool = True,
    output_channels: int = 1,
    size: int = 64,
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
            size=size,
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
