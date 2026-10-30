import pytest
import torch
from torch import nn

from src.models import (
    DIPHourglass,
    LocallyConnected2d,
    ModelConfig,
    ablation_config,
    build_model,
)


@pytest.mark.parametrize("ablation_id", range(1, 9))
def test_all_ablation_models_preserve_shape_and_range(ablation_id):
    config = ablation_config(ablation_id, channels=4, levels=3)
    model = DIPHourglass(config)
    x = torch.randn(1, 4, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)
    assert torch.isfinite(y).all()
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_shared_model_has_fewer_parameters_than_unshared_model():
    shared = build_model(channels=4, levels=2, weight_sharing=True)
    unshared = build_model(channels=4, levels=2, weight_sharing=False)
    assert sum(p.numel() for p in shared.parameters()) < sum(
        p.numel() for p in unshared.parameters()
    )


def test_unshared_model_is_locally_connected_throughout():
    model = build_model(channels=4, levels=2, weight_sharing=False)
    assert not any(isinstance(layer, nn.Conv2d) for layer in model.modules())
    assert isinstance(model.network[-2], LocallyConnected2d)
    assert model.network[-2].k == 1


def test_shared_model_has_shared_output_layer():
    model = build_model(channels=4, levels=2, weight_sharing=True)
    assert isinstance(model.network[-2], nn.Conv2d)
    assert model.network[-2].kernel_size == (1, 1)


@pytest.mark.parametrize("ablation_id", [5, 6, 7, 8])
def test_upsampling_off_keeps_every_layer_at_full_resolution(ablation_id):
    model = DIPHourglass(ablation_config(ablation_id, channels=4, levels=4))
    x = torch.randn(1, 4, 64, 64)
    assert model(x).shape == (1, 1, 64, 64)
    assert not any(isinstance(layer, (nn.AvgPool2d, nn.Upsample, nn.ConvTranspose2d))
                   for layer in model.modules())


def test_filter_size_is_validated():
    with pytest.raises(ValueError, match="filter_size"):
        ModelConfig(filter_size=5)


def test_invalid_ablation_id_is_rejected():
    with pytest.raises(ValueError, match="ablation_id"):
        ablation_config(9)


def test_model_supports_backward_pass():
    model = build_model(channels=4, levels=2, filter_size=1, size=16)
    x = torch.randn(1, 4, 16, 16)
    loss = model(x).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

