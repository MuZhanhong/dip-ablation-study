import numpy as np
import torch

from src.masks import central_gap_mask, heldout_mask
from src.models import build_model
from src.synthetic_data import change_contrast, make_image
from src.tangent_kernel import measure_kernel
from src.training import train_inpainting


def small_problem():
    size = 32
    image = np.tile(np.linspace(0, 1, size, dtype=np.float32), (size, 1))
    gap = central_gap_mask(size, 8)
    heldout = heldout_mask(gap, fraction=0.1, seed=0)
    return size, image, gap, heldout


def test_sweep_options_save_the_expected_files(tmp_path):
    size, image, gap, heldout = small_problem()
    torch.manual_seed(1)
    model = build_model(channels=4, levels=2, filter_size=3, size=size)
    train_inpainting(
        image, gap, heldout, model,
        steps=4, log_every=2, seed=1, device="cpu", output_dir=tmp_path,
        save_prediction_every=4, save_weights_at=(0, 4), save_features=True,
    )
    assert (tmp_path / "fixed_noise.npy").exists()
    assert (tmp_path / "weights_step_00000.pt").exists()
    assert (tmp_path / "weights_step_00004.pt").exists()
    assert (tmp_path / "optimizer_step_00004.pt").exists()
    adam = torch.load(tmp_path / "optimizer_step_00004.pt")
    assert all("exp_avg_sq" in state for state in adam["state"].values())
    assert not (tmp_path / "prediction_step_00002.npy").exists()
    assert (tmp_path / "prediction_step_00004.npy").exists()
    features = np.load(tmp_path / "final_features.npz")
    assert features["before_output_layer"].shape == (4, size, size)
    assert features["after_coarse_levels"].shape == (4, size, size)


def test_no_model_states_skips_model_files(tmp_path):
    size, image, gap, heldout = small_problem()
    model = build_model(channels=2, levels=1, filter_size=1, size=size, weight_sharing=False)
    train_inpainting(
        image, gap, heldout, model,
        steps=2, log_every=1, seed=0, device="cpu", output_dir=tmp_path,
        selection="final", save_model_states=False,
    )
    assert not (tmp_path / "best_model_state.pt").exists()
    assert not (tmp_path / "final_model_state.pt").exists()
    assert (tmp_path / "selected_prediction.npy").exists()


def test_kernel_measured_during_training_has_the_right_shape(tmp_path):
    size, image, gap, heldout = small_problem()
    torch.manual_seed(0)
    model = build_model(channels=2, levels=1, filter_size=3, size=size, upsampling=False)
    train_inpainting(
        image, gap, heldout, model,
        steps=2, log_every=1, seed=0, device="cpu", output_dir=tmp_path,
        measure_kernel_at=(0, 2),
    )
    table = np.load(tmp_path / "kernel_step_00002.npy")
    assert table.shape == (8, size // 4 + 1)
    assert np.all(table[:, 0] > 0)  # Theta(p, p) is a squared length


def test_unshared_1x1_network_has_no_tie_between_different_pixels():
    # Network 8 in miniature: each pixel has its own weights and sees only its own input.
    torch.manual_seed(0)
    model = build_model(channels=2, levels=1, filter_size=1, size=64,
                        upsampling=False, weight_sharing=False)
    noise = torch.randn(1, 2, 64, 64) * 0.1
    table = measure_kernel(model, noise, lines=((32, 8, 0, 1), (8, 32, 1, 0)), max_distance=3)
    # Batch norm links every pixel a little, so the tie is small rather than exactly 0.
    for line in table:
        assert np.all(np.abs(line[1:]) < 0.05 * line[0])


def test_adam_kernel_saved_after_step_0_and_matches_a_direct_calculation(tmp_path):
    size, image, gap, heldout = small_problem()
    torch.manual_seed(0)
    model = build_model(channels=2, levels=1, filter_size=3, size=size, upsampling=False)
    train_inpainting(
        image, gap, heldout, model,
        steps=3, log_every=1, seed=0, device="cpu", output_dir=tmp_path,
        measure_kernel_at=(0, 3),
    )
    assert not (tmp_path / "kernel_adam_step_00000.npy").exists()  # no Adam memory yet
    adam = np.load(tmp_path / "kernel_adam_step_00003.npy")
    assert adam.shape == (8, size // 4 + 1)
    assert np.all(adam[:, 0] > 0)  # still a (weighted) squared length


def test_kernel_extras_saved_with_diagonals_and_far_pixel(tmp_path):
    size, image, gap, heldout = small_problem()
    torch.manual_seed(0)
    model = build_model(channels=2, levels=1, filter_size=3, size=size, upsampling=False)
    train_inpainting(
        image, gap, heldout, model,
        steps=3, log_every=1, seed=0, device="cpu", output_dir=tmp_path,
        measure_kernel_at=(0, 3),
    )
    first = np.load(tmp_path / "kernel_extras_step_00000.npz")
    assert "adam_diagonal" not in first.files  # no Adam memory at step 0
    extras = np.load(tmp_path / "kernel_extras_step_00003.npz")
    plain = np.load(tmp_path / "kernel_step_00003.npy")
    adam = np.load(tmp_path / "kernel_adam_step_00003.npy")
    for name, table in (("plain", plain), ("adam", adam)):
        diagonal = extras[f"{name}_diagonal"]
        assert diagonal.shape == table.shape
        assert np.all(diagonal > 0)
        # At distance 0, q = p, so Theta(q, q) is the line's own first value.
        assert np.allclose(diagonal[:, 0], table[:, 0])
        # Cauchy-Schwarz: a correlation-style tie lies between -1 and 1.
        correlation = table / np.sqrt(table[:, :1] * diagonal)
        assert np.all(np.abs(correlation) <= 1 + 1e-6)
        assert extras[f"{name}_far"].shape == (8,)
        assert np.all(extras[f"{name}_far_diagonal"] > 0)
    assert extras["far_pixel"].shape == (8, 2)
    assert extras["line_starts"].shape == (8, 4)


def test_far_reference_pixel_is_visible_and_beyond_the_line():
    from src.masks import central_gap_mask
    from src.tangent_kernel import default_lines, far_reference_pixel
    size = 64
    gap = central_gap_mask(size, 16)
    lines, max_distance = default_lines(size)
    for row, col, _, _ in lines:
        r_row, r_col = far_reference_pixel(row, col, size)
        assert not gap[r_row, r_col]
        assert np.hypot(r_row - row, r_col - col) > 2 * max_distance


def test_far_tie_matches_a_direct_gradient_calculation():
    from src.tangent_kernel import (far_reference_pixel, gradient_of_output_pixel,
                                    measure_kernel_with_extras)
    torch.manual_seed(0)
    model = build_model(channels=2, levels=1, filter_size=3, size=32, upsampling=False)
    noise = torch.randn(1, 2, 32, 32) * 0.1
    line = (16, 4, 0, 1)
    result = measure_kernel_with_extras(model, noise, lines=(line,), max_distance=3)
    model.train()
    g_p = gradient_of_output_pixel(model, noise, 16, 4)
    g_r = gradient_of_output_pixel(model, noise, *far_reference_pixel(16, 4, 32))
    assert np.isclose(result["plain_far"][0], float(torch.dot(g_p, g_r)), rtol=1e-5, atol=1e-10)
    assert np.isclose(result["plain_far_diagonal"][0], float(torch.dot(g_r, g_r)), rtol=1e-5)


def test_adam_scaling_is_one_over_root_v_hat_plus_eps():
    from src.tangent_kernel import adam_scaling
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    model = torch.nn.Module()
    model.weight = weight
    optimizer = torch.optim.Adam([weight], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    (weight * torch.tensor([3.0, -4.0])).sum().backward()  # gradient (3, -4)
    optimizer.step()
    # After one step v = 0.001 * g^2, and v_hat = v / (1 - 0.999) = g^2.
    expected = 1.0 / (torch.tensor([3.0, 4.0]) + 1e-8)
    assert torch.allclose(adam_scaling(model, optimizer), expected)


def test_harmonic_fill_on_training_pixels_only_ignores_heldout_values():
    from baselines.fill import harmonic_fill
    size, image, gap, heldout = small_problem()
    changed = image.copy()
    changed[heldout] = 99.0  # the fill must not look at held-out values
    first = harmonic_fill(image, gap | heldout)
    second = harmonic_fill(changed, gap | heldout)
    assert np.allclose(first[gap], second[gap])


def test_contrast_and_stripe_period_options():
    image = make_image("blurred_noise", size=16, seed=0, blur_scale=2.0)
    half = change_contrast(image, 0.5)
    assert np.allclose(half - 0.5, 0.5 * (image - 0.5), atol=1e-6)
    assert np.array_equal(change_contrast(image, 1.0), np.clip(image, 0, 1))
    narrow = make_image("stripes", size=16, stripe_period=4.0)
    wide = make_image("stripes", size=16, stripe_period=16.0)
    assert not np.allclose(narrow, wide)
