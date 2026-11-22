"""Measure the tangent kernel of a network along lines of output pixels.

The tangent kernel between two output pixels p and q is the dot product of
their gradients with respect to all the weights:

    Theta(p, q) = g_p . g_q

It says how much a training step that fixes the output at q also moves the
output at p.

Training uses Adam, which divides each weight's step by sqrt(v_hat) + eps,
where v_hat is Adam's running average of that weight's squared gradient.
Ignoring momentum, the step is then described by the Adam-weighted kernel

    Theta_adam(p, q) = sum over weights i of g_p[i] * g_q[i] / (sqrt(v_hat[i]) + eps)

The plain kernel depends only on the network's weights; the Adam-weighted one
also depends on Adam's memory of past gradients on this image.
"""
from __future__ import annotations

import numpy as np
import torch


def default_lines(size):
    """Starting pixels and directions (row, col, row step, col step), and how far to go.

    For a 64x64 image (gap = rows and columns 24 to 39), each line runs 16 pixels.
    Horizontal, moving right along row 32: from column 8 (visible, ends at the
    gap's edge), from column 24 (the gap's first column, runs across the gap),
    from column 40 (just right of the gap, visible); and along row 16 from
    column 24 (above the gap, visible).
    Vertical: the same four lines turned onto column 32 (and column 16), moving down.
    """
    a, b, c, d = size // 8, 3 * size // 8, 5 * size // 8, size // 4
    middle = size // 2
    lines = (
        (middle, a, 0, 1), (middle, b, 0, 1), (middle, c, 0, 1), (d, b, 0, 1),
        (a, middle, 1, 0), (b, middle, 1, 0), (c, middle, 1, 0), (b, d, 1, 0),
    )
    return lines, size // 4


def gradient_of_output_pixel(model, noise, row, col):
    model.zero_grad(set_to_none=False)
    output = model(noise)
    output[0, 0, row, col].backward()
    pieces = []
    for weights in model.parameters():
        if weights.grad is None:
            pieces.append(torch.zeros(weights.numel(), device=weights.device))
        else:
            pieces.append(weights.grad.reshape(-1).clone())
    return torch.cat(pieces)


def adam_scaling(model, optimizer):
    """Return 1 / (sqrt(v_hat) + eps) for every weight, in the same order as the gradients.

    v_hat is Adam's running average of the squared gradient, divided by
    1 - beta2^t to correct its start from 0. Weights Adam has not updated yet
    get 0 (their gradient is 0 anyway). Needs at least one optimizer step.
    """
    settings = {}
    for group in optimizer.param_groups:
        for weights in group["params"]:
            settings[weights] = (group["betas"][1], group["eps"])
    pieces = []
    for weights in model.parameters():
        state = optimizer.state.get(weights, {})
        if "exp_avg_sq" not in state:
            pieces.append(torch.zeros(weights.numel(), device=weights.device))
            continue
        beta2, eps = settings[weights]
        t = float(state["step"])
        v_hat = state["exp_avg_sq"].reshape(-1) / (1 - beta2 ** t)
        pieces.append(1.0 / (v_hat.sqrt() + eps))
    return torch.cat(pieces)


def far_reference_pixel(row, col, size):
    """A visible pixel far from (row, col): the farthest of four points set in from the corners.

    The corners are inset by 3 * size // 16 (12 pixels for a 64x64 image): network 5
    sees 12 pixels in every direction, so at this inset the reference pixel is not
    distorted by the zero padding at the border, and it is still outside the central
    gap. For the default lines the reference is about 34 to 47 pixels from the line's
    start, beyond twice the 16 pixels a line covers.
    """
    inset = 3 * size // 16
    corners = [(inset, inset), (inset, size - 1 - inset),
               (size - 1 - inset, inset), (size - 1 - inset, size - 1 - inset)]
    return max(corners, key=lambda c: (c[0] - row) ** 2 + (c[1] - col) ** 2)


def kernel_along_a_line(model, noise, row, col, row_step, col_step, max_distance, scaling=None):
    """Ties from the line's start p to each pixel q on the line, and each q's own Theta(q, q).

    Returns a dict of lists: "plain" Theta(p, q), "plain_diagonal" Theta(q, q), and,
    if scaling is given, the same two with Adam's weighting ("adam", "adam_diagonal").
    Theta(q, q) costs nothing extra (gradient_q is already computed), and lets the tie be
    normalised like a correlation, Theta(p, q) / sqrt(Theta(p, p) Theta(q, q)), instead
    of only dividing by Theta(p, p).
    """
    gradient_p = gradient_of_output_pixel(model, noise, row, col)
    out = {"plain": [], "plain_diagonal": [], "adam": [], "adam_diagonal": []}
    for distance in range(max_distance + 1):
        q_row = row + distance * row_step
        q_col = col + distance * col_step
        gradient_q = gradient_of_output_pixel(model, noise, q_row, q_col)
        out["plain"].append(float(torch.dot(gradient_p, gradient_q)))
        out["plain_diagonal"].append(float(torch.dot(gradient_q, gradient_q)))
        if scaling is not None:
            # Same gradients, each weight's product multiplied by Adam's factor.
            out["adam"].append(float(torch.dot(gradient_p * scaling, gradient_q)))
            out["adam_diagonal"].append(float(torch.dot(gradient_q * scaling, gradient_q)))
    # One far visible reference pixel r: where the tie settles far away (the floor),
    # for curves that have not settled by the end of the line.
    r_row, r_col = far_reference_pixel(row, col, noise.shape[-1])
    gradient_r = gradient_of_output_pixel(model, noise, r_row, r_col)
    out["far_pixel"] = (r_row, r_col)
    out["plain_far"] = float(torch.dot(gradient_p, gradient_r))
    out["plain_far_diagonal"] = float(torch.dot(gradient_r, gradient_r))
    if scaling is not None:
        out["adam_far"] = float(torch.dot(gradient_p * scaling, gradient_r))
        out["adam_far_diagonal"] = float(torch.dot(gradient_r * scaling, gradient_r))
    return out


def measure_kernel_with_extras(model, noise, lines=None, max_distance=None, scaling=None):
    """Measure every line and return a dict of arrays.

    Always: "plain" and "plain_diagonal", shape (lines, max_distance + 1);
    "plain_far" and "plain_far_diagonal", shape (lines,); "far_pixel", shape (lines, 2);
    "line_starts", shape (lines, 4) = (row, col, row step, col step).
    With scaling (from adam_scaling): the same with "adam" in place of "plain".
    """
    if lines is None or max_distance is None:
        lines, max_distance = default_lines(noise.shape[-1])
    was_training = model.training
    model.train()
    names = ["plain"] + (["adam"] if scaling is not None else [])
    result = {}
    for name in names:
        result[name] = np.zeros((len(lines), max_distance + 1))
        result[f"{name}_diagonal"] = np.zeros((len(lines), max_distance + 1))
        result[f"{name}_far"] = np.zeros(len(lines))
        result[f"{name}_far_diagonal"] = np.zeros(len(lines))
    result["far_pixel"] = np.zeros((len(lines), 2), dtype=int)
    result["line_starts"] = np.array(lines, dtype=int)
    for i, (row, col, row_step, col_step) in enumerate(lines):
        line = kernel_along_a_line(
            model, noise, row, col, row_step, col_step, max_distance, scaling
        )
        result["far_pixel"][i] = line["far_pixel"]
        for name in names:
            result[name][i] = line[name]
            result[f"{name}_diagonal"][i] = line[f"{name}_diagonal"]
            result[f"{name}_far"][i] = line[f"{name}_far"]
            result[f"{name}_far_diagonal"][i] = line[f"{name}_far_diagonal"]
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return result


def measure_kernel(model, noise, lines=None, max_distance=None, scaling=None):
    """Return an array of shape (number of lines, max_distance + 1).

    Row i holds Theta(p, p + d * direction) for d = 0, 1, ..., max_distance,
    where p and the direction are line i. Values are raw, not divided by d = 0.

    If scaling (from adam_scaling) is given, return two arrays (plain, adam):
    the Adam-weighted kernel uses the same gradients, so it costs almost nothing extra.
    For the diagonals and the far reference pixel, use measure_kernel_with_extras.
    """
    result = measure_kernel_with_extras(model, noise, lines, max_distance, scaling)
    if scaling is None:
        return result["plain"]
    return result["plain"], result["adam"]
