"""Tests for the BC loss (LAB-34) — rotation math + masked per-channel reduction.

The orientation channel is the subtle one (axis-angle is non-Euclidean), so the
rotation helpers get their own known-value checks; the loss itself is verified for
the equal-inputs floor, padding-mask invariance, channel weighting, and finite
gradients at the θ = 0 singularity the Rodrigues map is written to survive.
"""

from __future__ import annotations

import math

import torch

from ai_teleop.policy.losses import (
    LossConfig,
    axis_angle_to_matrix,
    geodesic_angle,
    residual_bc_loss,
)

# ---------------------------------------------------------------------------
# axis_angle_to_matrix — Rodrigues
# ---------------------------------------------------------------------------


def test_zero_axis_angle_is_identity():
    matrix = axis_angle_to_matrix(torch.zeros(3))
    torch.testing.assert_close(matrix, torch.eye(3))


def test_quarter_turn_about_z_is_known_matrix():
    matrix = axis_angle_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2]))
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(matrix, expected, atol=1e-6, rtol=1e-6)


def test_result_is_a_rotation_matrix():
    """R Rᵀ = I and det = 1 for a batch of random axis-angles."""
    axis_angle = torch.randn(8, 3)
    matrix = axis_angle_to_matrix(axis_angle)
    identity = matrix @ matrix.transpose(-1, -2)
    torch.testing.assert_close(identity, torch.eye(3).expand(8, 3, 3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(8), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# geodesic_angle
# ---------------------------------------------------------------------------


def test_geodesic_angle_zero_for_equal_rotations():
    axis_angle = torch.tensor([0.1, -0.2, 0.05])
    angle = geodesic_angle(axis_angle, axis_angle)
    assert float(angle) < 2e-3  # arccos-clamp floor (~0.0014 rad ≈ 0.08°), not exactly 0


def test_geodesic_angle_matches_known_separation():
    identity = torch.zeros(3)
    quarter_turn = torch.tensor([0.0, 0.0, math.pi / 2])
    angle = geodesic_angle(quarter_turn, identity)
    assert abs(float(angle) - math.pi / 2) < 1e-4


# ---------------------------------------------------------------------------
# residual_bc_loss
# ---------------------------------------------------------------------------


def _delta(batch: int, steps: int, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, steps, 7, generator=generator)


def test_loss_is_near_zero_for_identical_prediction_geodesic():
    target = _delta(2, 4)
    mask = torch.ones(2, 4)
    loss = residual_bc_loss(target, target, mask)
    assert float(loss) < 1e-3  # geodesic floor; position/grip terms are exactly 0


def test_loss_is_exactly_zero_for_identical_prediction_mse():
    target = _delta(2, 4)
    mask = torch.ones(2, 4)
    loss = residual_bc_loss(target, target, mask, config=LossConfig(orientation="mse"))
    assert float(loss) == 0.0


def test_padding_steps_do_not_affect_loss():
    """A masked-out (padding) step with garbage values must not change the loss."""
    predicted = _delta(1, 2, seed=1)
    target = _delta(1, 2, seed=2)

    # Append a third, padded step full of garbage; mask it out.
    padded_predicted = torch.cat([predicted, 1e3 * torch.ones(1, 1, 7)], dim=1)
    padded_target = torch.cat([target, -1e3 * torch.ones(1, 1, 7)], dim=1)
    padded_mask = torch.tensor([[1.0, 1.0, 0.0]])

    full = residual_bc_loss(predicted, target, torch.ones(1, 2))
    masked = residual_bc_loss(padded_predicted, padded_target, padded_mask)
    torch.testing.assert_close(masked, full)


def test_channel_weights_scale_their_term():
    predicted = torch.zeros(1, 1, 7)
    target = torch.zeros(1, 1, 7)
    target[..., 0] = 0.5  # position error only
    mask = torch.ones(1, 1)

    base = residual_bc_loss(predicted, target, mask, config=LossConfig(weight_position=1.0))
    doubled = residual_bc_loss(predicted, target, mask, config=LossConfig(weight_position=2.0))
    torch.testing.assert_close(doubled, 2.0 * base)


def test_gradient_is_finite_at_zero_rotation():
    """The Rodrigues θ→0 singularity must not produce NaN/Inf gradients."""
    predicted = torch.zeros(1, 1, 7, requires_grad=True)
    target = _delta(1, 1)
    loss = residual_bc_loss(predicted, target, torch.ones(1, 1))
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
