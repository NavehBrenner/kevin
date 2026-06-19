"""Behavioral-cloning loss for the Phase-1 residual (LAB-34).

The policy regresses the expert's per-step ``Δ`` = ``(Δposition ∈ ℝ³,
Δorientation ∈ ℝ³ axis-angle, Δgrip ∈ ℝ¹)``. The three channels live in
different units (m, rad, N) and matter differently, so the loss is **per-channel
weighted** rather than a single MSE over the 7-vector.

The orientation channel is the subtle one: ``Δorientation`` is an *axis-angle
rotation*, so a naive component-wise difference is wrong near the ±π wrap and
ignores that axis-angle is a non-Euclidean parameterization. The default
``orientation="geodesic"`` loss measures the **true rotation angle** between the
predicted and target rotations (``angle(R̂ · R*ᵀ)``) — the proper distance on
SO(3). ``orientation="mse"`` is the documented simpler fallback (smooth-L1 on the
raw axis-angle components), kept behind a flag because the spec lists the exact
rotation loss as a calibration knob (`docs/milestone-5-spec.md` *Known unknowns*).

All reductions are **masked**: training batches are zero-padded to the longest
episode (see ``data.collate_episodes``), so the loss is averaged only over real
steps via a ``(B, T)`` validity mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

OrientationLoss = Literal["geodesic", "mse"]

# Channel slices of the 7-vector Δ — position(3) + orientation(3) + grip(1).
_POS = slice(0, 3)
_ORI = slice(3, 6)
_GRIP = slice(6, 7)

# Smoothing floor so ||axis_angle|| and its gradient stay finite at θ = 0
# (torch.norm has a NaN gradient at the origin; the +eps under the sqrt avoids it).
_THETA_EPS = 1e-8


@dataclass(frozen=True)
class LossConfig:
    """Per-channel weights and the orientation-loss flavor for the BC loss.

    Weights are relative — only their ratios matter. Defaults lean on position
    (the dominant alignment signal) with a modest orientation term and a small
    grip term; calibrate against the validation curve.
    """

    weight_position: float = 1.0
    weight_orientation: float = 0.5
    weight_grip: float = 0.1
    orientation: OrientationLoss = "geodesic"
    huber_beta: float = 1.0


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Batched, gradient-stable Rodrigues: ``(..., 3)`` axis-angle → ``(..., 3, 3)``.

    Uses the rotation-*vector* form ``R = I + (sinθ/θ)·K + ((1-cosθ)/θ²)·K²`` with
    ``K = skew(axis_angle)``, so no axis normalization (and thus no 0/0) is needed;
    ``sinθ/θ → 1`` and ``(1-cosθ)/θ² → ½`` are finite at θ = 0, and θ is smoothed
    with ``_THETA_EPS`` so its gradient is finite at the origin too.
    """
    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)

    theta = torch.sqrt((axis_angle * axis_angle).sum(dim=-1) + _THETA_EPS)  # (...)
    sin_over_theta = (torch.sin(theta) / theta)[..., None, None]
    one_minus_cos_over_theta_sq = ((1.0 - torch.cos(theta)) / (theta * theta))[..., None, None]

    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return identity + sin_over_theta * skew + one_minus_cos_over_theta_sq * (skew @ skew)


def geodesic_angle(predicted_axis_angle: Tensor, target_axis_angle: Tensor) -> Tensor:
    """Rotation angle (rad) between the two axis-angle rotations: ``(..., 3) → (...)``.

    ``angle(R̂ · R*ᵀ)`` — the geodesic distance on SO(3). The ``arccos`` argument is
    clamped just inside ``[-1, 1]`` so the gradient stays finite at 0 and π.
    """
    relative = axis_angle_to_matrix(predicted_axis_angle) @ axis_angle_to_matrix(
        target_axis_angle
    ).transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(cosine)


def residual_bc_loss(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    config: LossConfig | None = None,
) -> Tensor:
    """Masked, per-channel, rotation-aware BC loss → scalar.

    Args:
        predicted: ``(B, T, 7)`` predicted Δ.
        target:    ``(B, T, 7)`` expert Δ (the BC target).
        mask:      ``(B, T)`` 1 for real steps, 0 for padding.
        config:    weights + orientation flavor (defaults to ``LossConfig()``).

    Position and grip use smooth-L1 (Huber); orientation uses the geodesic angle
    (default) or smooth-L1 on the raw axis-angle. Each is reduced to a per-step
    ``(B, T)`` scalar, weighted, summed, then averaged over the real steps only.
    """
    config = config or LossConfig()
    mask = mask.to(predicted.dtype)

    position_step = F.smooth_l1_loss(
        predicted[..., _POS], target[..., _POS], beta=config.huber_beta, reduction="none"
    ).mean(dim=-1)  # (B, T)
    grip_step = F.smooth_l1_loss(
        predicted[..., _GRIP], target[..., _GRIP], beta=config.huber_beta, reduction="none"
    ).mean(dim=-1)  # (B, T)

    if config.orientation == "geodesic":
        angle = geodesic_angle(predicted[..., _ORI], target[..., _ORI])  # (B, T)
        orientation_step = F.smooth_l1_loss(
            angle, torch.zeros_like(angle), beta=config.huber_beta, reduction="none"
        )
    else:  # "mse" fallback — smooth-L1 on raw axis-angle components
        orientation_step = F.smooth_l1_loss(
            predicted[..., _ORI], target[..., _ORI], beta=config.huber_beta, reduction="none"
        ).mean(dim=-1)

    per_step = (
        config.weight_position * position_step
        + config.weight_orientation * orientation_step
        + config.weight_grip * grip_step
    )  # (B, T)

    denominator = mask.sum().clamp_min(1.0)
    return (per_step * mask).sum() / denominator
