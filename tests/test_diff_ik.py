"""Sanity tests for `ai_teleop.control.diff_ik.differential_ik`.

These are regression-detection tests — we want to catch sign flips,
shape errors, and order-of-operations bugs, not assert numerical purity.
Tolerances are generous on purpose.

The Jacobian fixture is a 6×7 identity-padded matrix: the first six joints
control the six task-space DoF directly, and the seventh is purely
null-space. That gives every assertion a closed-form expected value
without needing to spin up a MuJoCo model.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_teleop.control.diff_ik import differential_ik

# 6×7 identity-padded Jacobian: joints 1..6 each drive one EE DoF,
# joint 7 is pure null-space.
J_ID = np.hstack([np.eye(6), np.zeros((6, 1))])
DT = 0.01
SMALL_DAMPING = 1e-3  # so DLS ≈ true pinv


def _identity_quat() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def test_zero_pose_error_drives_only_null_space():
    """Target == current and q != q_nominal: EE motion should be zero."""
    q = np.array([0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.5])
    q_nominal = np.zeros(7)
    ee_pos = np.array([0.5, 0.0, 0.4])

    qdot = differential_ik(
        q=q,
        ee_pos=ee_pos,
        ee_quat=_identity_quat(),
        target_pos=ee_pos.copy(),
        target_quat=_identity_quat(),
        jacobian=J_ID,
        dt=DT,
        damping=SMALL_DAMPING,
        q_nominal=q_nominal,
        posture_gain=1.0,
    )

    # J @ qdot = qdot[:6], which is the EE velocity. Should be ~0.
    ee_vel = J_ID @ qdot
    assert np.allclose(ee_vel, 0.0, atol=1e-6), f"EE velocity not zero: {ee_vel}"

    # Joint 7 is the null-space; posture term should drive it toward q_nominal[6] = 0.
    # Direction check: qdot[6] should have opposite sign to q[6] (i.e., be negative here).
    assert qdot[6] < 0, f"posture term should reduce q[6]={q[6]}, got qdot[6]={qdot[6]}"


def test_position_error_produces_matching_ee_velocity():
    """A pure +x position offset should yield ~+x EE velocity equal to e/dt."""
    q = np.zeros(7)
    target_pos = np.array([0.51, 0.0, 0.4])
    ee_pos = np.array([0.50, 0.0, 0.4])

    qdot = differential_ik(
        q=q,
        ee_pos=ee_pos,
        ee_quat=_identity_quat(),
        target_pos=target_pos,
        target_quat=_identity_quat(),
        jacobian=J_ID,
        dt=DT,
        damping=SMALL_DAMPING,
        q_nominal=q.copy(),  # so the posture term is zero
        posture_gain=1.0,
    )

    ee_vel = J_ID @ qdot
    expected = np.array([0.01 / DT, 0.0, 0.0, 0.0, 0.0, 0.0])
    # DLS damping bleeds a few percent off the magnitude; allow 5% slack.
    assert np.allclose(ee_vel, expected, atol=0.05 * expected.max()), (
        f"EE velocity {ee_vel} not aligned with expected {expected}"
    )


def test_orientation_error_produces_angular_velocity():
    """A small rotation about z in the target should yield ω_z > 0 only."""
    q = np.zeros(7)
    ee_pos = np.array([0.5, 0.0, 0.4])

    theta = 0.05  # ~3°
    half = theta / 2.0
    target_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    qdot = differential_ik(
        q=q,
        ee_pos=ee_pos,
        ee_quat=_identity_quat(),
        target_pos=ee_pos.copy(),
        target_quat=target_quat,
        jacobian=J_ID,
        dt=DT,
        damping=SMALL_DAMPING,
        q_nominal=q.copy(),
        posture_gain=1.0,
    )

    ee_vel = J_ID @ qdot
    linear = ee_vel[:3]
    angular = ee_vel[3:]
    assert np.allclose(linear, 0.0, atol=1e-6), f"unexpected linear motion: {linear}"
    assert angular[2] > 0, f"expected positive ω_z, got {angular}"
    assert abs(angular[0]) < 1e-6 and abs(angular[1]) < 1e-6, (
        f"unexpected off-axis angular components: {angular}"
    )


def test_invalid_shapes_raise():
    """Defensive shape checks fail loudly."""
    q = np.zeros(7)
    ee_pos = np.zeros(3)
    eq = _identity_quat()
    with pytest.raises(ValueError):
        differential_ik(
            q=np.zeros(5),
            ee_pos=ee_pos, ee_quat=eq, target_pos=ee_pos, target_quat=eq,
            jacobian=J_ID, dt=DT, damping=SMALL_DAMPING,
            q_nominal=q, posture_gain=1.0,
        )
    with pytest.raises(ValueError):
        differential_ik(
            q=q,
            ee_pos=ee_pos, ee_quat=eq, target_pos=ee_pos, target_quat=eq,
            jacobian=np.zeros((6, 6)), dt=DT, damping=SMALL_DAMPING,
            q_nominal=q, posture_gain=1.0,
        )
    with pytest.raises(ValueError):
        differential_ik(
            q=q,
            ee_pos=ee_pos, ee_quat=eq, target_pos=ee_pos, target_quat=eq,
            jacobian=J_ID, dt=0.0, damping=SMALL_DAMPING,
            q_nominal=q, posture_gain=1.0,
        )
