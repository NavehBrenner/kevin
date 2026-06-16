"""Unit tests for the analytical privileged-info expert (M4 / LAB-27).

The headline property is *far-field zero by construction* — the distance gate
makes ``Δ* == 0`` whenever the tip is at or beyond ``d_far`` from the hole. The
rest pin the seam conformance and that the geometric corrections point the right
way (lateral shift reduces lateral error; the orientation Δ rotates the peg axis
toward the bore through the seam's ``apply_delta``).
"""

from __future__ import annotations

import mujoco
import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation
from ai_teleop.common.utils.rotations import axis_from_quat
from ai_teleop.domain import AssistProvider, apply_delta
from ai_teleop.expert import Expert

# Scene-consistent conventions used to build synthetic observations: hole bore
# along world +x (identity-oriented site), peg long axis along its local +z.
_HOLE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # identity ⇒ bore = world +x
_INSERTION_AXIS = np.array([1.0, 0.0, 0.0])
_DEFAULT_HOLE_POSITION = np.array([0.79, 0.0, 0.45])
_PEG_HALF_LENGTH = 0.030


def _peg_quat_pointing_x() -> np.ndarray:
    # Rotate local +z onto world +x: 90° about world -y.
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, -1.0, 0.0]), np.pi / 2)
    return quat


def _make_observation(
    *,
    tip_position: np.ndarray,
    peg_quat: np.ndarray | None = None,
    hole_position: np.ndarray = _DEFAULT_HOLE_POSITION,
    wrist_ft: np.ndarray | None = None,
) -> Observation:
    if peg_quat is None:
        peg_quat = _peg_quat_pointing_x()
    # Body origin = tip - half_length * a, where a = R(peg_quat) @ z.
    peg_axis = axis_from_quat(peg_quat, 2)
    body_position = tip_position - _PEG_HALF_LENGTH * peg_axis
    return Observation(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_pose=np.concatenate([body_position, peg_quat]),
        wrist_ft=np.zeros(6) if wrist_ft is None else wrist_ft,
        gripper_width=0.08,
        peg_pose=np.concatenate([body_position, peg_quat]),
        hole_poses=np.concatenate([hole_position, _HOLE_QUAT]).reshape(1, 7),
        target_hole_index=0,
        sim_time=0.0,
    )


def _dummy_command() -> Command:
    return Command(np.array([0.5, 0.0, 0.45]), np.array([1.0, 0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Seam conformance
# ---------------------------------------------------------------------------


def test_expert_satisfies_assist_provider_protocol():
    assert isinstance(Expert(), AssistProvider)


# ---------------------------------------------------------------------------
# Far-field zero BY CONSTRUCTION (the headline property)
# ---------------------------------------------------------------------------


def test_delta_is_zero_far_from_hole():
    expert = Expert(d_far=0.08)
    hole = np.array([0.79, 0.0, 0.45])
    # A grid of tip positions all strictly beyond d_far from the hole.
    for dx in (0.10, 0.15, 0.3):
        for lateral in (np.zeros(3), np.array([0.0, 0.05, 0.0]), np.array([0.0, 0.0, -0.1])):
            tip = hole - np.array([dx, 0.0, 0.0]) + lateral
            assert np.linalg.norm(hole - tip) >= 0.08
            delta = expert.get_delta(_make_observation(tip_position=tip), _dummy_command())
            np.testing.assert_array_equal(delta.delta_position, np.zeros(3))
            np.testing.assert_array_equal(delta.delta_orientation, np.zeros(3))
            assert delta.delta_grip_force == 0.0


def test_delta_is_nonzero_near_hole():
    expert = Expert(d_near=0.01, d_far=0.08)
    hole = np.array([0.79, 0.0, 0.45])
    # Tip 2 cm short of the hole, 1 cm laterally off ⇒ inside the gate band.
    tip = hole - np.array([0.02, 0.0, 0.0]) + np.array([0.0, 0.01, 0.0])
    delta = expert.get_delta(_make_observation(tip_position=tip), _dummy_command())
    assert np.linalg.norm(delta.delta_position) > 0.0


# ---------------------------------------------------------------------------
# Lateral correction points toward the hole axis
# ---------------------------------------------------------------------------


def test_lateral_delta_reduces_lateral_error():
    expert = Expert(d_near=0.05, d_far=0.2)  # full gate at the test distance
    hole = np.array([0.79, 0.0, 0.45])
    lateral_offset = np.array([0.0, 0.01, 0.0])  # 1 cm off-axis in +y
    tip = hole - np.array([0.03, 0.0, 0.0]) + lateral_offset
    delta = expert.get_delta(_make_observation(tip_position=tip), _dummy_command())
    # The commanded shift must have a -y component to cancel the +y offset.
    assert delta.delta_position[1] < 0.0


# ---------------------------------------------------------------------------
# Orientation correction rotates the peg axis toward the bore
# ---------------------------------------------------------------------------


def test_orientation_delta_rotates_peg_axis_toward_bore():
    expert = Expert(d_near=0.05, d_far=0.2)
    hole = np.array([0.79, 0.0, 0.45])
    # Peg tilted ~12° off the bore.
    tilt = np.zeros(4)
    mujoco.mju_axisAngle2Quat(tilt, np.array([0.0, -1.0, 0.0]), np.pi / 2 + np.deg2rad(12.0))
    obs = _make_observation(tip_position=hole - np.array([0.03, 0.0, 0.0]), peg_quat=tilt)

    peg_axis_before = axis_from_quat(tilt, 2)
    angle_before = np.arccos(np.clip(peg_axis_before @ _INSERTION_AXIS, -1.0, 1.0))

    delta = expert.get_delta(obs, _dummy_command())
    # Apply the orientation Δ through the seam, then rotate the peg axis by it.
    command = Command(np.zeros(3), tilt)
    rotated_command = apply_delta(command, delta)
    peg_axis_after = axis_from_quat(rotated_command.target_quaternion, 2)
    angle_after = np.arccos(np.clip(peg_axis_after @ _INSERTION_AXIS, -1.0, 1.0))

    assert angle_after < angle_before


# ---------------------------------------------------------------------------
# Output stays within the residual-interface clamp
# ---------------------------------------------------------------------------


def test_delta_respects_clamp_bounds():
    expert = Expert(d_near=0.05, d_far=0.5)
    hole = np.array([0.79, 0.0, 0.45])
    # A large lateral error would over-shoot the 2 cm clamp pre-clamp.
    tip = hole - np.array([0.03, 0.0, 0.0]) + np.array([0.0, 0.1, 0.0])
    delta = expert.get_delta(_make_observation(tip_position=tip), _dummy_command())
    assert np.linalg.norm(delta.delta_position) <= 0.02 + 1e-9
    assert np.linalg.norm(delta.delta_orientation) <= np.deg2rad(10.0) + 1e-9
    assert abs(delta.delta_grip_force) <= 5.0 + 1e-9
