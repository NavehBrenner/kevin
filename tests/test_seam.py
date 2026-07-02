"""Unit tests for the M3 domain seam: Delta, apply_delta, NoAssist, protocols."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation
from ai_teleop.domain import (
    ZERO_DELTA,
    AssistProvider,
    Delta,
    NoAssist,
    apply_delta,
    clamp_delta,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_command(
    position: np.ndarray | None = None,
    quaternion: np.ndarray | None = None,
    grip_force: float = 0.0,
) -> Command:
    if position is None:
        position = np.array([0.5, 0.0, 0.5])
    if quaternion is None:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # identity (w, x, y, z)
    return Command(position, quaternion, grip_force)


def _make_observation() -> Observation:
    return Observation(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_pose=np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        wrist_ft=np.zeros(6),
        gripper_width=0.08,
        peg_pose=np.zeros(7),
        hole_poses=np.zeros((1, 7)),
        sim_time=0.0,
    )


# ---------------------------------------------------------------------------
# apply_delta identity: ZERO_DELTA round-trips the command
# ---------------------------------------------------------------------------


def test_apply_zero_delta_is_identity_simple():
    command = _make_command()
    result = apply_delta(command, ZERO_DELTA)
    np.testing.assert_allclose(result.target_position, command.target_position, atol=1e-9)
    np.testing.assert_allclose(result.target_quaternion, command.target_quaternion, atol=1e-9)
    assert abs(result.delta_grip_force - command.delta_grip_force) < 1e-12


def test_apply_zero_delta_is_identity_nontrivial():
    quat = np.array([0.707, 0.707, 0.0, 0.0])
    quat /= np.linalg.norm(quat)
    command = _make_command(
        position=np.array([0.3, 0.2, 0.6]),
        quaternion=quat,
        grip_force=2.5,
    )
    result = apply_delta(command, ZERO_DELTA)
    np.testing.assert_allclose(result.target_position, command.target_position, atol=1e-9)
    np.testing.assert_allclose(result.target_quaternion, command.target_quaternion, atol=1e-9)
    assert abs(result.delta_grip_force - command.delta_grip_force) < 1e-12


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def test_clamp_position_exceeds_bound_clamped_to_bound():
    large_delta = Delta(np.array([0.1, 0.0, 0.0]), np.zeros(3), 0.0)
    clamped = clamp_delta(large_delta)
    np.testing.assert_allclose(np.linalg.norm(clamped.delta_position), 0.02, atol=1e-12)


def test_clamp_position_within_bound_unchanged():
    small_delta = Delta(np.array([0.01, 0.0, 0.0]), np.zeros(3), 0.0)
    clamped = clamp_delta(small_delta)
    np.testing.assert_allclose(clamped.delta_position, small_delta.delta_position, atol=1e-12)


def test_clamp_orientation_exceeds_bound_clamped_to_bound():
    large_axis_angle = np.array([0.0, 0.0, np.deg2rad(30.0)])
    large_delta = Delta(np.zeros(3), large_axis_angle, 0.0)
    clamped = clamp_delta(large_delta)
    np.testing.assert_allclose(
        np.linalg.norm(clamped.delta_orientation),
        np.deg2rad(10.0),
        atol=1e-12,
    )


def test_clamp_grip_force_positive_clamped():
    delta = Delta(np.zeros(3), np.zeros(3), 10.0)
    assert clamp_delta(delta).delta_grip_force == pytest.approx(5.0)


def test_clamp_grip_force_negative_clamped():
    delta = Delta(np.zeros(3), np.zeros(3), -10.0)
    assert clamp_delta(delta).delta_grip_force == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# Quaternion composition
# ---------------------------------------------------------------------------


def test_apply_small_orientation_delta_result_is_unit_norm():
    command = _make_command()
    delta = Delta(np.zeros(3), np.array([0.0, 0.0, np.deg2rad(5.0)]), 0.0)
    result = apply_delta(command, delta)
    np.testing.assert_allclose(np.linalg.norm(result.target_quaternion), 1.0, atol=1e-9)


def test_apply_orientation_delta_clamped_to_10deg_on_identity():
    """90° rotation around z gets clamped to 10° then left-multiplied onto identity."""
    command = _make_command()  # identity quaternion
    delta = Delta(np.zeros(3), np.array([0.0, 0.0, np.deg2rad(90.0)]), 0.0)
    result = apply_delta(command, delta)
    half = np.deg2rad(10.0) / 2.0
    expected_quaternion = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    np.testing.assert_allclose(result.target_quaternion, expected_quaternion, atol=1e-9)


def test_apply_unclamped_orientation_delta_on_identity():
    """5° rotation around z (within bound) lands at exactly cos/sin(2.5°)."""
    command = _make_command()
    angle = np.deg2rad(5.0)
    delta = Delta(np.zeros(3), np.array([0.0, 0.0, angle]), 0.0)
    result = apply_delta(command, delta)
    half = angle / 2.0
    expected_quaternion = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    np.testing.assert_allclose(result.target_quaternion, expected_quaternion, atol=1e-9)


# ---------------------------------------------------------------------------
# NoAssist conformance
# ---------------------------------------------------------------------------


def test_no_assist_satisfies_assist_provider_protocol():
    assert isinstance(NoAssist(), AssistProvider)


def test_no_assist_get_delta_returns_zero_delta():
    provider = NoAssist()
    observation = _make_observation()
    command = _make_command()
    delta = provider.get_delta(observation, command)
    np.testing.assert_allclose(delta.delta_position, np.zeros(3))
    np.testing.assert_allclose(delta.delta_orientation, np.zeros(3))
    assert delta.delta_grip_force == 0.0


# ---------------------------------------------------------------------------
# domain/ has no ai_teleop.sim imports (pure-domain constraint)
# ---------------------------------------------------------------------------


def test_domain_has_no_sim_import():
    import ai_teleop.domain as domain_pkg

    domain_dir = pathlib.Path(domain_pkg.__file__).parent
    for python_file in domain_dir.rglob("*.py"):
        source = python_file.read_text(encoding="utf-8")
        import re

        sim_imports = re.findall(r"^\s*(import|from)\s+ai_teleop\.sim", source, re.MULTILINE)
        assert not sim_imports, f"sim import found in domain/{python_file.name}"
