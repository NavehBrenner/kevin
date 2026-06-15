"""Unit tests for ScriptedNoisyHuman input strategy."""

from __future__ import annotations

import numpy as np
import pytest

from ai_teleop.common.observation import Observation
from ai_teleop.domain import InputStrategy
from ai_teleop.input import ScriptedNoisyHuman


def _make_target_pose(
    position: np.ndarray | None = None,
    quaternion: np.ndarray | None = None,
) -> np.ndarray:
    if position is None:
        position = np.array([0.5, 0.0, 0.3])
    if quaternion is None:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([position, quaternion])


def _make_observation() -> Observation:
    return Observation(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_pose=np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
        wrist_ft=np.zeros(6),
        peg_pose=np.zeros(7),
        hole_poses=np.zeros((1, 7)),
        target_hole_index=0,
        sim_time=0.0,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_scripted_noisy_human_satisfies_input_strategy_protocol():
    actor = ScriptedNoisyHuman(_make_target_pose())
    assert isinstance(actor, InputStrategy)


# ---------------------------------------------------------------------------
# Target tracking: mean command position ≈ target position
# ---------------------------------------------------------------------------


def test_mean_command_position_tracks_target():
    target_pose = _make_target_pose(position=np.array([0.6, 0.1, 0.4]))
    actor = ScriptedNoisyHuman(target_pose, position_noise_std=0.005, seed=42)
    obs = _make_observation()

    positions = np.array([actor.get_command(obs).target_position for _ in range(500)])
    np.testing.assert_allclose(positions.mean(axis=0), target_pose[:3], atol=1e-2)


def test_command_quaternion_is_unit_norm():
    actor = ScriptedNoisyHuman(_make_target_pose(), seed=0)
    obs = _make_observation()
    for _ in range(20):
        cmd = actor.get_command(obs)
        np.testing.assert_allclose(np.linalg.norm(cmd.target_quaternion), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Reproducibility: same seed → identical command sequence
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_commands():
    target_pose = _make_target_pose()
    obs = _make_observation()

    actor_a = ScriptedNoisyHuman(target_pose, seed=7)
    actor_b = ScriptedNoisyHuman(target_pose, seed=7)

    for _ in range(10):
        cmd_a = actor_a.get_command(obs)
        cmd_b = actor_b.get_command(obs)
        np.testing.assert_array_equal(cmd_a.target_position, cmd_b.target_position)
        np.testing.assert_array_equal(cmd_a.target_quaternion, cmd_b.target_quaternion)


def test_different_seeds_produce_different_commands():
    target_pose = _make_target_pose()
    obs = _make_observation()

    actor_a = ScriptedNoisyHuman(target_pose, seed=1)
    actor_b = ScriptedNoisyHuman(target_pose, seed=2)

    positions_a = [actor_a.get_command(obs).target_position for _ in range(5)]
    positions_b = [actor_b.get_command(obs).target_position for _ in range(5)]
    assert not all(np.array_equal(a, b) for a, b in zip(positions_a, positions_b, strict=True))


# ---------------------------------------------------------------------------
# Noise is actually applied
# ---------------------------------------------------------------------------


def test_position_noise_std_is_respected():
    std = 0.01
    target_pose = _make_target_pose()
    actor = ScriptedNoisyHuman(target_pose, position_noise_std=std, seed=0)
    obs = _make_observation()

    positions = np.array([actor.get_command(obs).target_position for _ in range(2000)])
    # Each axis independently: sample std should be close to the configured std.
    np.testing.assert_allclose(positions.std(axis=0), [std, std, std], atol=3e-3)


def test_zero_noise_command_equals_target():
    target_pose = _make_target_pose()
    actor = ScriptedNoisyHuman(
        target_pose, position_noise_std=0.0, orientation_noise_std=0.0, seed=0
    )
    obs = _make_observation()

    cmd = actor.get_command(obs)
    np.testing.assert_allclose(cmd.target_position, target_pose[:3], atol=1e-12)
    np.testing.assert_allclose(cmd.target_quaternion, target_pose[3:], atol=1e-9)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_wrong_target_pose_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        ScriptedNoisyHuman(np.zeros(6))
