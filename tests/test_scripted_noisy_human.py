"""Unit tests for the realistic structured-noise ScriptedNoisyHuman (M4).

These pin the *form* of the noise model (biased + drifting + held), not the
magnitudes (placeholders, calibrated post-baseline). The key properties:
per-episode constant bias, temporally-correlated drift (NOT per-step white
noise), and a refresh-and-hold command rate.
"""

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
        gripper_width=0.08,
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


def test_command_quaternion_is_unit_norm():
    actor = ScriptedNoisyHuman(_make_target_pose(), seed=0)
    obs = _make_observation()
    for _ in range(200):
        cmd = actor.get_command(obs)
        np.testing.assert_allclose(np.linalg.norm(cmd.target_quaternion), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Determinism / seeding
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_commands():
    target_pose = _make_target_pose()
    obs = _make_observation()
    actor_a = ScriptedNoisyHuman(target_pose, seed=7)
    actor_b = ScriptedNoisyHuman(target_pose, seed=7)

    # Many ticks so the comparison crosses several refresh boundaries.
    for _ in range(300):
        cmd_a = actor_a.get_command(obs)
        cmd_b = actor_b.get_command(obs)
        np.testing.assert_array_equal(cmd_a.target_position, cmd_b.target_position)
        np.testing.assert_array_equal(cmd_a.target_quaternion, cmd_b.target_quaternion)


def test_different_seeds_produce_different_commands():
    target_pose = _make_target_pose()
    obs = _make_observation()
    actor_a = ScriptedNoisyHuman(target_pose, seed=1)
    actor_b = ScriptedNoisyHuman(target_pose, seed=2)

    positions_a = np.array([actor_a.get_command(obs).target_position for _ in range(100)])
    positions_b = np.array([actor_b.get_command(obs).target_position for _ in range(100)])
    assert not np.allclose(positions_a, positions_b)


# ---------------------------------------------------------------------------
# Per-episode bias: constant within an episode, varies across seeds
# ---------------------------------------------------------------------------


def test_bias_is_constant_within_episode():
    # With drift and tremor disabled, every command must equal goal = target +
    # bias exactly, across the whole episode (bias never resamples per step).
    target_pose = _make_target_pose()
    actor = ScriptedNoisyHuman(
        target_pose,
        drift_position_std=0.0,
        drift_orientation_std=0.0,
        tremor_std=0.0,
        seed=3,
    )
    obs = _make_observation()
    expected_position = target_pose[:3] + actor.position_bias

    commands = np.array([actor.get_command(obs).target_position for _ in range(500)])
    for command_position in commands:
        np.testing.assert_allclose(command_position, expected_position, atol=1e-12)


def test_bias_varies_across_seeds():
    target_pose = _make_target_pose()
    actor_a = ScriptedNoisyHuman(target_pose, seed=10)
    actor_b = ScriptedNoisyHuman(target_pose, seed=11)
    assert not np.allclose(actor_a.position_bias, actor_b.position_bias)
    assert not np.allclose(actor_a.orientation_bias, actor_b.orientation_bias)


def test_zero_noise_command_equals_target():
    # No bias, no drift, no tremor ⇒ the command is exactly the target forever.
    target_pose = _make_target_pose()
    actor = ScriptedNoisyHuman(
        target_pose,
        position_bias_std=0.0,
        orientation_bias_std=0.0,
        drift_position_std=0.0,
        drift_orientation_std=0.0,
        tremor_std=0.0,
        seed=0,
    )
    obs = _make_observation()
    for _ in range(50):
        cmd = actor.get_command(obs)
        np.testing.assert_allclose(cmd.target_position, target_pose[:3], atol=1e-12)
        np.testing.assert_allclose(cmd.target_quaternion, target_pose[3:], atol=1e-9)


# ---------------------------------------------------------------------------
# Refresh-and-hold: low-frequency command, not per-step jitter
# ---------------------------------------------------------------------------


def test_command_is_held_between_refreshes():
    # control_hz / refresh_hz = 500 / 10 = 50 ⇒ the command is constant for 50
    # ticks, then changes (drift on, tremor off).
    target_pose = _make_target_pose()
    actor = ScriptedNoisyHuman(
        target_pose,
        drift_position_std=0.01,
        tremor_std=0.0,
        refresh_hz=10.0,
        control_hz=500.0,
        seed=5,
    )
    obs = _make_observation()
    commands = np.array([actor.get_command(obs).target_position for _ in range(60)])

    # First 50 ticks identical (one hold window)...
    for command_position in commands[:50]:
        np.testing.assert_array_equal(command_position, commands[0])
    # ...then the next refresh moves it.
    assert not np.array_equal(commands[50], commands[0])


# ---------------------------------------------------------------------------
# Drift is temporally correlated (the whole point — reject white noise)
# ---------------------------------------------------------------------------


def test_drift_is_temporally_correlated():
    # Sample the held target once per refresh window; the per-refresh series
    # should be strongly autocorrelated at lag 1 (OU), unlike white noise (~0).
    target_pose = _make_target_pose()
    refresh_hz, control_hz = 10.0, 500.0
    hold_steps = round(control_hz / refresh_hz)
    actor = ScriptedNoisyHuman(
        target_pose,
        position_bias_std=0.0,
        drift_position_std=0.01,
        drift_tau=0.5,
        tremor_std=0.0,
        refresh_hz=refresh_hz,
        control_hz=control_hz,
        seed=1,
    )
    obs = _make_observation()

    n_refreshes = 400
    samples = np.array([
        actor.get_command(obs).target_position[0] for _ in range(n_refreshes * hold_steps)
    ])[::hold_steps]
    deviations = samples - samples.mean()
    lag1 = float(np.corrcoef(deviations[:-1], deviations[1:])[0, 1])
    assert lag1 > 0.3, f"drift lag-1 autocorrelation too low ({lag1:.3f}) — looks like white noise"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_wrong_target_pose_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        ScriptedNoisyHuman(np.zeros(6))
