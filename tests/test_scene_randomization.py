"""Tests for per-episode coverage randomization in SimEnv (M4 / LAB-40).

Pins three properties: (1) randomize=False reproduces the deterministic home
pose M1–M3 relied on; (2) randomize=True varies the start across episode
indices but reproduces a given index exactly; (3) the peg weld stays satisfied
at t=0 after randomization (no integrator transient).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from ai_teleop.sim.scene import SimEnv

SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "full_scene.xml"


@pytest.fixture
def scene_path() -> str:
    if not SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {SCENE_PATH}")
    return str(SCENE_PATH)


def _peg_in_hand(env: SimEnv) -> tuple[np.ndarray, np.ndarray]:
    """Peg pose expressed in the hand frame — the quantity the weld fixes."""
    data = env.data
    hand_position = data.xpos[env._hand_body_id].copy()
    hand_quaternion = data.xquat[env._hand_body_id].copy()
    peg_position = data.xpos[env._peg_body_id].copy()
    peg_quaternion = data.xquat[env._peg_body_id].copy()

    hand_quaternion_inv = np.zeros(4)
    mujoco.mju_negQuat(hand_quaternion_inv, hand_quaternion)
    relative_position = np.zeros(3)
    mujoco.mju_rotVecQuat(relative_position, peg_position - hand_position, hand_quaternion_inv)
    relative_quaternion = np.zeros(4)
    mujoco.mju_mulQuat(relative_quaternion, hand_quaternion_inv, peg_quaternion)
    return relative_position, relative_quaternion


def test_default_reset_is_deterministic_home(scene_path):
    # randomize=False: reset is the home pose, and an episode index is ignored.
    env = SimEnv(scene_path, render_mode="headless")
    obs_a = env.reset()
    obs_b = env.reset(episode_index=5)
    np.testing.assert_array_equal(obs_a.ee_pose, obs_b.ee_pose)
    np.testing.assert_array_equal(obs_a.peg_pose, obs_b.peg_pose)
    assert obs_a.target_hole_index == obs_b.target_hole_index


def test_randomized_reset_varies_across_episodes(scene_path):
    env = SimEnv(scene_path, render_mode="headless", randomize=True)
    obs_0 = env.reset(episode_index=0)
    obs_1 = env.reset(episode_index=1)
    # Different episodes differ in target hole and/or peg start pose.
    differs = (obs_0.target_hole_index != obs_1.target_hole_index) or not np.allclose(
        obs_0.peg_pose, obs_1.peg_pose
    )
    assert differs


def test_randomized_reset_is_reproducible(scene_path):
    env = SimEnv(scene_path, render_mode="headless", randomize=True)
    obs_a = env.reset(episode_index=7)
    obs_other = env.reset(episode_index=3)  # noqa: F841 — perturb internal state
    obs_b = env.reset(episode_index=7)
    np.testing.assert_array_equal(obs_a.ee_pose, obs_b.ee_pose)
    np.testing.assert_array_equal(obs_a.peg_pose, obs_b.peg_pose)
    assert obs_a.target_hole_index == obs_b.target_hole_index


def test_joint_offset_actually_moves_the_arm(scene_path):
    home = SimEnv(scene_path, render_mode="headless").reset()
    env = SimEnv(scene_path, render_mode="headless", randomize=True, randomize_target_hole=False)
    randomized = env.reset(episode_index=2)
    assert not np.allclose(home.joint_positions, randomized.joint_positions)


def test_weld_preserved_after_randomization(scene_path):
    # The peg-in-hand transform after a randomized reset must match the home
    # transform — that is what keeps the weld satisfied at t=0.
    home_env = SimEnv(scene_path, render_mode="headless")
    home_env.reset()
    home_relative_position, home_relative_quaternion = _peg_in_hand(home_env)

    env = SimEnv(scene_path, render_mode="headless", randomize=True)
    env.reset(episode_index=4)
    relative_position, relative_quaternion = _peg_in_hand(env)

    np.testing.assert_allclose(relative_position, home_relative_position, atol=1e-6)
    # Quaternion equality up to sign.
    dot = abs(float(np.dot(relative_quaternion, home_relative_quaternion)))
    np.testing.assert_allclose(dot, 1.0, atol=1e-6)


def test_randomized_start_is_physically_stable(scene_path):
    # Stepping the randomized start a few ticks must not blow up (NaNs) — the
    # weld-preserving reset should hand the controller a sane state.
    env = SimEnv(scene_path, render_mode="headless", randomize=True)
    env.reset(episode_index=1)
    for _ in range(20):
        env.step()
    obs = env.get_observation()
    assert np.all(np.isfinite(obs.ee_pose))
    assert np.all(np.isfinite(obs.peg_pose))
