"""Tests for the EnvConfig + argument-less reset contract (LAB-84).

Pins the new model: (1) ``reset()`` takes no arguments and is a pure restore —
the same env returns to the same t=0 state every time; (2) the env is defined by
its ``EnvConfig`` — a different ``wall_seed`` is a different env (different holes),
the same ``wall_seed`` reproduces the same env; (3) reset hands the controller a
physically sane state. Per-episode variation now comes from *building a different
env*, never from a reset argument.

The wall-seed cases need the optional ``scenegen`` (CadQuery) extra to build
procedural walls, so they skip when it is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_teleop.sim.config import EnvConfig, episode_wall_seed
from ai_teleop.sim.scene import SimEnv

SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "full_scene.xml"


@pytest.fixture
def scene_path() -> str:
    if not SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {SCENE_PATH}")
    return str(SCENE_PATH)


def test_reset_is_argumentless_and_deterministic(scene_path):
    # Two resets of the same env land in exactly the same t=0 state.
    env = SimEnv(scene_path, render_mode="headless")
    obs_a = env.reset()
    env.step()  # perturb, then restore
    obs_b = env.reset()
    np.testing.assert_array_equal(obs_a.ee_pose, obs_b.ee_pose)
    np.testing.assert_array_equal(obs_a.peg_pose, obs_b.peg_pose)
    np.testing.assert_array_equal(obs_a.joint_positions, obs_b.joint_positions)


def test_default_config_is_static_wall(scene_path):
    # No config ⇒ a static-wall env that knows it has no wall seed.
    env = SimEnv(scene_path, render_mode="headless")
    assert env.config == EnvConfig()
    assert env.config.wall_seed is None


def test_reset_then_step_is_physically_stable(scene_path):
    # Stepping from the restored home a few ticks must not blow up (NaNs).
    env = SimEnv(scene_path, render_mode="headless")
    env.reset()
    for _ in range(20):
        env.step()
    obs = env.get_observation()
    assert np.all(np.isfinite(obs.ee_pose))
    assert np.all(np.isfinite(obs.peg_pose))


def test_episode_wall_seed_is_deterministic_and_distinct():
    # Reproducible per (master, index); distinct across indices and across masters.
    assert episode_wall_seed(0, 0) == episode_wall_seed(0, 0)
    assert episode_wall_seed(0, 0) != episode_wall_seed(0, 1)
    assert episode_wall_seed(0, 5) != episode_wall_seed(1, 5)


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_different_wall_seed_gives_a_different_wall():
    pytest.importorskip("cadquery", reason="generated walls need the scenegen extra")
    from ai_teleop.sim.env_setup import make_env

    env_a = make_env(EnvConfig(wall_seed=1))
    env_b = make_env(EnvConfig(wall_seed=2))
    try:
        holes_a = env_a.reset().hole_poses
        holes_b = env_b.reset().hole_poses
        # Different procedural walls ⇒ a different hole layout (count and/or pose).
        differs = holes_a.shape != holes_b.shape or not np.allclose(holes_a, holes_b)
        assert differs
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_same_wall_seed_reproduces_the_wall():
    pytest.importorskip("cadquery", reason="generated walls need the scenegen extra")
    from ai_teleop.sim.env_setup import make_env

    env_a = make_env(EnvConfig(wall_seed=7))
    env_b = make_env(EnvConfig(wall_seed=7))
    try:
        np.testing.assert_allclose(env_a.reset().hole_poses, env_b.reset().hole_poses)
        assert env_a.config.wall_seed == env_b.config.wall_seed == 7
    finally:
        env_a.close()
        env_b.close()
