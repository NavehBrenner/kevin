"""M3 acceptance: end-to-end runner + dummy-Δ seam injection (LAB-25).

Two checks, both driving the real composed loop (`run_episode`) headless:

1. `test_no_assist_runs_end_to_end` — the no-assist stack runs a full
   episode-length loop through the seam without intervention; the lock stays
   sane and the EE tracks the noisy scripted target toward the hole. This is
   plumbing, not insertion performance — the tolerance is deliberately loose.

2. `test_dummy_delta_reaches_controller_through_seam` — the key acceptance.
   A throwaway `_FixedDelta` provider is swapped in for `NoAssist` with **no
   edit** to `ScriptedNoisyHuman` or `Controller`, and we assert the controller
   received exactly `apply_delta(base_command, Δ)` — proving the seam accepts an
   injected Δ from any source (dependency inversion). Recording wrappers spy on
   the base command and the command handed to `compute`; the wrapped objects are
   the unmodified production classes.

`run_episode` lives in the package (`ai_teleop.sim.runner`); the
`scripts/run_episode.py` CLI is just a thin front door over it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_teleop.control import Controller, LockState
from ai_teleop.domain import ZERO_DELTA, Delta, NoAssist, apply_delta
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.runner import EpisodeResult, run_episode
from ai_teleop.sim.scene import SimEnv

SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "full_scene.xml"

E2E_STEPS = 400  # ~0.8 s — enough free-space approach to register clear progress
SEAM_STEPS = 120  # the seam check is about composition, not dynamics; keep it short


@pytest.fixture(scope="module")
def env():
    if not SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {SCENE_PATH}")
    e = SimEnv(str(SCENE_PATH), render_mode="headless")
    yield e
    e.close()


def _target_pose_at_hole(env_, controller) -> np.ndarray:
    """(7,) EE target: active hole position + the controller's home orientation."""
    obs = env_.reset()
    hole_position = obs.hole_poses[0][:3].copy()  # task goal: hole_0
    home_quat = controller.home_pose[3:].copy()
    return np.concatenate([hole_position, home_quat])


def _assert_commands_equal(actual, expected) -> None:
    np.testing.assert_allclose(actual.target_position, expected.target_position, atol=1e-9)
    np.testing.assert_allclose(actual.target_quaternion, expected.target_quaternion, atol=1e-9)
    assert actual.delta_grip_force == pytest.approx(expected.delta_grip_force)


# ---------------------------------------------------------------------------
# Recording wrappers — spy on the seam boundaries without touching production
# classes. Each delegates to a real, unmodified object.
# ---------------------------------------------------------------------------


class _RecordingInput:
    """Wraps an InputStrategy, recording every base Command it emits."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.commands: list = []

    def get_command(self, observation):
        command = self._inner.get_command(observation)
        self.commands.append(command)
        return command


class _RecordingController:
    """Wraps a Controller, recording every Command handed to compute()."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.commands: list = []

    def compute(self, observation, command) -> None:
        self.commands.append(command)
        self._inner.compute(observation, command)

    @property
    def status(self):
        return self._inner.status


class _FixedDelta:
    """Throwaway AssistProvider returning a fixed non-zero Δ every tick."""

    def __init__(self, delta: Delta) -> None:
        self._delta = delta

    def get_delta(self, observation, command) -> Delta:
        return self._delta


# ---------------------------------------------------------------------------
# 1. No-assist end-to-end
# ---------------------------------------------------------------------------


def test_no_assist_runs_end_to_end(env):
    """The no-assist loop runs a full episode and tracks toward the hole."""
    controller = Controller(env)
    target_pose = _target_pose_at_hole(env, controller)
    target_position = target_pose[:3]
    human = ScriptedNoisyHuman(target_pose, seed=0)

    start_obs = env.reset()
    start_dist = float(np.linalg.norm(start_obs.ee_pose[:3] - target_position))

    result = run_episode(env, controller, human, NoAssist(), max_steps=E2E_STEPS, render=False)

    assert isinstance(result, EpisodeResult)
    assert result.n_steps == E2E_STEPS
    # The --profile probe is wired: per-phase timings accumulate, and render_count
    # stays 0 with render=False (the viewer never syncs headless).
    assert sum(result.step_timings.values()) > 0.0
    assert result.step_timings["step"] > 0.0
    assert result.render_count == 0
    # Lock stays in a sane state — free tracking (ACTIVE) or a clean force-cap
    # hold if contact happened; never an undefined state.
    assert result.lock_status.state in (LockState.ACTIVE, LockState.HOLD)
    final_dist = float(np.linalg.norm(result.final_observation.ee_pose[:3] - target_position))
    assert final_dist < start_dist - 5e-3, (
        f"EE did not move toward hole: {start_dist * 1000:.1f} mm -> {final_dist * 1000:.1f} mm"
    )


# ---------------------------------------------------------------------------
# 2. Dummy-Δ seam injection (the M3 acceptance)
# ---------------------------------------------------------------------------


def test_dummy_delta_reaches_controller_through_seam(env):
    """A dummy Δ source swaps in for NoAssist; the controller receives base+Δ.

    Same `run_episode`, same `ScriptedNoisyHuman`, same `Controller` — only the
    `assist` argument changes. Proves the seam's dependency inversion: a Δ from
    any source reaches the controller with no upstream/downstream edits.
    """
    target_pose = _target_pose_at_hole(env, Controller(env))

    # --- no-assist: the controller receives the base command unchanged ---
    rec_input_na = _RecordingInput(ScriptedNoisyHuman(target_pose, seed=3))
    rec_ctrl_na = _RecordingController(Controller(env))
    run_episode(env, rec_ctrl_na, rec_input_na, NoAssist(), max_steps=SEAM_STEPS, render=False)

    assert len(rec_ctrl_na.commands) == SEAM_STEPS
    for base, received in zip(rec_input_na.commands, rec_ctrl_na.commands, strict=True):
        _assert_commands_equal(received, apply_delta(base, ZERO_DELTA))

    # --- dummy Δ swapped in, nothing else changed ---
    fixed_delta = Delta(
        delta_position=np.array([0.01, -0.005, 0.0]),
        delta_orientation=np.array([0.0, 0.0, np.deg2rad(3.0)]),
        delta_grip_force=1.0,
    )
    rec_input_dd = _RecordingInput(ScriptedNoisyHuman(target_pose, seed=3))
    rec_ctrl_dd = _RecordingController(Controller(env))
    run_episode(
        env,
        rec_ctrl_dd,
        rec_input_dd,
        _FixedDelta(fixed_delta),
        max_steps=SEAM_STEPS,
        render=False,
    )

    assert len(rec_ctrl_dd.commands) == SEAM_STEPS
    for base, received in zip(rec_input_dd.commands, rec_ctrl_dd.commands, strict=True):
        _assert_commands_equal(received, apply_delta(base, fixed_delta))
        # The injected Δ genuinely altered the command — it wasn't dropped.
        assert not np.allclose(received.target_position, base.target_position)
