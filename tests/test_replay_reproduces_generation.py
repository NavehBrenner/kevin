"""LAB-85: replaying a recorded episode reproduces it tick-for-tick.

The bug this guards: replay rebuilt the scene from CLI args (not the episode's own
stored spec), so the recorded commands drove a *different* physical episode. The fix
stamps the scene spec (``wall_seed``) into metadata and replays the recorded commands
verbatim (dumb iterator) in the reconstructed scene. Same wall_seed + same commands ⇒
the exact same trajectory — so the realized peg pose must match the recording to the
step. Everything is keyed on ``wall_seed`` + arg-less ``reset()`` (LAB-84); the old
``reset_episode_index`` / ``randomize`` reset model is gone.
"""

from __future__ import annotations

import numpy as np

from ai_teleop.common import Command
from ai_teleop.control import Controller
from ai_teleop.data.generate import generate_dataset
from ai_teleop.data.step_callbacks import episode_terminal_reason
from ai_teleop.data.trajectory import TerminalReason, load_episode
from ai_teleop.domain import Delta
from ai_teleop.sim.config import EnvConfig
from ai_teleop.sim.env_setup import make_env
from ai_teleop.sim.runner import run_episode


class _ReplayInput:
    """Dumb iterator over the recorded base commands (as run_episode's input)."""

    def __init__(self, columns):
        self._p, self._q, self._g, self._i = (
            columns["cmd_position"],
            columns["cmd_quaternion"],
            columns["cmd_grip"],
            0,
        )

    def get_command(self, observation):
        i = min(self._i, len(self._p) - 1)
        self._i += 1
        return Command(self._p[i].copy(), self._q[i].copy(), float(self._g[i]))


class _ReplayAssist:
    """Dumb iterator over the recorded deltas (as run_episode's assist)."""

    def __init__(self, columns):
        self._p, self._o, self._g, self._i = (
            columns["delta_position"],
            columns["delta_orientation"],
            columns["delta_grip"],
            0,
        )

    def get_delta(self, observation, command):
        i = min(self._i, len(self._p) - 1)
        self._i += 1
        return Delta(self._p[i].copy(), self._o[i].copy(), float(self._g[i]))


class _PegRecorder:
    """step_callback that captures the realized peg pose each tick."""

    def __init__(self):
        self.poses: list[np.ndarray] = []

    def __call__(self, step, observation, *_):
        self.poses.append(observation.peg_pose.copy())
        return False


def test_replay_reproduces_recorded_trajectory(tmp_path):
    paths = generate_dataset(tmp_path, n_episodes=2, seed=0, max_steps=300, baseline=False)

    for path in paths:
        columns, meta = load_episode(path)
        assert meta["target_hole_index"] == 0

        # Rebuild the exact scene from the stored wall_seed (the part that was broken —
        # it used to come from CLI args). Arg-less reset() lands on the same wall.
        env = make_env(EnvConfig(wall_seed=int(meta["wall_seed"])), render_mode="headless")
        controller = Controller(env, max_dpos_per_step=float(meta["max_dpos"]))

        # Replay the recorded commands + deltas verbatim; the realized peg trajectory
        # must match the recording tick-for-tick (would diverge if the scene were wrong).
        recorder = _PegRecorder()
        run_episode(
            env,
            controller,
            _ReplayInput(columns),
            _ReplayAssist(columns),
            max_steps=len(columns["step"]),
            step_callback=recorder,
        )
        np.testing.assert_allclose(np.array(recorder.poses), columns["peg_pose"], atol=1e-9)


def test_replay_is_faithful_under_finite_time_factor(tmp_path):
    """LAB-88: the loop is always physics-rate (one command per physics step), so replay
    reproduces the recording at any time_factor — the pacing/sleep path must not perturb
    physics. A large time_factor keeps the sleeps ~0 so the test stays fast; render=True
    (viewer) can't run in CI but only adds a sync that never touches sim data, so headless
    coverage carries the guarantee.
    """
    (path,) = generate_dataset(tmp_path, n_episodes=1, seed=0, max_steps=300, baseline=False)
    columns, meta = load_episode(path)

    env = make_env(EnvConfig(wall_seed=int(meta["wall_seed"])), render_mode="headless")
    controller = Controller(env, max_dpos_per_step=float(meta["max_dpos"]))
    recorder = _PegRecorder()
    run_episode(
        env,
        controller,
        _ReplayInput(columns),
        _ReplayAssist(columns),
        max_steps=len(columns["step"]),
        time_factor=1e6,  # finite → exercises the sleep path, but ~never actually sleeps
        step_callback=recorder,
    )
    np.testing.assert_allclose(np.array(recorder.poses), columns["peg_pose"], atol=1e-9)


def test_episode_terminal_reason_policy():
    deep = dict(success_depth=0.015, lateral_tolerance=0.006, force_cap=50.0)
    # seated → SUCCESS (wins even with high force / lock).
    assert (
        episode_terminal_reason(
            penetration=0.02, lateral_error=0.003, force_magnitude=99, locked=True, **deep
        )
        is TerminalReason.SUCCESS
    )
    # HOLD lock (frozen arm) → FORCE_ABORT, independent of the raw force cap.
    assert (
        episode_terminal_reason(
            penetration=0.0, lateral_error=0.05, force_magnitude=1.0, locked=True, **deep
        )
        is TerminalReason.FORCE_ABORT
    )
    # over the force cap → FORCE_ABORT.
    assert (
        episode_terminal_reason(
            penetration=0.0, lateral_error=0.05, force_magnitude=60, locked=False, **deep
        )
        is TerminalReason.FORCE_ABORT
    )
    # mid-approach, no contact → keep going.
    assert (
        episode_terminal_reason(
            penetration=-0.1, lateral_error=0.05, force_magnitude=1.0, locked=False, **deep
        )
        is None
    )
