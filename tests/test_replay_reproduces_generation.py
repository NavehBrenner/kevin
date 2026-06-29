"""LAB-78: replaying a recorded episode reproduces it byte-for-byte.

The bug this guards: replay rebuilt the wrong scene (incomplete metadata), so the
recorded commands drove a different physical episode. The fix stamps the complete
scene/controller spec into metadata and replays the recorded commands verbatim
(dumb iterator) in the reconstructed scene. Same scene + same commands ⇒ the exact
same trajectory — so the realized peg pose must match the recording tick-for-tick.
"""

from __future__ import annotations

import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.control import Controller
from ai_teleop.data.generate import (
    SCENE_PATH,
    episode_terminal_reason,
    generate_dataset,
)
from ai_teleop.data.trajectory import TerminalReason, load_episode
from ai_teleop.domain import Delta
from ai_teleop.sim.runner import run_episode
from ai_teleop.sim.scene import SimEnv


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

    def get_delta(self, observation, base_command):
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
        master_seed, episode_index = (int(v) for v in meta["scene_seed"])

        # Rebuild the exact scene from the stored spec (the part that was broken).
        env = SimEnv(str(SCENE_PATH), seed=master_seed, randomize=True)
        observation = env.reset(episode_index)
        assert observation.target_hole_index == meta["target_hole_index"]
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
            reset_episode_index=episode_index,
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
