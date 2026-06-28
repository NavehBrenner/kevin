"""LAB-78: `kvn episode` replay must reproduce the generation episode.

The structural bug this guards: the replay used to rebuild the wrong scene (default
home hole, not the coverage-randomized one) and feed back the *recorded* command
stream (truncated at the recorded run's terminal step). The fix reconstructs the
generation scene from ``scene_seed`` and the operator from its seed — so a replay
matches its generated episode under any policy. These pin the two halves:

1. the reconstructed scene + operator reproduce the recorded command stream exactly;
2. the shared termination policy is the single source of truth for "episode over".
"""

from __future__ import annotations

import numpy as np

from ai_teleop.control import Controller
from ai_teleop.data.generate import (
    SCENE_PATH,
    episode_terminal_reason,
    generate_dataset,
    make_episode_operator,
)
from ai_teleop.data.trajectory import TerminalReason, load_episode
from ai_teleop.sim.scene import SimEnv


def test_reconstructed_scene_and_operator_reproduce_generation(tmp_path):
    paths = generate_dataset(tmp_path, n_episodes=2, seed=0, max_steps=300, baseline=False)

    for path in paths:
        columns, meta = load_episode(path)
        master_seed, episode_index = (int(v) for v in meta["scene_seed"])

        env = SimEnv(str(SCENE_PATH), seed=master_seed, randomize=True)
        observation = env.reset(episode_index)
        # The coverage-randomized scene is reconstructed (not the default hole).
        assert observation.target_hole_index == meta["target_hole_index"]

        controller = Controller(env, max_dpos_per_step=float(meta["max_dpos"]))
        operator = make_episode_operator(
            observation.target_hole_position.copy(),
            controller.home_pose[3:],
            seed=master_seed,
            episode_index=episode_index,
            max_approach_speed=float(meta["max_approach_speed"]),
        )

        # Re-running the reconstructed operator reproduces the recorded command stream
        # byte-for-byte (so a replay needs no recorded commands and can run any policy).
        for recorded_position, recorded_quaternion in zip(
            columns["cmd_position"], columns["cmd_quaternion"], strict=True
        ):
            command = operator.get_command(observation)
            np.testing.assert_array_equal(command.target_position, recorded_position)
            np.testing.assert_array_equal(command.target_quaternion, recorded_quaternion)
            controller.compute(observation, command)
            env.step()
            observation = env.get_observation()


def test_episode_terminal_reason_policy():
    deep = dict(success_depth=0.015, lateral_tolerance=0.006, force_cap=50.0)
    # seated → SUCCESS (and SUCCESS wins even with high force).
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
