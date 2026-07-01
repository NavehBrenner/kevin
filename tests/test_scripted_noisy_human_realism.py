"""LAB-78 realism regressions for ScriptedNoisyHuman's command stream.

The command is a live policy input (the command-history GRU), so its *dynamics*
matter, not just where it ends up. These pin the three behaviors that the old
"parked at the goal from tick 0" model lacked and that would silently regress:
an approach phase, per-tick continuity (no holds, no jumps), and determinism.
Magnitudes are still placeholders (calibrated in LAB-77); only the form is fixed.
"""

from __future__ import annotations

import numpy as np

from ai_teleop.common.observation import Observation
from ai_teleop.input import ScriptedNoisyHuman

CONTROL_HZ = 500.0
MAX_APPROACH_SPEED = 0.35


def _make_observation(ee_position: np.ndarray) -> Observation:
    return Observation(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_pose=np.concatenate([ee_position, [1.0, 0.0, 0.0, 0.0]]),
        wrist_ft=np.zeros(6),
        gripper_width=0.08,
        peg_pose=np.zeros(7),
        hole_poses=np.zeros((1, 7)),
        sim_time=0.0,
    )


def _make_actor(seed: int = 0) -> ScriptedNoisyHuman:
    goal = np.array([0.9, 0.0, 0.3])  # ~400 mm from the arm start below
    return ScriptedNoisyHuman(
        np.concatenate([goal, [1.0, 0.0, 0.0, 0.0]]),
        max_approach_speed=MAX_APPROACH_SPEED,
        control_hz=CONTROL_HZ,
        seed=seed,
    )


def test_approach_phase_exists():
    # First command sits at the arm's start pose (far from the goal); a late
    # command has swept in to the goal. The old model parked at the goal tick 0.
    ee_position = np.array([0.5, 0.0, 0.3])
    obs = _make_observation(ee_position)
    actor = _make_actor()
    goal = actor._goal_position  # noqa: SLF001 — test pins the seeded approach

    first = actor.get_command(obs).target_position
    one_tick = MAX_APPROACH_SPEED / CONTROL_HZ
    # Seeded at the arm, within one tick's travel of it...
    assert np.linalg.norm(first - ee_position) <= one_tick + 1e-9
    # ...and far from the goal (≫ chamfer band, ~5 mm).
    assert np.linalg.norm(first - goal) > 0.05

    for _ in range(3000):
        late = actor.get_command(obs).target_position
    # Arrived: within the drift envelope of the goal (the command chases
    # goal + drift_t, not the bare goal), i.e. ≪ the ~400 mm it started out.
    assert np.linalg.norm(late - goal) < 0.03


def test_command_is_continuous_no_holds_no_jumps():
    obs = _make_observation(np.array([0.5, 0.0, 0.3]))
    actor = _make_actor(seed=2)
    positions = np.array([actor.get_command(obs).target_position for _ in range(2000)])

    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    max_step = MAX_APPROACH_SPEED / CONTROL_HZ
    # No jumps: every per-tick move respects the cap (small float margin).
    assert steps.max() <= max_step + 1e-9
    # No holds: the command keeps moving (per-tick drift), most ticks are nonzero.
    assert (steps > 1e-7).mean() > 0.4


def test_deterministic_for_same_seed_and_observations():
    obs = _make_observation(np.array([0.5, 0.0, 0.3]))
    actor_a, actor_b = _make_actor(seed=7), _make_actor(seed=7)
    for _ in range(500):
        cmd_a = actor_a.get_command(obs)
        cmd_b = actor_b.get_command(obs)
        np.testing.assert_array_equal(cmd_a.target_position, cmd_b.target_position)
        np.testing.assert_array_equal(cmd_a.target_quaternion, cmd_b.target_quaternion)
