"""Headless backbone regression smoke test (LAB-21).

The impedance law and lock state machine are otherwise exercised only by
`scripts/dev_harness_controller.py --headless`, which CI does not run. This
module promotes the harness's core invariants into fast pytest checks so a
regression in the control gains or the contact/force handling shows up in
the gate when M3+ work shifts solver settings.

It is deliberately *looser and shorter* than the full harness — trimmed step
counts and relaxed tolerances. For the authoritative acceptance numbers run
the harness (see `docs/milestone-2-spec.md` *Acceptance criteria*).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_teleop.common.command import Command
from ai_teleop.control import Controller, LockState
from ai_teleop.sim.scene import SimEnv

SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "full_scene.xml"

# Sim runs at 500 Hz (dt=2 ms in the MJCF); one control tick == one sim step.
# Trimmed vs. the harness (which holds each phase for 2 s / 1000 steps).
WAYPOINT_STEPS = 800  # ~1.6 s — enough for the impedance to settle
PARK_STEPS = 1000  # ~2.0 s — slew home + auto-lock
TRIP_STEPS = 100  # gravity load alone trips the lowered cap within a few ticks

# Loosened tolerance (harness asserts < 5 mm; we accept < 10 mm here).
POS_TOL_M = 10e-3


@pytest.fixture(scope="module")
def env():
    if not SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {SCENE_PATH}")
    e = SimEnv(str(SCENE_PATH), render_mode="headless")
    yield e
    e.close()


def _drive(env, controller, target_pos, target_quat, n_steps):
    """Step the closed loop n times toward a fixed target; return summary."""
    peak_force = 0.0
    transitions = []
    prev_state = controller.status.state
    last_pos_err = float("nan")
    for _ in range(n_steps):
        obs = env.get_observation()
        controller.compute(obs, Command(target_position=target_pos, target_quaternion=target_quat))
        env.step()
        peak_force = max(peak_force, float(np.linalg.norm(obs.wrist_ft[:3])))
        last_pos_err = float(np.linalg.norm(obs.ee_pose[:3] - target_pos))
        state = controller.status.state
        if state != prev_state:
            transitions.append((state, controller.status.last_transition_reason))
            prev_state = state
    return last_pos_err, peak_force, transitions


def test_waypoint_tracking(env):
    """Impedance law tracks a commanded waypoint to within tolerance."""
    env.reset()
    controller = Controller(env)
    home_pos, home_quat = controller.home_pose[:3], controller.home_pose[3:]
    target = home_pos + np.array([0.0, 0.05, 0.05])
    pos_err, _, transitions = _drive(env, controller, target, home_quat, WAYPOINT_STEPS)
    assert controller.status.state is LockState.ACTIVE
    assert not transitions, f"unexpected lock transitions during free tracking: {transitions}"
    assert pos_err < POS_TOL_M, (
        f"waypoint pos err {pos_err * 1000:.2f} mm >= {POS_TOL_M * 1000:.0f} mm"
    )


def test_force_cap_trips_to_hold(env):
    """Wrist force exceeding the cap drives ACTIVE -> HoldLock exactly once.

    Rather than slew to the wall and stiffen the impedance (slow), we lower the
    force cap below the ~7.75 N resting gravity load on the distal mass, so the
    watchdog trips on the real force signal within a few ticks. Same code path
    the harness's force-trip phase exercises, far cheaper.
    """
    env.reset()
    controller = Controller(env, force_cap_n=5.0)
    home_pos, home_quat = controller.home_pose[:3], controller.home_pose[3:]
    _, peak_force, transitions = _drive(env, controller, home_pos, home_quat, TRIP_STEPS)
    assert peak_force > controller.force_cap_n  # the signal really did exceed the cap
    assert controller.status.state is LockState.HOLD
    trips = [r for s, r in transitions if s is LockState.HOLD and r.startswith("force_cap_trip")]
    assert len(trips) == 1, f"expected exactly one force-cap trip, got transitions {transitions}"


def test_reset_clears_latched_lock(env):
    """controller.reset() releases a tripped lock back to ACTIVE.

    Regression guard: a `Controller` reused across episodes (the data-gen loop)
    must clear its lock between them — otherwise one episode's force-cap → HOLD
    trip silently freezes every episode after it (observed as a corpus collapse
    to ~5% seated). Reuses the cheap force-trip from above, then resets.
    """
    env.reset()
    controller = Controller(env, force_cap_n=5.0)
    home_pos, home_quat = controller.home_pose[:3], controller.home_pose[3:]
    _drive(env, controller, home_pos, home_quat, TRIP_STEPS)
    assert controller.status.state is LockState.HOLD  # latched by the force cap

    controller.reset()
    assert controller.status.state is LockState.ACTIVE, "reset() did not clear the lock"


def test_park_returns_home_and_locks(env):
    """request_park_lock() slews to home and auto-transitions PARK -> HoldLock."""
    env.reset()
    controller = Controller(env)
    home_pos, home_quat = controller.home_pose[:3], controller.home_pose[3:]
    # Move ~7 cm off home so the park slew is a real move, not a no-op.
    _drive(env, controller, home_pos + np.array([0.0, 0.05, 0.05]), home_quat, WAYPOINT_STEPS)
    controller.request_park_lock()
    assert controller.status.state is LockState.PARK
    _drive(env, controller, home_pos, home_quat, PARK_STEPS)
    obs = env.get_observation()
    pos_err = float(np.linalg.norm(obs.ee_pose[:3] - home_pos))
    assert controller.status.state is LockState.HOLD, "park did not auto-lock to HoldLock"
    assert pos_err < POS_TOL_M, (
        f"parked pos err {pos_err * 1000:.2f} mm >= {POS_TOL_M * 1000:.0f} mm"
    )
