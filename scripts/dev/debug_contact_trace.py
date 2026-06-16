"""Trace what the EE does during compliance + force-trip.

The harness's force-trip phase keeps printing |F| ≈ 12 N even though we
command 5 cm deeper than the compliance target. This script prints
ee_pose, |F|, and net contact F = |F| − gravity_baseline every 100 ms
so we can see whether the peg is sliding back, bouncing off, or just
not pushing hard enough.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ai_teleop.common.command import Command  # noqa: E402
from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402

SCENE = Path(__file__).resolve().parent.parent.parent / "assets" / "mjcf" / "full_scene.xml"
FLAT_WALL = np.array([0.79, 0.0, 0.55])
GRAVITY_BASELINE = 7.75  # N — distal-mass weight at rest


def trace(env, ctrl, target, label, n_steps):
    cmd = Command(target_position=target, target_quaternion=ctrl.home_pose[3:].copy())
    for step in range(n_steps):
        obs = env.get_observation()
        ctrl.compute(obs, cmd)
        env.step()
        if step % 50 == 49:  # every 100 ms
            f_mag = float(np.linalg.norm(obs.wrist_ft[:3]))
            ee_x = obs.ee_pose[0]
            gap_x = (target[0] - ee_x) * 1000  # mm, positive = EE is short of target
            print(
                f"  {label}  t={obs.sim_time:6.3f}s  ee_x={ee_x:.4f}  "
                f"gap_x={gap_x:+7.1f} mm  |F|={f_mag:5.2f}  "
                f"contact≈{f_mag - GRAVITY_BASELINE:+5.2f} N"
            )


def main() -> int:
    env = SimEnv(str(SCENE), render_mode="headless")
    env.reset()
    ctrl = Controller(env)

    # 1. Approach + compliance (target 5 cm past the wall).
    compliance_target = FLAT_WALL + np.array([0.05, 0.0, 0.0])
    print(f"COMPLIANCE target {compliance_target.tolist()} (wall_x=0.80, intrusion 5 cm)")
    trace(env, ctrl, compliance_target, "[compl]", n_steps=4000)  # 8 s

    # 2. Force-trip (10 cm intrusion).
    trip_target = FLAT_WALL + np.array([0.10, 0.0, 0.0])
    print(f"\nFORCE-TRIP target {trip_target.tolist()} (intrusion 10 cm)")
    trace(env, ctrl, trip_target, "[trip] ", n_steps=2000)  # 4 s

    print(f"\nFinal lock state: {ctrl.status.state.value} ({ctrl.status.last_transition_reason})")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
