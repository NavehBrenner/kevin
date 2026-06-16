"""Sweep `joint_damping` to find the value that stabilises hold-at-target
without crushing free-space slew speed.

Two measurements per sweep value:
  1. Approach speed   — slew from home to a point 40 cm away; report time-to-target.
  2. Hold stability   — after settling on a 5 cm lateral target, hold for 4 s;
                        report drift magnitude and qvel_max at the end.

joint_damping=8 stabilises hold but cuts slew to ~0.03 m/s. We want hold drift
< 2 mm with slew speed > 0.08 m/s.
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


def run_approach(env, ctrl, target, max_s):
    cmd = Command(target_position=target, target_quaternion=ctrl.home_pose[3:].copy())
    n = int(max_s / 0.002)
    time_to_target = None
    for step in range(n):
        obs = env.get_observation()
        ctrl.compute(obs, cmd)
        env.step()
        if time_to_target is None and np.linalg.norm(obs.ee_pose[:3] - target) < 0.005:
            time_to_target = (step + 1) * 0.002
    final_err = float(np.linalg.norm(env.get_observation().ee_pose[:3] - target))
    return time_to_target, final_err


def run_hold(env, ctrl, target, hold_s):
    cmd = Command(target_position=target, target_quaternion=ctrl.home_pose[3:].copy())
    n = int(hold_s / 0.002)
    init = env.get_observation().ee_pose[:3].copy()
    for _ in range(n):
        obs = env.get_observation()
        ctrl.compute(obs, cmd)
        env.step()
    obs = env.get_observation()
    drift = float(np.linalg.norm(obs.ee_pose[:3] - init))
    qvel_max = float(np.abs(obs.joint_velocities).max())
    final_err = float(np.linalg.norm(obs.ee_pose[:3] - target))
    return drift, qvel_max, final_err


def main() -> int:
    print(
        f"{'kd_joint':>8}  {'approach t→tgt':>14}  {'approach err':>13}  "
        f"{'hold drift':>11}  {'hold qvel_max':>13}"
    )
    print("-" * 75)
    for kd in (1.0, 2.0, 4.0, 6.0, 8.0, 12.0):
        # Approach.
        env = SimEnv(str(SCENE), render_mode="headless")
        env.reset()
        ctrl = Controller(env, joint_damping=kd)
        far_target = np.array([0.76, 0.0, 0.45])
        t_to_target, app_err = run_approach(env, ctrl, far_target, max_s=8.0)
        env.close()

        # Hold (separate env so previous slew doesn't contaminate).
        env = SimEnv(str(SCENE), render_mode="headless")
        env.reset()
        ctrl = Controller(env, joint_damping=kd)
        lat_target = ctrl.home_pose[:3] + np.array([0.0, 0.05, 0.0])
        # First settle on the lateral target for 3 s, then start the hold measurement.
        cmd = Command(target_position=lat_target, target_quaternion=ctrl.home_pose[3:].copy())
        for _ in range(1500):
            obs = env.get_observation()
            ctrl.compute(obs, cmd)
            env.step()
        drift, qvel_max, hold_err = run_hold(env, ctrl, lat_target, hold_s=4.0)
        env.close()

        ttt = f"{t_to_target:.2f}s" if t_to_target else "—"
        print(
            f"{kd:8.1f}  {ttt:>14}  {app_err * 1000:8.1f} mm  "
            f"{drift * 1000:6.2f} mm  {qvel_max:11.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
