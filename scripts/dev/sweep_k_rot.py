"""Sweep K_rot (and matching D_rot) to find the highest value that doesn't
destabilise free-space position tracking but is still fast enough to
unwind a ~25° orientation error within ~6 seconds (needed for the park
phase after force-trip).

For each K_rot:
  - 1. Hold-at-target (5 cm lateral move): report pos drift over 4 s after
       initial settle, and rot tracking error.
  - 2. Orientation slew: start with EE at home, command target_quat rotated
       30° about z, hold for 4 s; report orientation error vs time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ai_teleop.common.command import Command  # noqa: E402
from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402

SCENE = Path(__file__).resolve().parent.parent.parent / "assets" / "mjcf" / "full_scene.xml"


def quat_rotated_about_z(q, deg):
    theta = np.deg2rad(deg)
    rot = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, rot, q)
    return out


def measure(K_rot, D_rot):
    # Hold lateral test.
    env = SimEnv(str(SCENE), render_mode="headless")
    env.reset()
    ctrl = Controller(env)
    ctrl.stiffness_tcp = np.array([400.0, 400.0, 500.0, K_rot, K_rot, K_rot])
    ctrl.damping_tcp = np.array([80.0, 80.0, 89.0, D_rot, D_rot, D_rot])
    target_pos = ctrl.home_pose[:3] + np.array([0, 0.05, 0])
    cmd = Command(target_position=target_pos, target_quaternion=ctrl.home_pose[3:].copy())
    # Settle 3 s.
    for _ in range(1500):
        ctrl.compute(env.get_observation(), cmd)
        env.step()
    # Measure drift over next 4 s.
    poses = []
    for _ in range(2000):
        obs = env.get_observation()
        ctrl.compute(obs, cmd)
        env.step()
        poses.append(obs.ee_pose[:3].copy())
    poses = np.array(poses)
    pos_drift_mm = float((poses.max(0) - poses.min(0)).max() * 1000)
    env.close()

    # Orientation slew test.
    env = SimEnv(str(SCENE), render_mode="headless")
    env.reset()
    ctrl = Controller(env)
    ctrl.stiffness_tcp = np.array([400.0, 400.0, 500.0, K_rot, K_rot, K_rot])
    ctrl.damping_tcp = np.array([80.0, 80.0, 89.0, D_rot, D_rot, D_rot])
    target_quat = quat_rotated_about_z(ctrl.home_pose[3:], 25.0)
    cmd = Command(target_position=ctrl.home_pose[:3].copy(), target_quaternion=target_quat)
    rot_at_t = {}
    for step in range(4000):  # 8 s
        obs = env.get_observation()
        ctrl.compute(obs, cmd)
        env.step()
        if step + 1 in {1000, 2000, 3000, 4000}:  # 2, 4, 6, 8 s
            ax = np.zeros(3)
            mujoco.mju_subQuat(ax, target_quat, obs.ee_pose[3:])
            rot_at_t[(step + 1) * 0.002] = np.rad2deg(np.linalg.norm(ax))
    env.close()
    return pos_drift_mm, rot_at_t


def main():
    print(f"{'K_rot':>6}  {'D_rot':>6}  {'pos_drift_4s_mm':>16}  rot_err at t=2,4,6,8 s (deg)")
    print("-" * 80)
    for K_rot in (3, 5, 8, 12, 18):
        D_rot = max(2 * np.sqrt(K_rot * 0.5), 4.0)  # zeta=1 at I=0.5, floor at 4
        pos_drift, rot_t = measure(K_rot, D_rot)
        rot_str = "  ".join(f"{rot_t[t]:6.2f}" for t in sorted(rot_t))
        print(f"{K_rot:6.1f}  {D_rot:6.2f}  {pos_drift:14.2f}    {rot_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
