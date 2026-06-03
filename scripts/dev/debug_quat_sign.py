"""Verify the orientation impedance sign by commanding a 10° rotation
about world-z from home and watching where the EE quaternion goes.

If impedance is correctly signed, rot_err should decrease over time.
If reversed, it grows.
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

SCENE = (
    Path(__file__).resolve().parent.parent.parent / "assets" / "mjcf" / "full_scene.xml"
)


def main() -> int:
    env = SimEnv(str(SCENE), render_mode="headless")
    env.reset()
    ctrl = Controller(env)

    home_q = ctrl.home_pose[3:].copy()
    # Target = world-frame rotation of home_q by 10° about world +z.
    half = np.deg2rad(10.0) / 2
    rot_world_z = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    target_q = np.zeros(4)
    mujoco.mju_mulQuat(target_q, rot_world_z, home_q)

    cmd = Command(target_position=ctrl.home_pose[:3].copy(), target_quaternion=target_q)
    print(f"home_q  = {home_q.round(4).tolist()}")
    print(f"target_q= {target_q.round(4).tolist()}")
    ax = np.zeros(3); mujoco.mju_subQuat(ax, target_q, home_q)
    print(f"axis-angle target←home: {ax.round(4).tolist()}  norm={np.rad2deg(np.linalg.norm(ax)):.2f}°")

    print()
    print("Step      rot_err_deg  ee_q[0]    ee_q[1]    ee_q[2]    ee_q[3]")
    for step in range(2000):
        obs = env.get_observation()
        ctrl.compute(obs, cmd)
        env.step()
        if step % 100 == 99:
            cur_q = obs.ee_pose[3:]
            ax = np.zeros(3); mujoco.mju_subQuat(ax, target_q, cur_q)
            err_deg = np.rad2deg(np.linalg.norm(ax))
            print(f"  t={(step+1)*0.002:5.2f}s   {err_deg:6.2f}°    "
                  f"{cur_q[0]:+.4f}  {cur_q[1]:+.4f}  {cur_q[2]:+.4f}  {cur_q[3]:+.4f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
