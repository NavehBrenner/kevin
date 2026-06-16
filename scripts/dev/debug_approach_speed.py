"""Why does the EE only travel ~20 cm in 5 s during the approach phase?

Theory says with K_xy=400 and Cartesian D=80 plus joint_damping=8, the
steady-state slew speed should be K·Δ_clamp/D ≈ 0.1 m/s, giving a 41 cm
approach to the wall in ~4 s. The dev harness measured ~0.04 m/s. This
script slews straight from the home pose to a target 3 cm in front of the
target hole and prints the per-second position error and joint velocity,
so we can see whether the slow speed is real or a harness artifact (e.g.,
the previous waypoint phase parking the arm in a bad starting
configuration).
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


def main() -> int:
    env = SimEnv(str(SCENE), render_mode="headless")
    env.reset()
    ctrl = Controller(env)

    target = np.array([0.76, 0.0, 0.45])
    cmd = Command(target_position=target, target_quaternion=ctrl.home_pose[3:].copy())
    print(
        f"Approach from home {ctrl.home_pose[:3].round(3).tolist()} "
        f"to {target.round(3).tolist()} (distance {np.linalg.norm(target - ctrl.home_pose[:3]) * 100:.1f} cm)"
    )
    for step in range(4000):  # 8 s
        obs = env.get_observation()
        ctrl.compute(obs, cmd)
        env.step()
        if step % 250 == 249:
            e = (obs.ee_pose[:3] - target) * 1000
            print(
                f"  t={(step + 1) * 0.002:5.2f}s  err xyz mm = "
                f"({e[0]:+7.1f}, {e[1]:+7.1f}, {e[2]:+7.1f})  "
                f"qvel_max={np.abs(obs.joint_velocities).max():.4f}  "
                f"|F|={np.linalg.norm(obs.wrist_ft[:3]):.2f}"
            )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
