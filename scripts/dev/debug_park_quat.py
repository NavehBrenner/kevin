"""Park is not auto-transitioning to HOLD even though pos err is < 5 mm.
Hypothesis: orientation error is > 3 ° because the arm started from a
wall-contact pose where the gripper got tilted slightly. This script
reproduces the harness scenario and prints the orientation error during
the park slew.
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

    # Simulate the harness up through force-trip.
    home_pos = ctrl.home_pose[:3]
    home_quat = ctrl.home_pose[3:]

    # Phase 1: waypoints
    for wp in [home_pos + d for d in [
        [0, 0.05, 0.05], [0, 0.05, -0.05], [0, -0.05, -0.05], [0, -0.05, 0.05]
    ]]:
        cmd = Command(target_position=wp, target_quaternion=home_quat.copy())
        for _ in range(1000):  # 2 s
            obs = env.get_observation(); ctrl.compute(obs, cmd); env.step()

    # Phase 2: compliance
    cmd = Command(target_position=np.array([0.84, 0.0, 0.55]), target_quaternion=home_quat.copy())
    for _ in range(4000):  # 8 s
        obs = env.get_observation(); ctrl.compute(obs, cmd); env.step()

    # Phase 3: force-trip
    ctrl.stiffness_tcp = np.array([2000., 2000., 2000., 3., 3., 3.])
    ctrl.damping_tcp = np.array([180., 180., 180., 4., 4., 4.])
    cmd = Command(target_position=np.array([0.89, 0.0, 0.55]), target_quaternion=home_quat.copy())
    for _ in range(1500):  # 3 s
        obs = env.get_observation(); ctrl.compute(obs, cmd); env.step()
        if ctrl.status.state.value != "active":
            print(f"force-trip tripped at t={obs.sim_time:.3f}s, state={ctrl.status.state.value}")
            break
    # Reset gains.
    ctrl.stiffness_tcp = np.array([400., 400., 500., 3., 3., 3.])
    ctrl.damping_tcp = np.array([80., 80., 89., 4., 4., 4.])

    # Phase 4: release + park
    ctrl.release_lock()
    ctrl.request_park_lock()
    obs = env.get_observation()
    print(f"\nAfter release+park request: state={ctrl.status.state.value}")
    print(f"  ee_pos = {obs.ee_pose[:3].round(4).tolist()}  home = {home_pos.round(4).tolist()}")
    ax = np.zeros(3); mujoco.mju_subQuat(ax, home_quat, obs.ee_pose[3:])
    print(f"  rot err = {np.rad2deg(np.linalg.norm(ax)):.3f}°")

    print("\nPark trace:")
    cmd = Command(target_position=home_pos, target_quaternion=home_quat.copy())
    for step in range(4500):  # 9 s
        obs = env.get_observation(); ctrl.compute(obs, cmd); env.step()
        if step % 250 == 249:
            pos_err = np.linalg.norm(obs.ee_pose[:3] - home_pos) * 1000
            ax = np.zeros(3); mujoco.mju_subQuat(ax, home_quat, obs.ee_pose[3:])
            rot_err = np.rad2deg(np.linalg.norm(ax))
            print(f"  t={obs.sim_time:6.2f}s  pos={pos_err:6.2f} mm  "
                  f"rot={rot_err:5.2f}°  state={ctrl.status.state.value}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
