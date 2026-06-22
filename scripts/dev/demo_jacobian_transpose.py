"""Make tau = J^T F physical, and connect it back to the Phase-2 peg sag.

A force F at the gripper maps to joint torques via the *transpose* Jacobian:
tau = J^T F. We test two things at the home pose:

  1. A downward force at the TCP (like the peg's weight) maps to torques
     concentrated on the PITCH joints (2, 4, 6) -- exactly the joints that
     sagged in Phase 2 and exactly where qfrc_constraint was nonzero.
  2. A sideways (TCP +x) force loads a *different* joint pattern -- showing
     the mapping is direction-dependent, not a fixed "these joints are heavy".

Run from kevin/:
    .venv/bin/python scripts/dev/demo_jacobian_transpose.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.sim.scene import SimEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "assets" / "mjcf" / "full_scene.xml"
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
PEG_WEIGHT_N = 0.030 * 9.81  # peg mass * g


def main() -> int:
    environment = SimEnv(str(SCENE_PATH), render_mode="headless")
    model, data = environment.model, environment.data
    arm_dof = model.jnt_dofadr[[model.joint(n).id for n in ARM_JOINTS]]
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")

    environment.reset()

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, tcp)
    J = np.vstack([jacp[:, arm_dof], jacr[:, arm_dof]])  # (6,7)

    def tau_for(wrench, label):
        tau = J.T @ wrench
        print(f"\n[{label}]  F = {wrench[:3].round(3)} N,  M = {wrench[3:].round(3)} N·m")
        print("    " + "  ".join(f"j{i + 1}:{tau[i]:+7.4f}" for i in range(7)))

    print("tau = J^T @ F   (joint torques, N·m, that realise an EE wrench F)")
    # 1) Peg weight: a downward (world -z) force at the gripper.
    tau_for(np.array([0, 0, -PEG_WEIGHT_N, 0, 0, 0]), "downward force ~ peg weight")
    # 2) Same magnitude, sideways.
    tau_for(np.array([PEG_WEIGHT_N, 0, 0, 0, 0, 0]), "sideways (+x) force, same magnitude")

    print("\nRecall Phase 2 measured qfrc_constraint (the peg's weld load) as:")
    print("    j2:+0.114  j4:-0.178  j6:-0.067   (≈0 on 1,3,5,7)")
    print("Same pitch-joint signature -> the weld load IS a downward EE force, J^T-mapped.")
    environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
