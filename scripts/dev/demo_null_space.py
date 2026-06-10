"""Make the Jacobian, DLS, and the null space physical & visible.

At the home pose we:
  1. Build the 6x7 TCP Jacobian J.
  2. Find its null-space direction n (J n ~= 0) via SVD.
  3. Nudge the joints along n, re-run forward kinematics, and measure how far
     the TCP moved vs how far an inner link (the "elbow", link4) moved.
        -> TCP barely moves, elbow moves a lot  == null-space / "elbow swing".
  4. For contrast, nudge the joints by the SAME joint-space magnitude but along
     a *task* direction (DLS-solved to move TCP in +x) and measure again.
        -> TCP moves ~as commanded.

Run from code/:
    .venv/bin/python scripts/dev/demo_null_space.py
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
EPS = 0.05  # rad — size of the joint nudge


def tcp_and_elbow(data, tcp_site_id, elbow_body_id):
    return data.site_xpos[tcp_site_id].copy(), data.xpos[elbow_body_id].copy()


def main() -> int:
    environment = SimEnv(str(SCENE_PATH), render_mode="headless")
    model, data = environment.model, environment.data
    arm_qpos = model.jnt_qposadr[[model.joint(n).id for n in ARM_JOINTS]]
    arm_dof = model.jnt_dofadr[[model.joint(n).id for n in ARM_JOINTS]]
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link4")

    environment.reset()
    q0 = data.qpos[arm_qpos].copy()
    tcp0, elbow0 = tcp_and_elbow(data, tcp, elbow)

    # --- Build J at home ---
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, tcp)
    J = np.vstack([jacp[:, arm_dof], jacr[:, arm_dof]])  # (6,7)

    # --- Null-space direction: smallest singular value's right vector ---
    U, S, Vt = np.linalg.svd(J)
    n = Vt[-1]                       # (7,) — J @ n should be ~0
    n = n / np.linalg.norm(n)
    print(f"singular values of J: {S.round(4)}")
    print(f"||J @ n||  (should be ~0): {np.linalg.norm(J @ n):.2e}")

    def apply_and_measure(dq, label):
        data.qpos[arm_qpos] = q0 + dq
        mujoco.mj_forward(model, data)
        tcp1, elbow1 = tcp_and_elbow(data, tcp, elbow)
        d_tcp = np.linalg.norm(tcp1 - tcp0)
        d_elbow = np.linalg.norm(elbow1 - elbow0)
        print(f"\n[{label}]  joint nudge ||dq|| = {np.linalg.norm(dq):.4f} rad")
        print(f"    TCP   moved: {d_tcp*1000:8.3f} mm")
        print(f"    elbow moved: {d_elbow*1000:8.3f} mm")

    # 1) Nudge along the null space.
    apply_and_measure(EPS * n, "NULL-SPACE nudge (elbow swing)")

    # 2) Nudge by the same magnitude along a TASK direction (+x at TCP).
    e = np.array([1.0, 0, 0, 0, 0, 0])  # want +x translation
    JJT = J @ J.T + (0.05 ** 2) * np.eye(6)
    j_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))
    dq_task = j_pinv @ e
    dq_task = EPS * dq_task / np.linalg.norm(dq_task)  # same ||dq|| as the null nudge
    apply_and_measure(dq_task, "TASK nudge (commanded +x)")

    environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
