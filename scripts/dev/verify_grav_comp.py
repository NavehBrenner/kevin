"""Verify the qfrc_constraint gravity-comp fix by measuring joint drift.

Runs 500 settle steps under each compensation scheme and reports how far each
joint drifts from the home keyframe. The fix should shrink joint2/joint4 drift
by ~an order of magnitude.

Run from code/:
    .venv/bin/python scripts/dev/verify_grav_comp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.sim.scene import SimEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "assets" / "mjcf" / "full_scene.xml"
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
N_STEPS = 500


def drift_under(scheme: str) -> np.ndarray:
    environment = SimEnv(str(SCENE_PATH), render_mode="headless")
    model = environment.model
    arm_dof = model.jnt_dofadr[[model.joint(n).id for n in ARM_JOINTS]]
    arm_qpos = model.jnt_qposadr[[model.joint(n).id for n in ARM_JOINTS]]

    environment.reset()
    q0 = environment.data.qpos[arm_qpos].copy()
    for _ in range(N_STEPS):
        bias = environment.data.qfrc_bias[arm_dof]
        if scheme == "old":
            environment.data.ctrl[:7] = bias
        else:  # "fixed"
            environment.data.ctrl[:7] = bias - environment.data.qfrc_constraint[arm_dof]
        environment.step()
    drift = environment.data.qpos[arm_qpos] - q0
    environment.close()
    return drift


def main() -> int:
    old = drift_under("old")
    fixed = drift_under("fixed")
    print(f"Joint drift after {N_STEPS} steps (rad):")
    print(f"{'joint':>8} {'old (bias only)':>18} {'fixed (- constraint)':>22}")
    print("-" * 50)
    for i, name in enumerate(ARM_JOINTS):
        print(f"{name:>8} {old[i]:>18.5f} {fixed[i]:>22.5f}")
    print(f"\n||drift|| old   = {np.linalg.norm(old):.5f} rad")
    print(f"||drift|| fixed = {np.linalg.norm(fixed):.5f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
