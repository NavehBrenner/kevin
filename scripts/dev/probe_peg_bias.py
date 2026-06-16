"""Read-only probe: where does the pre-grasped peg's weight show up in joint space?

Hypothesis under test: `data.qfrc_bias` accounts only for the arm's own link
gravity, NOT the welded peg. The peg's weight reaches the arm through the weld
*constraint*, so it lives in `data.qfrc_constraint`, not `qfrc_bias`.

This script does NOT fix the smoke test. It only prints the per-joint force
terms after settling, so we can confirm the leak and see it concentrated on the
pitch joints (joint2, joint4, joint6).

Run from code/:
    .venv/bin/python scripts/dev/probe_peg_bias.py
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


def main() -> int:
    environment = SimEnv(str(SCENE_PATH), render_mode="headless")
    model = environment.model
    arm_dof = model.jnt_dofadr[[model.joint(n).id for n in ARM_JOINTS]]

    environment.reset()

    # Settle with the CURRENT (buggy) gravity comp: ctrl = qfrc_bias only.
    for _ in range(200):
        environment.data.ctrl[:7] = environment.data.qfrc_bias[arm_dof]
        environment.step()

    data = environment.data
    bias = data.qfrc_bias[arm_dof]
    constraint = data.qfrc_constraint[arm_dof]
    passive = data.qfrc_passive[arm_dof]
    # What a perfect hold would need: bias - constraint - passive.
    needed = bias - constraint - passive
    applied = data.ctrl[:7]
    deficit = needed - applied  # uncompensated torque == what causes the sag

    header = f"{'joint':>8} {'qfrc_bias':>12} {'qfrc_constr':>12} {'qfrc_pass':>12} {'deficit':>12}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(ARM_JOINTS):
        print(
            f"{name:>8} {bias[i]:>12.4f} {constraint[i]:>12.4f} "
            f"{passive[i]:>12.4f} {deficit[i]:>12.4f}"
        )

    print(
        f"\n|deficit| = {np.linalg.norm(deficit):.4f} N·m  (this torque is what the arm is sagging under)"
    )
    environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
