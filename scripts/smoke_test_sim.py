"""M1 smoke test — load the scene, step it, save a wrist-cam PNG, open viewer.

Run from the `kevin/` directory:

    uv run python scripts/smoke_test_sim.py
    uv run python scripts/smoke_test_sim.py --no-viewer    # CI / headless

The acceptance criteria in `docs/milestone-1-spec.md` correspond to the
sections labelled [criterion N] below.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.sim.scene import SimEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = REPO_ROOT / "assets" / "mjcf" / "full_scene.xml"
OUTPUT_DIR = REPO_ROOT / "outputs"
WRIST_CAM_PNG = OUTPUT_DIR / "m1_wrist_cam.png"

log = get_logger("smoke")

CAMERA_HEIGHT = 256  # bumped from the M1 spec's 128×128 for legibility in the PNG
CAMERA_WIDTH = 256
SETTLE_STEPS = 100
PRINT_EVERY = 10


def _format_obs(obs, label: str) -> str:
    F = obs.wrist_ft[:3]
    T = obs.wrist_ft[3:]
    return (
        f"[{label}] t={obs.sim_time:.3f}s  "
        f"q[:3]={obs.joint_positions[:3].round(3).tolist()}  "
        f"ee_pos={obs.ee_pose[:3].round(3).tolist()}  "
        f"peg_pos={obs.peg_pose[:3].round(3).tolist()}  "
        f"|F|={np.linalg.norm(F):.3f}N  |T|={np.linalg.norm(T):.4f}Nm"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the interactive viewer step (use in CI / over SSH without a display).",
    )
    add_logging_arguments(parser)

    args = parser.parse_args()
    configure_from_args(args)

    if not SCENE_PATH.exists():
        log.error("scene file not found at %s", SCENE_PATH)
        return 2

    log.info("loading scene: %s", SCENE_PATH)
    env = SimEnv(
        str(SCENE_PATH),
        render_mode="headless",
        camera_height=CAMERA_HEIGHT,
        camera_width=CAMERA_WIDTH,
    )
    log.info(
        "model: nq=%d nv=%d nu=%d nbody=%d",
        env.model.nq,
        env.model.nv,
        env.model.nu,
        env.model.nbody,
    )

    # ---- [criterion: reset works] -----------------------------------
    obs = env.reset()
    log.info("%s", _format_obs(obs, "reset"))

    # Sanity-log every hole pose [criterion: hole poses match MJCF].
    log.info("hole sites (world frame, position only):")
    for i, hole_pose in enumerate(obs.hole_poses):
        marker = "  <-- hole_0 (task goal)" if i == 0 else ""
        log.info("  hole_%d: pos=%s%s", i, hole_pose[:3].round(4).tolist(), marker)

    # ---- [criterion: step + sensor read] ----------------------------
    # Since M2 swapped the arm to motor (torque) actuators, ctrl=0 means
    # zero torque — the arm would fall under gravity. We hold it in place
    # with a one-line gravity / Coriolis compensation read straight from
    # data.qfrc_bias so the F/T sanity check below still reflects "arm
    # holding distal mass against gravity". This is not the M2 controller
    # (no impedance, no IK) — just enough to keep the static-equilibrium
    # assumption of the original smoke test valid.
    log.info("stepping %d steps, logging every %d:", SETTLE_STEPS, PRINT_EVERY)
    arm_dof = env.model.jnt_dofadr[
        [
            env.model.joint(n).id
            for n in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
        ]
    ]
    for step in range(SETTLE_STEPS):
        env.data.ctrl[:7] = env.data.qfrc_bias[arm_dof] - env.data.qfrc_constraint[arm_dof]
        env.step()
        if (step + 1) % PRINT_EVERY == 0:
            log.info("%s", _format_obs(env.get_observation(), f"step {step + 1:3d}"))

    final_obs = env.get_observation()

    # ---- F/T sanity check [criterion: F/T reflects gravity] ---------
    # The wrist F/T sensor measures the load distal to it: hand + fingers + peg.
    # In MuJoCo this corresponds to ~(0.73 + 0.015*2 + 0.030)*g = ~7.75 N.
    # Spec's "~0.3 N peg-only" expectation assumes a sensor between gripper
    # and peg, which isn't where the real Panda's wrist sensor sits.
    distal_mass = 0.73 + 2 * 0.015 + 0.030
    expected_distal_weight = distal_mass * 9.81
    F_mag = float(np.linalg.norm(final_obs.wrist_ft[:3]))
    log.info(
        "|F| = %.3f N    (expected ~%.3f N from gravity on distal mass %.3f kg)",
        F_mag,
        expected_distal_weight,
        distal_mass,
    )
    if abs(F_mag - expected_distal_weight) > 0.5:
        log.warning("|F| differs from expected by %.2f N", abs(F_mag - expected_distal_weight))

    # ---- Render wrist cam [criterion: PNG visually shows wall+holes+peg] ----
    frame = env.render_wrist_camera()
    log.info(
        "rendered wrist camera: shape=%s dtype=%s range=[%s, %s]",
        frame.shape,
        frame.dtype,
        frame.min(),
        frame.max(),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(WRIST_CAM_PNG)
    log.info("saved %s", WRIST_CAM_PNG)

    # ---- Viewer [criterion: viewer window opens and rotates] --------
    if args.no_viewer:
        log.info("--no-viewer passed; skipping interactive viewer.")
    else:
        log.info("opening interactive viewer — mouse-drag to rotate, scroll to zoom, ESC to close.")
        env_v = SimEnv(str(SCENE_PATH), render_mode="viewer")
        env_v.reset()
        try:
            env_v.launch_viewer()
            while env_v.viewer is not None and env_v.viewer.is_running():
                env_v.data.ctrl[:7] = (
                    env_v.data.qfrc_bias[arm_dof] - env_v.data.qfrc_constraint[arm_dof]
                )
                env_v.step()
                env_v.sync_viewer()
                # Mujoco's default timestep is 2 ms; sleep proportionally so
                # the viewer runs at roughly real-time rather than as fast as
                # the CPU can chew through frames.
                time.sleep(env_v.model.opt.timestep)
        finally:
            env_v.close()

    env.close()
    log.info("M1 smoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
