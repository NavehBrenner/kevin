"""M2 dev harness — drive the backbone controller through the five phases
of the milestone-2 spec and (in headless mode) emit assertions + a CSV
trace for tuning plots.

Run from the `kevin/` directory:

    uv run python scripts/dev_harness_controller.py                # interactive viewer
    uv run python scripts/dev_harness_controller.py --headless     # CI / regression

Phases (see `docs/milestone-2-spec.md` Step 7 for the design contract):

1. Waypoint  — 10 cm square in front of the wall, four corners, 1 s each.
2. Compliance — target 5 cm past the wall surface; arm contacts, peg seats,
               wrist force plateaus below the force cap.
3. Force-trip — keep ramping commanded depth until the watchdog trips.
4. Release   — `release_lock()`; controller is back in ACTIVE.
5. Park      — `request_park_lock()`; auto-returns home and locks.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common.command import Command  # noqa: E402
from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.control import Controller, LockState  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = REPO_ROOT / "assets" / "mjcf" / "full_scene.xml"
OUTPUT_DIR = REPO_ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "m2_harness_trace.csv"

log = get_logger("harness")

# Sim runs at 500 Hz (dt=2 ms in the MJCF). One control tick == one sim step.
SIM_DT = 0.002

# Phase durations in sim seconds.
WAYPOINT_HOLD_S = 2.0
# Approach + compliance are merged into one phase: the EE has to slew ~40 cm
# from the last waypoint to the wall, then make contact and settle. The
# Cartesian impedance handles the contact transition smoothly because the
# wall just bounds the commanded intrusion.
COMPLIANCE_S = 8.0
FORCE_TRIP_S = 3.0
PARK_TIMEOUT_S = 8.0

# Wall geometry (must match assets/mjcf/wall_with_holes.xml). Holes sit at
# y = −0.1, 0.0, +0.1; pushing the peg directly at a hole inserts it rather
# than seating it against flat wall, which is the wrong test for compliance.
WALL_X = 0.80
TARGET_HOLE_POS = np.array([0.79, 0.0, 0.45])
# Flat-wall contact point — 5 cm above the middle hole so the peg meets
# unbroken wall rather than a rim/chamfer.
FLAT_WALL_POS = np.array([0.79, 0.0, 0.55])

# Compliance phase: push 5 cm past the wall surface (the peg can't physically
# get there — the wall stops it, and the impedance gives laterally).
COMPLIANCE_INTRUSION = 0.05

# Force-trip phase: deeper intrusion so K_z · intrusion clearly exceeds the cap.
FORCE_TRIP_INTRUSION = 0.10

# Acceptance thresholds (Step 7 of the spec).
WAYPOINT_POS_TOL = 5e-3  # 5 mm steady-state error after 1 s hold


def _arr_to_str(a: np.ndarray, n: int = 3) -> str:
    return np.array2string(a, precision=n, separator=", ", suppress_small=True)


def run_phase(
    env: SimEnv,
    controller: Controller,
    *,
    label: str,
    duration_s: float,
    target_pos_fn,
    target_quat: np.ndarray,
    csv_writer: csv.writer | None,
    viewer_real_time: bool,
) -> dict:
    """Run a single phase for `duration_s` of sim time, collecting summary stats.

    `target_pos_fn(t)` returns the commanded target position at sim time `t`
    (so phases can ramp). `target_quat` is held constant within a phase.
    """
    n_steps = int(round(duration_s / SIM_DT))
    peak_force = 0.0
    last_pos_err = float("nan")
    lock_state_changes: list[tuple[float, str, str]] = []
    prev_state = controller.status.state

    for _ in range(n_steps):
        obs = env.get_observation()
        t = obs.sim_time
        cmd = Command(
            target_position=target_pos_fn(t),
            target_quaternion=target_quat,
        )
        controller.compute(obs, cmd)
        env.step()
        env.sync_viewer()

        force_mag = float(np.linalg.norm(obs.wrist_ft[:3]))
        peak_force = max(peak_force, force_mag)
        last_pos_err = float(np.linalg.norm(obs.ee_pose[:3] - cmd.target_position))

        status = controller.status
        if status.state != prev_state:
            lock_state_changes.append(
                (status.last_transition_sim_time, status.state.value, status.last_transition_reason)
            )
            prev_state = status.state

        if csv_writer is not None:
            csv_writer.writerow(
                [
                    f"{t:.4f}",
                    label,
                    status.state.value,
                    f"{obs.ee_pose[0]:.5f}",
                    f"{obs.ee_pose[1]:.5f}",
                    f"{obs.ee_pose[2]:.5f}",
                    f"{obs.wrist_ft[0]:.4f}",
                    f"{obs.wrist_ft[1]:.4f}",
                    f"{obs.wrist_ft[2]:.4f}",
                    f"{force_mag:.4f}",
                ]
            )

        if viewer_real_time:
            # MuJoCo's default 2 ms timestep is way faster than wall time; sleep
            # one timestep so the viewer runs at roughly real-time and the
            # operator can actually see the arm move.
            time.sleep(SIM_DT)

    return {
        "label": label,
        "peak_force": peak_force,
        "last_pos_err": last_pos_err,
        "lock_changes": lock_state_changes,
        "final_state": controller.status.state,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--headless", action="store_true", help="Skip the viewer; run assertions and emit CSV."
    )
    p.add_argument(
        "--force-cap", type=float, default=30.0, help="Force-cap watchdog threshold in newtons."
    )
    add_logging_arguments(p)
    args = p.parse_args()
    configure_from_args(args)

    if not SCENE_PATH.exists():
        log.error("scene file not found at %s", SCENE_PATH)
        return 2

    render_mode = "headless" if args.headless else "viewer"
    log.info("loading scene (%s): %s", render_mode, SCENE_PATH)
    env = SimEnv(str(SCENE_PATH), render_mode=render_mode)
    env.reset()
    if not args.headless:
        env.launch_viewer()

    controller = Controller(env, force_cap_n=args.force_cap)
    home_pos = controller.home_pose[:3]
    home_quat = controller.home_pose[3:]
    log.info("home EE pose: pos=%s quat=%s", _arr_to_str(home_pos), _arr_to_str(home_quat, 4))
    log.info("force cap: %.1f N", controller.force_cap_n)

    # ------------------------------------------------------------------
    # CSV setup (headless only).
    # ------------------------------------------------------------------
    csv_file = None
    csv_writer = None
    if args.headless:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_file = open(CSV_PATH, "w", newline="")  # noqa: SIM115 (streamed across the loop, closed below)
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "sim_time",
                "phase",
                "lock_state",
                "ee_x",
                "ee_y",
                "ee_z",
                "Fx",
                "Fy",
                "Fz",
                "F_mag",
            ]
        )

    summaries: list[dict] = []

    # ------------------------------------------------------------------
    # Phase 1 — waypoints in a 10 cm square in front of the wall.
    # ------------------------------------------------------------------
    log.info("=== Phase 1: Waypoint square ===")
    waypoints = [
        home_pos + np.array([0.0, 0.05, 0.05]),
        home_pos + np.array([0.0, 0.05, -0.05]),
        home_pos + np.array([0.0, -0.05, -0.05]),
        home_pos + np.array([0.0, -0.05, 0.05]),
    ]
    for i, wp in enumerate(waypoints):
        log.info("  waypoint %d: target = %s", i, _arr_to_str(wp))
        summary = run_phase(
            env,
            controller,
            label=f"waypoint_{i}",
            duration_s=WAYPOINT_HOLD_S,
            target_pos_fn=lambda t, wp=wp: wp,
            target_quat=home_quat,
            csv_writer=csv_writer,
            viewer_real_time=not args.headless,
        )
        log.info(
            "    final pos err = %.2f mm  peak |F| = %.2f N",
            summary["last_pos_err"] * 1000,
            summary["peak_force"],
        )
        summaries.append(summary)

    # ------------------------------------------------------------------
    # Phase 2 — compliance: target inside the wall.
    # Combines approach (slew from waypoint square to wall) + contact + settle.
    # ------------------------------------------------------------------
    log.info("=== Phase 2: Compliance (target 5 cm inside flat wall) ===")
    compliance_target = FLAT_WALL_POS + np.array([COMPLIANCE_INTRUSION, 0.0, 0.0])
    log.info("  target = %s", _arr_to_str(compliance_target))
    summary = run_phase(
        env,
        controller,
        label="compliance",
        duration_s=COMPLIANCE_S,
        target_pos_fn=lambda t: compliance_target,
        target_quat=home_quat,
        csv_writer=csv_writer,
        viewer_real_time=not args.headless,
    )
    log.info(
        "  peak |F| = %.2f N  final state = %s",
        summary["peak_force"],
        summary["final_state"].value,
    )
    summaries.append(summary)

    # ------------------------------------------------------------------
    # Phase 3 — force-trip: keep pushing deeper.
    # ------------------------------------------------------------------
    # The ±2 cm/step command clamp caps the impedance's sustained Cartesian
    # force at K_z · 0.02 (≈ 10 N at K_z=500), so "ramp commanded depth" alone
    # never reaches the 30 N watchdog. Per spec Step 7 we *stiffen the
    # lateral impedance* for this phase instead — the same safety clamp now
    # multiplies through a much larger K.
    log.info("=== Phase 3: Force-trip (stiffen impedance until watchdog) ===")
    force_trip_target = FLAT_WALL_POS + np.array([FORCE_TRIP_INTRUSION, 0.0, 0.0])
    # K_rot scaled up alongside K_xyz: the higher translation gain produces
    # bigger contact reaction moments, and the nominal soft K_rot can't
    # resist them. We restore both after the phase — the stiffened gains
    # are great in contact but excite null-space modes in free space.
    saved_K = controller.stiffness_tcp.copy()
    saved_D = controller.damping_tcp.copy()
    controller.stiffness_tcp = np.array([2000.0, 2000.0, 2000.0, 50.0, 50.0, 50.0])
    controller.damping_tcp = np.array([180.0, 180.0, 180.0, 12.0, 12.0, 12.0])
    log.info(
        "  target = %s  K_xyz=2000 K_rot=50 (vs nominal K_xyz≈400, K_rot=3)",
        _arr_to_str(force_trip_target),
    )
    summary = run_phase(
        env,
        controller,
        label="force_trip",
        duration_s=FORCE_TRIP_S,
        target_pos_fn=lambda t: force_trip_target,
        target_quat=home_quat,
        csv_writer=csv_writer,
        viewer_real_time=not args.headless,
    )
    controller.stiffness_tcp = saved_K
    controller.damping_tcp = saved_D
    log.info(
        "  peak |F| = %.2f N  final state = %s  transitions = %d",
        summary["peak_force"],
        summary["final_state"].value,
        len(summary["lock_changes"]),
    )
    for t, state, reason in summary["lock_changes"]:
        log.info("    t=%.3f -> %s  (%s)", t, state, reason)
    summaries.append(summary)

    # ------------------------------------------------------------------
    # Phase 4 — release + park.
    # Contact during the force-trip phase rotates the gripper ~25 ° off the
    # home orientation. The nominal soft K_rot (=3) can't unwind that
    # within any reasonable time, so we temporarily stiffen rotational
    # impedance for the park slew.
    # ------------------------------------------------------------------
    log.info("=== Phase 4: Release + Park ===")
    controller.release_lock()
    log.info("  after release: state = %s", controller.status.state.value)
    controller.request_park_lock()
    log.info("  after request_park_lock: state = %s", controller.status.state.value)
    summary = run_phase(
        env,
        controller,
        label="park",
        duration_s=PARK_TIMEOUT_S,
        target_pos_fn=lambda t: home_pos,
        target_quat=home_quat,
        csv_writer=csv_writer,
        viewer_real_time=not args.headless,
    )
    obs = env.get_observation()
    final_pos_err = float(np.linalg.norm(obs.ee_pose[:3] - home_pos))
    rot_ax = np.zeros(3)
    import mujoco as _mj  # noqa: PLC0415

    _mj.mju_subQuat(rot_ax, home_quat, obs.ee_pose[3:])
    final_rot_err_deg = float(np.rad2deg(np.linalg.norm(rot_ax)))
    log.info(
        "  final state = %s  pos err from home = %.2f mm  rot err = %.2f°",
        summary["final_state"].value,
        final_pos_err * 1000,
        final_rot_err_deg,
    )
    summaries.append(summary)

    if csv_file is not None:
        csv_file.close()
        log.info("wrote CSV trace: %s", CSV_PATH)

    # ------------------------------------------------------------------
    # Headless assertions.
    # ------------------------------------------------------------------
    if args.headless:
        failures: list[str] = []

        # Waypoint position tolerances.
        for s in summaries[:4]:
            if s["last_pos_err"] > WAYPOINT_POS_TOL:
                failures.append(
                    f"{s['label']}: pos err {s['last_pos_err'] * 1000:.2f} mm > "
                    f"{WAYPOINT_POS_TOL * 1000:.1f} mm"
                )

        # Compliance: peak F must stay below the force cap.
        compliance = summaries[4]
        if compliance["peak_force"] >= controller.force_cap_n:
            failures.append(
                f"compliance phase: peak |F| {compliance['peak_force']:.2f} N >= "
                f"force cap {controller.force_cap_n:.2f} N"
            )

        # Force-trip: watchdog must trip exactly once, end in HOLD.
        force_trip = summaries[5]
        n_trips = sum(1 for _, _, r in force_trip["lock_changes"] if r.startswith("force_cap_trip"))
        if n_trips != 1:
            failures.append(f"force-trip phase: {n_trips} force-cap trips, expected exactly 1")
        if force_trip["final_state"] != LockState.HOLD:
            failures.append(
                f"force-trip phase: final state {force_trip['final_state'].value}, expected hold_lock"
            )

        # Park: end in HOLD at home, within PARK_TIMEOUT_S sim time.
        park = summaries[6]
        if park["final_state"] != LockState.HOLD:
            failures.append(
                f"park phase: final state {park['final_state'].value}, expected hold_lock"
            )
        if final_pos_err > 10 * WAYPOINT_POS_TOL:  # generous: 5 cm
            failures.append(
                f"park phase: pos err from home {final_pos_err * 1000:.2f} mm too large"
            )

        if not CSV_PATH.exists():
            failures.append(f"CSV trace not written: {CSV_PATH}")

        if failures:
            log.error("FAIL — assertion failures:")
            for f in failures:
                log.error("  - %s", f)
            env.close()
            return 1
        log.info("PASS — all M2 acceptance assertions hold.")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
