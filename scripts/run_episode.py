"""M3 runner CLI — thin entry point over `ai_teleop.sim.runner`.

The reusable per-episode loop is core functionality and lives in the package
(`ai_teleop.sim.runner.run_episode`); this script is just its command-line front
door (also reachable as `kvn episode`). It builds the concrete no-assist stack
(scene + controller + scripted human aimed at the trial's target hole +
NoAssist) and reports a one-line summary.

Run from the `kevin/` directory:

    uv run python scripts/run_episode.py                 # interactive viewer
    uv run python scripts/run_episode.py --headless      # CI / batch
    uv run python scripts/run_episode.py --headless --seed 7 --max-steps 1500
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.domain.interfaces import InputStrategy  # noqa: E402
from ai_teleop.input import (  # noqa: E402
    ScriptedNoisyHuman,
    VisionInput,
    WorkspaceCalibration,
    calibrate_neutral,
)
from ai_teleop.sim.runner import DEFAULT_MAX_STEPS, run_episode  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402
from ai_teleop.sim.scene_source import resolve_scene_path  # noqa: E402

log = get_logger("episode")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="Skip the viewer; run the loop and print a summary."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the scripted human's noise and the SimEnv."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Episode step budget (one step == one 2 ms sim tick). Use 0 for no limit — "
        "run until you close the viewer or Ctrl-C (handy for free-play with --input vision).",
    )
    parser.add_argument(
        "--generated-wall",
        action="store_true",
        help="Run on a freshly generated procedural wall instead of the static scene.",
    )
    parser.add_argument(
        "--cam",
        choices=["main", "wrist"],
        default="main",
        help="Which camera the interactive viewer opens with: 'main' (free camera, default) "
        "or 'wrist' (locked to the Panda's wrist camera, robot's-eye POV). Viewer keys still "
        "switch cameras live.",
    )
    parser.add_argument(
        "--input",
        choices=["scripted", "vision"],
        default="scripted",
        help="Base command source: 'scripted' noisy human (default) or 'vision' two-webcam "
        "stereo hand tracking (metric 3D + 6-DoF; needs the viewer, the stereo-input extra, "
        "and --stereo-calib).",
    )
    parser.add_argument(
        "--no-cam-window",
        action="store_true",
        help="Hide the live stereo camera/3D-skeleton window (--input vision; shown by default).",
    )
    parser.add_argument(
        "--stereo-calib",
        default=None,
        help="Path to a stereohand stereo_calib.json (required for --input vision). Camera "
        "sources are --left / --right.",
    )
    parser.add_argument(
        "--left",
        default="0",
        help="Left-camera source for --input vision: device index or stream URL.",
    )
    parser.add_argument(
        "--right",
        default="2",
        help="Right-camera source for --input vision: device index or stream URL.",
    )
    parser.add_argument(
        "--max-fps",
        type=int,
        default=None,
        help="Cap hand-tracking to N fps (--input vision), even if the cameras run faster "
        "(~30). Fewer MediaPipe passes = less GIL pressure on the control loop. Default: "
        "no cap (process every new camera frame).",
    )
    parser.add_argument(
        "--orientation",
        action="store_true",
        help="Enable 6-DoF orientation mirroring (--input vision): mirror the hand's "
        "roll/pitch/yaw too. Off by default — position-only is a calmer, round-peg baseline.",
    )
    parser.add_argument("--wall-seed", type=int, default=7, help="Seed for --generated-wall.")
    parser.add_argument(
        "--distractors", type=int, default=None, help="Distractor-hole count for --generated-wall."
    )
    parser.add_argument(
        "--max-dpos",
        type=float,
        default=None,
        help="Controller command clamp in m/step (approach-speed / strictness knob). "
        "Default 0.025 (careful insertion); --input vision defaults to 0.3 for responsive "
        "mirror-like tracking (raise it further if the arm still lags your hand).",
    )
    parser.add_argument(
        "--no-force-cap",
        action="store_true",
        help="Disable the force-cap watchdog (--input vision free-play). Normally a wrist "
        "force above ~30 N latches the arm into a HOLD lock that nothing in the vision path "
        "releases (e.g. after bumping the wall) — looks like the arm 'stops responding'. "
        "Use this to rule the watchdog in/out while debugging.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    if args.input == "vision" and args.headless:
        log.error("--input vision needs the viewer (drop --headless).")
        return 2
    if args.input == "vision" and not args.stereo_calib:
        log.error("--input vision requires --stereo-calib PATH.")
        return 2

    scene_path = resolve_scene_path(
        generated=args.generated_wall,
        wall_seed=args.wall_seed,
        distractors=args.distractors,
    )
    if not scene_path.exists():
        log.error("scene file not found at %s", scene_path)
        return 2

    render_mode = "headless" if args.headless else "viewer"
    log.info("Loading scene (%s): %s", render_mode, scene_path)
    env = SimEnv(str(scene_path), render_mode=render_mode, seed=args.seed)
    observation = env.reset()
    if not args.headless:
        env.launch_viewer(wrist_cam=args.cam == "wrist")
        # Mark the target hole for the human (viewer-only; never in the policy-facing
        # wrist-cam render). Lets the operator know which hole to aim at.
        env.highlight_target(observation.target_hole_position)

    # --input vision wants responsive, mirror-like tracking, not the slew-limited
    # careful-insertion backbone (which feels like velocity control): a bigger
    # command clamp lets the impedance spring toward the hand, and lower joint
    # damping unbounds the ~0.05 m/s free-space slew the default kd=4 caps.
    if args.input == "vision":
        max_dpos = args.max_dpos if args.max_dpos is not None else 0.3
        controller = Controller(
            env,
            max_dpos_per_step=max_dpos,
            joint_damping=1.5,
            force_cap_n=math.inf if args.no_force_cap else 30.0,
        )
    else:
        max_dpos = args.max_dpos if args.max_dpos is not None else 0.025
        controller = Controller(env, max_dpos_per_step=max_dpos)

    # Aim the scripted human at the active trial's hole *position*, but keep the
    # home grasp orientation rather than the hole-site frame: M3 is plumbing, and
    # commanding an arbitrary wrist reorientation would make the crude scripted
    # approach fight the impedance law. Real orientation corrections are the
    # expert's job (M4). The controller's 2 cm/step command clamp turns the
    # full-target command into a smooth bounded approach.
    target_position = observation.target_hole_position.copy()
    home_quaternion = controller.home_pose[3:]
    assist = NoAssist()

    input_strategy: InputStrategy
    tracker = None
    if args.input == "vision":
        from ai_teleop.input.hand_tracker import StereoHandSource

        # A bare integer is a device index; anything else is a stream URL / path.
        def _camera_source(value: str) -> int | str:
            return int(value) if value.isdigit() else value

        tracker = StereoHandSource(
            args.stereo_calib,
            left=_camera_source(args.left),
            right=_camera_source(args.right),
            show_window=not args.no_cam_window,
            max_fps=args.max_fps if args.max_fps is not None else "cam",
        )
        # Startup centering: hold an open palm still to set neutral *before* the sim runs, so
        # the arm holds home (no command piped) until a clean anchor — position and
        # orientation — exists. Ctrl-C here exits without touching the arm.
        try:
            neutral = calibrate_neutral(tracker, on_tick=env.sync_viewer)
        except KeyboardInterrupt:
            log.info("Interrupted during centering.")
            tracker.close()
            env.close()
            return 0
        tracker.set_renderer_origin(neutral.hand_position)  # center the 3D view on neutral
        input_strategy = VisionInput(
            tracker,
            calibration=WorkspaceCalibration(),
            track_orientation=args.orientation,  # opt-in 6-DoF mirroring (off by default)
            initial_anchor=neutral,
        )
        log.info(
            "Driving the arm via STEREO hand tracking (metric 3D, 6-DoF). "
            "Lift your hand out of frame to clutch."
        )
    else:
        target_pose = np.concatenate([target_position, home_quaternion])
        input_strategy = ScriptedNoisyHuman(target_pose, seed=args.seed)

    start_dist = float(np.linalg.norm(observation.ee_pose[:3] - target_position))
    log.info(
        "Target hole %d at %s (%.0f mm from home EE)",
        observation.target_hole_index,
        np.array2string(target_position, precision=3),
        start_dist * 1000,
    )
    # --max-steps 0 (or negative) => run effectively forever; range() is lazy.
    max_steps = args.max_steps if args.max_steps > 0 else sys.maxsize
    budget = "unlimited" if args.max_steps <= 0 else f"{args.max_steps} steps"
    log.info("Running %s with %s input + NoAssist (Ctrl-C to stop)...", budget, args.input)

    result = None
    try:
        result = run_episode(
            env,
            controller,
            input_strategy,
            assist,
            max_steps=max_steps,
            render=not args.headless,
        )
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        if tracker is not None:
            tracker.close()

    if result is not None:
        final_dist = float(np.linalg.norm(result.final_observation.ee_pose[:3] - target_position))
        log.info(
            "Episode done: %d steps  final lock state = %s  EE-to-hole %.0f mm -> %.0f mm",
            result.n_steps,
            result.lock_status.state.value,
            start_dist * 1000,
            final_dist * 1000,
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
