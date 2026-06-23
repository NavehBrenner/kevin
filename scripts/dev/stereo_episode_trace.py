"""Trace why the stereo teleop arm isn't moving (LAB-74 debug).

Runs the real stereo episode wiring (SimEnv + Controller + StereoHandSource +
VisionInput, metric calibration, track_orientation=True) headless, and logs per
tick: lock state, wrist-force magnitude, and how far the *commanded* target sits
from the actual EE. That distinguishes the candidate causes:

- lock state flips to `hold_lock` (and |F| spiked >30 N just before) -> the
  force-cap watchdog tripped (likely the orientation jump from track_orientation),
  and the arm is latched. THIS is the freeze.
- lock stays `active` but |target-ee| ~ 0 -> VisionInput isn't producing motion
  (mapping/clutch), look upstream.
- lock `active` and |target-ee| > 0 but EE doesn't move -> impedance/actuation.

Usage (same URLs/calib as kvn):
    cd kevin
    .venv/bin/python scripts/dev/stereo_episode_trace.py \
        --calib ../stereohand/stereo_calib.json \
        --left "http://$WIN:8080/0" --right "http://$WIN:8080/1"

Move your hand the whole time. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.input import VisionInput, WorkspaceCalibration  # noqa: E402
from ai_teleop.input.hand_tracker import StereoHandSource  # noqa: E402
from ai_teleop.sim.runner import run_episode  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402
from ai_teleop.sim.scene_source import resolve_scene_path  # noqa: E402


def _source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib", required=True)
    p.add_argument("--left", default="0")
    p.add_argument("--right", default="2")
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--every", type=int, default=100, help="log every N control ticks")
    p.add_argument(
        "--no-orientation",
        action="store_true",
        help="disable track_orientation to A/B test the force-cap-trip hypothesis",
    )
    p.add_argument("--mode", choices=["mirror", "expo", "rate"], default="expo")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--headless", action="store_true", help="no viewer/cam window (just the log)")
    args = p.parse_args()

    render = not args.headless
    scene_path = resolve_scene_path(generated=False, wall_seed=7, distractors=None)
    env = SimEnv(str(scene_path), render_mode="viewer" if render else "headless", seed=0)
    env.reset()
    if render:
        env.launch_viewer()
    controller = Controller(env, max_dpos_per_step=0.08, joint_damping=1.5)

    tracker = StereoHandSource(
        args.calib, left=_source(args.left), right=_source(args.right), show_window=render
    )
    vision = VisionInput(
        tracker,
        calibration=WorkspaceCalibration(),
        mode=args.mode,
        gain=args.gain,
        track_orientation=not args.no_orientation,
    )

    ee_start: np.ndarray | None = None
    target_start: np.ndarray | None = None
    max_ee_disp = 0.0

    def trace(step, observation, base_command, delta, command) -> bool:  # noqa: ANN001
        nonlocal ee_start, target_start, max_ee_disp
        ee = observation.ee_pose[:3]
        target = base_command.target_position
        if ee_start is None:
            ee_start = ee.copy()
            target_start = target.copy()
        ee_disp = float(np.linalg.norm(ee - ee_start))
        target_disp = float(np.linalg.norm(target - target_start))
        max_ee_disp = max(max_ee_disp, ee_disp)
        if step % args.every == 0:
            print(
                f"step {step:5d} | lock={controller.status.state.value:9s} "
                f"| EE moved {ee_disp * 1000:6.1f}mm (peak {max_ee_disp * 1000:6.1f}) "
                f"| target moved {target_disp * 1000:6.1f}mm"
            )
        return False

    print(f"mode={args.mode} gain={args.gain} track_orientation={not args.no_orientation}")
    try:
        run_episode(
            env,
            controller,
            vision,
            NoAssist(),
            max_steps=args.max_steps,
            render=render,
            step_callback=trace,
        )
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        tracker.close()
    print(
        f"final lock state: {controller.status.state.value} ({controller.status.last_transition_reason})"
    )


if __name__ == "__main__":
    main()
