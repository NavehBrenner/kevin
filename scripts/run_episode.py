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

The same episode (fixed --seed) can be replayed under different assist policies via
--policy {noassist,expert,tf,vision} to compare them head-to-head:

    uv run python scripts/run_episode.py --seed 7 --policy expert
    uv run python scripts/run_episode.py --seed 7 --policy tf --checkpoint runs/train/<run>/checkpoint.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

from ai_teleop.data.schema import EpisodeColumns, EpisodeMetadata

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common import Command  # noqa: E402
from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.common.seating import SeatingGeometry  # noqa: E402
from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.control.lock import LockState  # noqa: E402
from ai_teleop.data.generate import (  # noqa: E402
    DEFAULT_FORCE_CAP,
    DEFAULT_LATERAL_TOLERANCE,
    DEFAULT_MAX_DPOS,
    DEFAULT_SUCCESS_DEPTH,
    episode_terminal_reason,
    make_episode_operator,
)
from ai_teleop.data.trajectory import EpisodeRecorder, TerminalReason, load_episode  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.domain.interfaces import AssistProvider, InputStrategy  # noqa: E402
from ai_teleop.input import (  # noqa: E402
    DEFAULT_MAX_APPROACH_SPEED,
    ScriptedNoisyHuman,
    VisionInput,
    WorkspaceCalibration,
    calibrate_neutral,
)
from ai_teleop.sim.runner import DEFAULT_MAX_STEPS, run_episode  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402
from ai_teleop.sim.scene_source import resolve_scene_path  # noqa: E402

log = get_logger("episode")

_DEFAULT_RECORD_RUNS = Path("data/recorded/runs")


def _resolve_episode_npz(path: str) -> Path:
    """Resolve an episode folder or .npz path to the canonical episode.npz."""
    p = Path(path)
    return p / "episode.npz" if p.is_dir() else p


class _ReplayInput:
    """Feed recorded cmd_* columns back as Commands, one per tick."""

    def __init__(self, columns: dict) -> None:
        self._positions = columns["cmd_position"]
        self._quaternions = columns["cmd_quaternion"]
        self._grips = columns["cmd_grip"]
        self._i = 0

    def get_command(self, observation: object) -> Command:
        i = min(self._i, len(self._positions) - 1)
        self._i += 1
        return Command(
            self._positions[i].copy(), self._quaternions[i].copy(), float(self._grips[i])
        )


def _resolve_record_path(out: str) -> Path:
    """Return the episode.npz path for --record; auto-number when out is empty."""
    if out:
        return Path(out) / "episode.npz"
    existing = (
        [
            int(p.name.split("_")[1])
            for p in _DEFAULT_RECORD_RUNS.iterdir()
            if p.is_dir() and p.name.startswith("episode_")
        ]
        if _DEFAULT_RECORD_RUNS.exists()
        else []
    )
    index = max(existing) + 1 if existing else 0
    return _DEFAULT_RECORD_RUNS / f"episode_{index:05d}" / "episode.npz"


def _build_assist(policy: str, checkpoint: str | None) -> AssistProvider:
    """Resolve --policy to the AssistProvider under test (the correction layer).

    Same scene + operator command stream, different assist — so the same episode
    can be replayed under each policy. Heavy imports (Expert is pure; the trained
    residual pulls in torch) are deferred to the chosen branch.
    """
    if policy == "noassist":
        return NoAssist()
    if policy == "expert":
        from ai_teleop.expert import Expert

        return Expert()
    if policy == "tf":
        if not checkpoint:
            raise SystemExit("--policy tf requires --checkpoint PATH (a trained residual .pt).")
        from ai_teleop.policy import LearnedResidual  # lazy: pulls in torch

        return LearnedResidual.from_checkpoint(checkpoint)
    # ponytail: vision (Phase-2 vision-conditioned residual) isn't trained yet; fail
    # loud rather than silently fall back. Add the branch when the checkpoint exists.
    raise SystemExit(f"--policy {policy} is not implemented yet.")


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
        default=None,
        help="Episode step budget (one step == one 2 ms sim tick). Default: the replayed "
        f"episode's own length when --input is a path, else {DEFAULT_MAX_STEPS}. Use 0 for no "
        "limit — run until you close the viewer or Ctrl-C (handy for free-play with --input vision).",
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
        default="scripted",
        metavar="MODE_OR_PATH",
        help="Base command source: 'scripted' noisy human (default), 'vision' two-webcam "
        "stereo hand tracking (needs the viewer + --stereo-calib), or a path to an episode "
        "folder / episode.npz to replay recorded commands.",
    )
    parser.add_argument(
        "--policy",
        choices=["noassist", "expert", "tf", "vision"],
        default="noassist",
        help="Assist policy layered on the base command: 'noassist' (human-only, default), "
        "'expert' analytical privileged-info supervisor, 'tf' trained F/T residual (needs "
        "--checkpoint), or 'vision' Phase-2 vision-conditioned residual (not implemented yet). "
        "Same scene + operator stream across policies, so you can compare the same episode.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Trained residual checkpoint.pt for --policy tf (e.g. runs/train/<run>/checkpoint.pt).",
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
    parser.add_argument(
        "--record",
        nargs="?",
        const="",
        default=None,
        metavar="OUT",
        help="Record the trajectory to OUT/episode.npz (auto-numbered under data/recorded/ "
        "if OUT is omitted). Stops automatically on a successful insertion.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    # Detect whether --input is a keyword or an episode path.
    replay_columns: EpisodeColumns | None = None
    replay_meta: EpisodeMetadata = {}
    if args.input not in ("scripted", "vision"):
        episode_npz = _resolve_episode_npz(args.input)
        if not episode_npz.exists():
            log.error("--input %r: not 'scripted', 'vision', or a valid episode path.", args.input)
            return 2
        replay_columns, replay_meta = load_episode(episode_npz)
        log.info(
            "Reproducing %s (source=%s, %d recorded steps)",
            episode_npz,
            replay_meta.get("source", "unknown"),
            len(replay_columns["step"]),
        )

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
    # For replay, rebuild the *exact generation scene* from the episode metadata:
    # generation built SimEnv(seed=master_seed, randomize=True).reset(episode_index)
    # (coverage-randomized target hole + joint start). Without matching that, the
    # replay lands on the default home scene with the wrong target hole, so the
    # recorded commands aim into a wall — the episode no longer matches its data.
    reset_index: int | None
    if replay_columns is not None and "scene_seed" in replay_meta:
        scene_seed, reset_index = (int(v) for v in replay_meta["scene_seed"])
        randomize = True
    else:
        scene_seed, reset_index, randomize = args.seed, None, False
    env = SimEnv(str(scene_path), render_mode=render_mode, seed=scene_seed, randomize=randomize)
    observation = env.reset(reset_index)
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
    elif replay_columns is not None:
        # Replay faithfully: use the SAME command clamp generation recorded with.
        # The recorded cmd_* is the raw operator *base* command (the clamp is
        # applied to base+delta inside the controller, exactly as in generation),
        # so re-clamping at the recorded max_dpos reproduces the run — not math.inf.
        recorded_max_dpos = replay_meta.get("max_dpos")
        if args.max_dpos is not None:
            max_dpos = args.max_dpos
        elif recorded_max_dpos is not None:
            max_dpos = float(recorded_max_dpos)
        else:
            max_dpos = DEFAULT_MAX_DPOS
        controller = Controller(env, max_dpos_per_step=max_dpos)
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
    assist = _build_assist(args.policy, args.checkpoint)

    input_strategy: InputStrategy
    tracker = None
    if (
        replay_columns is not None
        and replay_meta.get("source") == "scripted"
        and reset_index is not None
    ):
        # Scripted episode: reconstruct the operator from its seed (deterministic,
        # byte-identical to the recorded stream) and run it LIVE for the full budget.
        # Replaying the recorded commands instead would freeze at the recorded run's
        # terminal step — fine same-policy, but a longer cross-policy run (e.g.
        # noassist over an expert-recorded success) would stall there and never match.
        input_strategy = make_episode_operator(
            target_position,
            home_quaternion,
            seed=scene_seed,
            episode_index=reset_index,
            max_approach_speed=float(
                replay_meta.get("max_approach_speed") or DEFAULT_MAX_APPROACH_SPEED
            ),
        )
    elif replay_columns is not None:
        # Recorded-human episode (no scripted seed): replay the recorded commands.
        input_strategy = _ReplayInput(replay_columns)
    elif args.input == "vision":
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
            "Driving the arm via STEREO hand tracking (metric 3D, 6-DoF). Lift your hand out of frame to clutch."
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
    # Resolve the step budget. Unset (None) defaults, for a replay, to the *generation*
    # budget the episode was produced under (so a cross-policy replay — e.g. noassist
    # over an expert-recorded episode — runs to the same cap, not the recorded outcome
    # length, which would truncate it). Live input falls back to DEFAULT_MAX_STEPS.
    # Explicit 0/negative => run effectively forever; range() is lazy.
    if args.max_steps is not None:
        requested_steps = args.max_steps
    elif replay_columns is not None:
        requested_steps = int(replay_meta.get("max_steps") or len(replay_columns["step"]))
    else:
        requested_steps = DEFAULT_MAX_STEPS
    max_steps = requested_steps if requested_steps > 0 else sys.maxsize
    budget = "unlimited" if requested_steps <= 0 else f"{requested_steps} steps"
    log.info(
        "Running %s with %s input + %s policy (Ctrl-C to stop)...", budget, args.input, args.policy
    )

    recorder: EpisodeRecorder | None = None
    terminal_reason = TerminalReason.TIMEOUT
    record_path: Path | None = None
    step_callback = None

    def _reason(obs, geometry: SeatingGeometry) -> TerminalReason | None:
        """Shared episode-outcome policy — identical to data generation."""
        return episode_terminal_reason(
            penetration=geometry.penetration,
            lateral_error=geometry.lateral_error,
            force_magnitude=float(np.linalg.norm(obs.wrist_ft[:3])),
            locked=controller.status.state is LockState.HOLD,
            success_depth=DEFAULT_SUCCESS_DEPTH,
            lateral_tolerance=DEFAULT_LATERAL_TOLERANCE,
            force_cap=DEFAULT_FORCE_CAP,
        )

    if args.record is not None:
        record_path = _resolve_record_path(args.record)
        ft_bias = observation.wrist_ft.copy()
        recorder = EpisodeRecorder()

        def step_callback(step: int, obs, base_command, delta, command) -> bool:
            nonlocal terminal_reason
            geometry = SeatingGeometry.from_observation(obs)
            reason = _reason(obs, geometry)
            recorder.add(  # type: ignore[union-attr]
                step=step,
                sim_time=obs.sim_time,
                wrist_ft=obs.wrist_ft - ft_bias,
                joint_positions=obs.joint_positions,
                joint_velocities=obs.joint_velocities,
                ee_pose=obs.ee_pose,
                gripper_width=obs.gripper_width,
                cmd_position=base_command.target_position,
                cmd_quaternion=base_command.target_quaternion,
                cmd_grip=base_command.delta_grip_force,
                delta_position=delta.delta_position,
                delta_orientation=delta.delta_orientation,
                delta_grip=delta.delta_grip_force,
                peg_pose=obs.peg_pose,
                target_hole_pose=geometry.target_hole_pose,
                distance=geometry.distance,
                step_success=reason is TerminalReason.SUCCESS,
            )
            if reason is not None:
                terminal_reason = reason
                return True
            return False

        log.info("Recording to: %s", record_path)
    elif args.input != "vision":
        # Scripted / replay (no recording): terminate exactly like data generation
        # (seating, force-cap, or HOLD lock). Vision free-play has no callback — it
        # runs until the viewer closes, so an incidental wall bump won't end it.
        def step_callback(step: int, obs, base_command, delta, command) -> bool:
            nonlocal terminal_reason
            reason = _reason(obs, SeatingGeometry.from_observation(obs))
            if reason is not None:
                terminal_reason = reason
                return True
            return False

    result = None
    try:
        result = run_episode(
            env,
            controller,
            input_strategy,
            assist,
            max_steps=max_steps,
            render=not args.headless,
            step_callback=step_callback,
            # run_episode resets internally; pass the SAME index so it lands on the
            # generation scene (not reset(None)) — else the pre-reset above is discarded.
            reset_episode_index=reset_index,
        )
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        if tracker is not None:
            tracker.close()

    if recorder is not None and len(recorder) > 0:
        assert record_path is not None
        recorder.save(
            record_path,
            metadata={
                "source": args.input,
                "seed": args.seed,
                "target_hole_index": int(observation.target_hole_index),
                "terminal_reason": terminal_reason.value,
                "episode_success": terminal_reason is TerminalReason.SUCCESS,
            },
        )
        log.info("Saved %d steps → %s  [%s]", len(recorder), record_path, terminal_reason.value)

    if result is not None:
        final_dist = float(np.linalg.norm(result.final_observation.ee_pose[:3] - target_position))
        log.info(
            "Episode done: %d steps (%.2f s) · %s · lock=%s · EE-to-hole %.0f mm -> %.0f mm",
            result.n_steps,
            result.final_observation.sim_time,
            terminal_reason.value,
            result.lock_status.state.value,
            start_dist * 1000,
            final_dist * 1000,
        )
    # ponytail: tearing down the MuJoCo viewer / offscreen renderer GL context
    # under WSLg blocks ~10s after the run is already done. Everything durable
    # (the .npz, the logs) is flushed above, so hard-exit instead of waiting on
    # a clean teardown the OS reclaims anyway. main() is only the __main__
    # entrypoint, never imported. Drop to env.close()+return if that changes.
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
