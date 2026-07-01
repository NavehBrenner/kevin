"""M3 runner CLI — thin entry point over `ai_teleop.sim.runner`.

The reusable per-episode loop is core functionality and lives in the package
(`ai_teleop.sim.runner.run_episode`); this script is just its command-line front
door (also reachable as `kvn episode`). It composes the concrete stack (scene +
controller + a base command source + a `--policy` correction layer) and reports
a one-line summary. The base commands come from `--input` (scripted noisy human,
stereo hand tracking, or a recorded episode replayed verbatim); the correction
comes from `--policy` (noassist / expert / trained residual). Replaying an
episode rebuilds its exact recorded scene from metadata, so `kvn episode
--input <ep>` reproduces its generation to the step.

Run from the `kevin/` directory:

    uv run python scripts/run_episode.py                 # interactive viewer
    uv run python scripts/run_episode.py --headless      # CI / batch
    uv run python scripts/run_episode.py --headless --seed 7 --max-steps 1500
    uv run python scripts/run_episode.py --headless --input runs/episode_00000  # replay
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
from ai_teleop.control import Controller, LockState  # noqa: E402
from ai_teleop.data.generate import (  # noqa: E402
    DEFAULT_FORCE_CAP,
    DEFAULT_LATERAL_TOLERANCE,
    DEFAULT_MAX_DPOS,
    DEFAULT_SUCCESS_DEPTH,
    episode_terminal_reason,
)
from ai_teleop.data.trajectory import EpisodeRecorder, TerminalReason, load_episode  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.domain.interfaces import AssistProvider, InputStrategy  # noqa: E402
from ai_teleop.input import (  # noqa: E402
    ScriptedNoisyHuman,
    VisionInput,
    WorkspaceCalibration,
    calibrate_neutral,
)
from ai_teleop.sim.config import EnvConfig  # noqa: E402
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
    """Resolve --policy to the AssistProvider (the correction layer) under test.

    Same scene + operator command stream, different assist — so one recorded episode
    can be replayed under each policy. Heavy imports (Expert is pure Python; the
    trained residual pulls in torch) are deferred to the chosen branch.
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


def _rebuild_for_replay(meta: EpisodeMetadata, render_mode):
    """Rebuild the exact scene + controller a recorded episode ran in, from its metadata.

    Replay reproduces the recording, so the scene (static vs generated wall + seed) and
    the controller (command clamp, joint damping, force cap) must match what was recorded —
    otherwise the replayed commands drive a different physical episode (the bug: the scene
    was rebuilt from CLI args, not the episode's own spec). Everything is keyed on
    ``wall_seed`` + arg-less ``reset()``; older episodes fall back to sensible defaults.

    Returns ``(env, observation, controller, scene_path)``.
    """
    generated = bool(meta.get("generated_wall", False))
    wall_seed = meta.get("wall_seed")
    scene_path = resolve_scene_path(
        generated=generated,
        wall_seed=wall_seed,
        distractors=meta.get("distractors"),
    )
    env = SimEnv(
        str(scene_path),
        render_mode=render_mode,
        config=EnvConfig(wall_seed=wall_seed if generated else None),
    )
    observation = env.reset()

    controller_kwargs: dict[str, float] = {
        "max_dpos_per_step": float(meta.get("max_dpos", DEFAULT_MAX_DPOS))
    }
    if "joint_damping" in meta:
        controller_kwargs["joint_damping"] = float(meta["joint_damping"])
    if "force_cap" in meta:  # stored None means the watchdog was off (--no-force-cap)
        force_cap = meta["force_cap"]
        controller_kwargs["force_cap_n"] = math.inf if force_cap is None else float(force_cap)
    controller = Controller(env, **controller_kwargs)
    return env, observation, controller, scene_path


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
        help="Episode step budget (one step == one 2 ms sim tick). Default: the recorded "
        f"length when --input is an episode path (so a replay reproduces it to the step), "
        f"else {DEFAULT_MAX_STEPS}. Use 0 for no limit — run until you close the viewer or "
        "Ctrl-C (handy for free-play with --input vision).",
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
        help="Correction layer applied on top of the base commands: 'noassist' (default, raw "
        "operator), 'expert' analytical residual, 'tf' trained residual (needs --checkpoint), "
        "'vision' Phase-2 vision-conditioned residual (not implemented). On replay the recorded "
        "commands are the same under every policy — only the correction differs.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Trained residual checkpoint .pt for --policy tf "
        "(e.g. runs/train/<run>/checkpoint.pt).",
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
            "Replaying %d steps from %s (source=%s)",
            len(replay_columns["step"]),
            episode_npz,
            replay_meta.get("source", "unknown"),
        )

    if args.input == "vision" and args.headless:
        log.error("--input vision needs the viewer (drop --headless).")
        return 2
    if args.input == "vision" and not args.stereo_calib:
        log.error("--input vision requires --stereo-calib PATH.")
        return 2

    render_mode = "headless" if args.headless else "viewer"

    if replay_columns is not None:
        # Replay: reconstruct the exact scene + controller from the episode's own
        # stored spec (NOT the CLI args), so `kvn episode --input <ep>` reproduces
        # its generation to the step regardless of the flags passed here.
        log.info("Rebuilding recorded scene (%s) from episode metadata", render_mode)
        env, observation, controller, scene_path = _rebuild_for_replay(replay_meta, render_mode)
    else:
        scene_path = resolve_scene_path(
            generated=args.generated_wall,
            wall_seed=args.wall_seed,
            distractors=args.distractors,
        )
        if not scene_path.exists():
            log.error("scene file not found at %s", scene_path)
            return 2
        log.info("Loading scene (%s): %s", render_mode, scene_path)
        env = SimEnv(
            str(scene_path),
            render_mode=render_mode,
            config=EnvConfig(wall_seed=args.wall_seed if args.generated_wall else None),
        )
        observation = env.reset()
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

    # The task's goal hole. Generated walls (and the static scene's home grasp)
    # sit in front of hole_0; a replayed episode targets whatever it recorded. The
    # env reports every hole's pose but not which is the goal — that's chosen here.
    if replay_columns is not None and "target_hole_index" in replay_meta:
        target_hole_index = int(replay_meta["target_hole_index"])
    else:
        target_hole_index = 0

    if not args.headless:
        env.launch_viewer(wrist_cam=args.cam == "wrist")
        # Mark the target hole for the human (viewer-only; never in the policy-facing
        # wrist-cam render). Lets the operator know which hole to aim at.
        env.highlight_target(observation.hole_poses[target_hole_index][:3])

    # Aim the scripted human at the active trial's hole *position*, but keep the
    # home grasp orientation rather than the hole-site frame: M3 is plumbing, and
    # commanding an arbitrary wrist reorientation would make the crude scripted
    # approach fight the impedance law. Real orientation corrections are the
    # expert's job (M4). The controller's 2 cm/step command clamp turns the
    # full-target command into a smooth bounded approach.
    target_position = observation.hole_poses[target_hole_index][:3].copy()
    home_quaternion = controller.home_pose[3:]
    assist = _build_assist(args.policy, args.checkpoint)

    input_strategy: InputStrategy
    tracker = None
    if replay_columns is not None:
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
        target_hole_index,
        np.array2string(target_position, precision=3),
        start_dist * 1000,
    )
    # Resolve the step budget. A replay plays back exactly the recorded commands, so it
    # runs for the recorded length; live input falls back to DEFAULT_MAX_STEPS. Explicit
    # 0/negative => run effectively forever; range() is lazy.
    if args.max_steps is not None:
        requested_steps = args.max_steps
    elif replay_columns is not None:
        requested_steps = len(replay_columns["step"])
    else:
        requested_steps = DEFAULT_MAX_STEPS
    max_steps = requested_steps if requested_steps > 0 else sys.maxsize
    budget = "unlimited" if requested_steps <= 0 else f"{requested_steps} steps"
    log.info(
        "Running %s with %s input + %s policy (Ctrl-C to stop)...",
        budget,
        args.input,
        args.policy,
    )

    recorder: EpisodeRecorder | None = None
    terminal_reason = TerminalReason.TIMEOUT
    record_path: Path | None = None
    step_callback = None

    def _reason(obs, geometry: SeatingGeometry) -> TerminalReason | None:
        """Shared episode-outcome policy — identical to data generation, so a
        replay ends exactly where its generated episode did (no path divergence)."""
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
            geometry = SeatingGeometry.from_observation(obs, target_hole_index)
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
        # Scripted / replay without recording: terminate exactly like data
        # generation (seating, force-cap, or HOLD lock). Vision free-play has no
        # callback — it runs until the viewer closes, so an incidental wall bump
        # won't end it.
        def step_callback(step: int, obs, base_command, delta, command) -> bool:
            nonlocal terminal_reason
            reason = _reason(obs, SeatingGeometry.from_observation(obs, target_hole_index))
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
            # Complete replay spec so the episode reproduces later: _rebuild_for_replay
            # reads these back to reconstruct the exact scene + controller. force_cap
            # None ⇒ the watchdog was off (--no-force-cap).
            metadata={
                "source": args.input,
                "policy": args.policy,
                "seed": args.seed,
                "target_hole_index": target_hole_index,
                "generated_wall": args.generated_wall,
                "wall_seed": args.wall_seed if args.generated_wall else None,
                "distractors": args.distractors,
                "scene": scene_path.name,
                "max_dpos": controller.max_dpos_per_step,
                "joint_damping": controller.joint_damping,
                "force_cap": (
                    None if math.isinf(controller.force_cap_n) else controller.force_cap_n
                ),
                "success_depth": DEFAULT_SUCCESS_DEPTH,
                "lateral_tolerance": DEFAULT_LATERAL_TOLERANCE,
                "terminal_reason": terminal_reason.value,
                "episode_success": terminal_reason is TerminalReason.SUCCESS,
            },
        )
        log.info("Saved %d steps → %s  [%s]", len(recorder), record_path, terminal_reason.value)

    if result is not None:
        final_dist = float(np.linalg.norm(result.final_observation.ee_pose[:3] - target_position))
        log.info(
            "Episode done: %d steps (%.2f s sim)  [%s]  final lock state = %s  "
            "EE-to-hole %.0f mm -> %.0f mm",
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
