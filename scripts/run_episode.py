"""M3 runner CLI — thin entry point over `ai_teleop.sim.runner`.

The reusable per-episode loop is core functionality and lives in the package
(`ai_teleop.sim.runner.run_episode`); this script is just its command-line front
door (also reachable as `kvn episode`). It composes the concrete stack (scene +
controller + a base command source + a `--policy` correction layer) and reports
a one-line summary. The base commands come from `--input` (scripted noisy human,
stereo hand tracking, or a recorded episode replayed verbatim); the correction
comes from `--policy` (noassist / expert / trained residual). Replaying an
episode rebuilds its exact recorded scene from metadata and runs physics-rate control
(one recorded command per physics step, as generation ran), so `kvn episode --input <ep>`
reproduces its generation to the step — in the viewer too, not just headless. `--time-factor`
caps the sim:wall speed (headless: unbounded; viewer default: real time; <1 slow-mo,
>1 fast-forward).

Run from the `kevin/` directory:

    uv run python scripts/run_episode.py                 # interactive viewer
    uv run python scripts/run_episode.py --headless      # CI / batch
    uv run python scripts/run_episode.py --headless --seed 7 --max-steps 1500
    uv run python scripts/run_episode.py --headless --input runs/episode_00000  # replay
    uv run python scripts/run_episode.py --input runs/episode_00000 --time-factor 0.3  # slow-mo
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_teleop.data.schema import EpisodeColumns, EpisodeMetadata

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common import Command, Observation  # noqa: E402
from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.data.generate import (  # noqa: E402
    DEFAULT_FORCE_CAP,
    DEFAULT_LATERAL_TOLERANCE,
    DEFAULT_SUCCESS_DEPTH,
)
from ai_teleop.data.step_callbacks import EpisodeLogger, TerminationProbe  # noqa: E402
from ai_teleop.data.trajectory import TerminalReason, load_episode  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.domain.interfaces import AssistProvider, InputStrategy  # noqa: E402
from ai_teleop.input import (  # noqa: E402
    DEFAULT_SPEED_LOGNORMAL_SIGMA,
    ScriptedNoisyHuman,
    VisionInput,
    WorkspaceCalibration,
    calibrate_neutral,
)
from ai_teleop.sim.config import EnvConfig  # noqa: E402
from ai_teleop.sim.runner import DEFAULT_MAX_STEPS, SIM_DT, run_episode  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402
from ai_teleop.sim.scene_source import resolve_scene_path  # noqa: E402

log = get_logger("episode")

_DEFAULT_RECORD_RUNS = Path("data/recorded/runs")


def _fast_exit(code: int = 0) -> None:
    """Hard-exit without tearing down the MuJoCo viewer / offscreen renderer GL context.

    That teardown (an un-invoked `env.close()`) blocks ~10s under WSLg; everything
    durable is expected to already be flushed by the caller. `os._exit()` alone still
    isn't enough on Windows: it funnels through the CRT's `_exit()` -> `ExitProcess()`,
    which (unlike a real external kill) runs `DLL_PROCESS_DETACH` for every loaded DLL —
    since the GL context here was never explicitly closed, that unload notification ends
    up doing the same slow teardown we're trying to skip. `TerminateProcess` on our own
    handle skips `DLL_PROCESS_DETACH` entirely (MSDN: "does not notify the DLLs attached
    to the process"), giving the instant exit `os._exit()` was meant to provide.
    """
    logging.shutdown()
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    os._exit(code)


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


def _resolve_record_path(out: str | None) -> Path:
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
        from ai_teleop.data.generate import (
            DEFAULT_EXPERT_BRAKE_GAIN,
            DEFAULT_EXPERT_BRAKE_LEAD_FLOOR,
            DEFAULT_EXPERT_D_FAR,
        )
        from ai_teleop.expert import Expert

        # The corpus operating point (LAB-98): live/replay expert runs brake the
        # approach exactly like the data-gen expert, so "--policy expert" shows
        # the behavior the policy is trained to clone (and the assist actually
        # prevents wall-slams under the deployment controller config).
        return Expert(
            d_far=DEFAULT_EXPERT_D_FAR,
            brake_gain=DEFAULT_EXPERT_BRAKE_GAIN,
            brake_lead_floor=DEFAULT_EXPERT_BRAKE_LEAD_FLOOR,
        )
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

    # Episodes old enough to omit max_dpos predate LAB-96's deployment-config
    # data-gen default (0.3), so the right fallback is the careful-insertion
    # clamp they actually ran under — not data.generate's current default.
    controller_kwargs: dict[str, float] = {"max_dpos_per_step": float(meta.get("max_dpos", 0.025))}
    if "joint_damping" in meta:
        controller_kwargs["joint_damping"] = float(meta["joint_damping"])
    # `force_cap` means two different things depending on who stamped it. Live
    # recordings (this script) stamp the controller *watchdog* (None ⇒ off,
    # --no-force-cap). Datagen episodes — recognizable by their `fingerprint` —
    # stamp the 50 N *episode-terminal* threshold under the same key; their
    # controller always ran the default watchdog. Mapping the 50 N threshold
    # onto the watchdog let a replayed arm sail past the 30 N transients the
    # original run locked on (surfaced by LAB-96's corpus, where ~half the
    # baselines lock).
    if "force_cap" in meta and "fingerprint" not in meta:
        force_cap = meta["force_cap"]
        controller_kwargs["force_cap_n"] = math.inf if force_cap is None else float(force_cap)
    controller = Controller(env, **controller_kwargs)
    return env, observation, controller, scene_path


@dataclass
class EpisodeConfig:
    """Parsed + validated CLI args plus resolved replay state.

    Built once by :func:`build_config`, then handed to the ``build_*`` helpers so each
    stage (env, input, callback) reads from one validated object instead of re-deriving
    ``--input`` flags from the raw argparse Namespace. Thin on purpose — it wraps
    ``args`` rather than mirroring every flag as its own field.
    """

    args: argparse.Namespace
    render_mode: str
    replay_columns: EpisodeColumns | None
    replay_meta: EpisodeMetadata

    @property
    def is_replay(self) -> bool:
        return self.replay_columns is not None

    @property
    def replay_as_baseline(self) -> bool:
        """Replaying `--policy noassist` over an *assisted* scripted recording → regenerate
        the human-only baseline instead of replaying the recorded commands.

        The recorded `cmd_*` stop at the assisted run's terminal step, so a plain replay is
        truncated (the viewer artifact). But the scripted operator is deterministic from
        `human_seed`, and the baseline's length is recorded, so we can reproduce the exact
        human-only rollout the dataset scored. Only for scripted-source recordings that carry
        both `human_seed` and `baseline_n_steps` (older episodes fall back to plain replay).
        """
        meta = self.replay_meta
        return (
            self.is_replay
            and self.args.policy == "noassist"
            and meta.get("policy") not in (None, "noassist")
            and meta.get("source") == "scripted"
            and "human_seed" in meta
            and "baseline_n_steps" in meta
        )


def build_config(args: argparse.Namespace) -> EpisodeConfig:
    """Resolve ``--input`` to a replay-or-live config and reject bad flag combinations.

    Detects whether ``--input`` is a keyword ('scripted'/'vision') or an episode path to
    replay, loading the recorded columns in the latter case. Validation errors log and
    ``raise SystemExit(2)`` — the same exit code the CLI used before, centralized here so
    every downstream helper gets a config it can trust.
    """
    replay_columns: EpisodeColumns | None = None
    replay_meta: EpisodeMetadata = {}
    if args.input not in ("scripted", "vision"):
        episode_npz = _resolve_episode_npz(args.input)
        if not episode_npz.exists():
            log.error("--input %r: not 'scripted', 'vision', or a valid episode path.", args.input)
            raise SystemExit(2)
        replay_columns, replay_meta = load_episode(episode_npz)
        log.info(
            "Replaying %d steps from %s (source=%s)",
            len(replay_columns["step"]),
            episode_npz,
            replay_meta.get("source", "unknown"),
        )

    if args.input == "vision" and args.headless:
        log.error("--input vision needs the viewer (drop --headless).")
        raise SystemExit(2)
    if args.input == "vision" and not args.stereo_calib:
        log.error("--input vision requires --stereo-calib PATH.")
        raise SystemExit(2)
    if args.record_out is not None and args.record is None:
        log.error("--record-out needs --record (nothing is written without a record mode).")
        raise SystemExit(2)

    render_mode = "headless" if args.headless else "viewer"
    return EpisodeConfig(args, render_mode, replay_columns, replay_meta)


def build_env(config: EpisodeConfig) -> tuple[SimEnv, Observation, Controller, Path]:
    """Construct ``(env, observation, controller, scene_path)`` for a live or replay run.

    Replay rebuilds the exact recorded scene + controller from the episode's own metadata
    (see :func:`_rebuild_for_replay`), so ``kvn episode --input <ep>`` reproduces its
    generation to the step regardless of the flags passed here. Live builds from the CLI
    args, with the ``--input vision`` responsiveness tweaks.
    """
    args = config.args
    if config.is_replay:
        log.info("Rebuilding recorded scene (%s) from episode metadata", config.render_mode)
        return _rebuild_for_replay(config.replay_meta, config.render_mode)

    scene_path = resolve_scene_path(
        generated=args.wall_seed is not None,
        wall_seed=args.wall_seed,
        distractors=args.distractors,
    )
    if not scene_path.exists():
        log.error("scene file not found at %s", scene_path)
        raise SystemExit(2)
    log.info("Loading scene (%s): %s", config.render_mode, scene_path)
    env = SimEnv(
        str(scene_path),
        render_mode=config.render_mode,
        config=EnvConfig(wall_seed=args.wall_seed),
    )
    observation = env.reset()
    # --input vision wants responsive, mirror-like tracking, not the slew-limited
    # careful-insertion backbone (which feels like velocity control): a bigger command
    # clamp lets the impedance spring toward the hand, and lower joint damping unbounds
    # the ~0.05 m/s free-space slew the default kd=4 caps.
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
    return env, observation, controller, scene_path


def resolve_target_hole_index(config: EpisodeConfig) -> int:
    """The task's goal hole. Generated walls (and the static scene's home grasp) sit in
    front of hole_0; a replayed episode targets whatever it recorded. The env reports
    every hole's pose but not which is the goal — that's chosen here.
    """
    if config.is_replay and "target_hole_index" in config.replay_meta:
        return int(config.replay_meta["target_hole_index"])
    return 0


def build_input(
    config: EpisodeConfig,
    env: SimEnv,
    target_position: np.ndarray,
    home_quaternion: np.ndarray,
) -> tuple[InputStrategy, Any]:
    """Resolve the base command source: replay columns, stereo vision, or scripted human.

    Returns ``(input_strategy, tracker)``; ``tracker`` is the ``StereoHandSource`` for
    main's ``finally`` to close (``None`` for non-vision). Vision does its startup
    centering here — hold an open palm still to set neutral *before* the sim runs, so the
    arm holds home until a clean anchor exists; Ctrl-C during centering closes cleanly and
    ``raise SystemExit(0)``.
    """
    args = config.args
    if config.replay_as_baseline:
        # Regenerate the deterministic operator (recorded commands stop at the assisted run's
        # terminal — replaying them would truncate the human-only rollout; the viewer artifact).
        seed = int(config.replay_meta["human_seed"])
        log.info(
            "Regenerating the human-only baseline operator (seed=%d) — the recorded commands "
            "stop at the assisted run's terminal, so a plain replay would cut it short.",
            seed,
        )
        target_pose = np.concatenate([target_position, home_quaternion])
        # Rebuild the operator's speed-draw config too (LAB-96): the drawn
        # max_approach_speed comes from the operator's own seeded RNG, so seed +
        # config reproduce it; pre-LAB-96 episodes carry no keys ⇒ draw disabled.
        return ScriptedNoisyHuman(
            target_pose,
            seed=seed,
            speed_lognormal_median=float(config.replay_meta.get("speed_lognormal_median", 0.0)),
            speed_lognormal_sigma=float(
                config.replay_meta.get("speed_lognormal_sigma", DEFAULT_SPEED_LOGNORMAL_SIGMA)
            ),
        ), None
    if config.is_replay:
        assert config.replay_columns is not None
        return _ReplayInput(config.replay_columns), None
    if args.input == "vision":
        from ai_teleop.input.hand_tracker import StereoHandSource

        # A bare integer is a device index; anything else is a stream URL / path.
        def _camera_source(value: str) -> int | str:
            return int(value) if value.isdigit() else value

        tracker = StereoHandSource(
            args.stereo_calib,
            left=_camera_source(args.cameras[0]),
            right=_camera_source(args.cameras[1]),
            show_window=not args.no_cam_window,
            max_fps=args.max_fps if args.max_fps is not None else "cam",
            max_skew_s=args.stereo_max_skew,
        )
        try:
            neutral = calibrate_neutral(tracker, on_tick=env.sync_viewer)
        except KeyboardInterrupt:
            log.info("Interrupted during centering.")
            tracker.close()
            _fast_exit(0)
        tracker.set_renderer_origin(neutral.hand_position)  # center the 3D view on neutral
        input_strategy: InputStrategy = VisionInput(
            tracker,
            calibration=WorkspaceCalibration(),
            track_orientation=args.orientation,  # opt-in 6-DoF mirroring (off by default)
            initial_anchor=neutral,
            dropout_grace_s=args.dropout_grace,
        )
        log.info(
            "Driving the arm via STEREO hand tracking (metric 3D, 6-DoF). Lift your hand out of frame to clutch."
        )
        return input_strategy, tracker
    # Aim the scripted human at the active trial's hole *position*, but keep the home grasp
    # orientation rather than the hole-site frame: M3 is plumbing, and commanding an
    # arbitrary wrist reorientation would make the crude scripted approach fight the
    # impedance law. Real orientation corrections are the expert's job (M4). The
    # controller's command clamp turns the full-target command into a bounded approach.
    target_pose = np.concatenate([target_position, home_quaternion])
    return ScriptedNoisyHuman(target_pose, seed=args.seed), None


def _terminal_callback_extra(recorded_reason: object) -> int:
    """1 if the recorded outcome was probe-fired (success/force_abort), else 0.

    ``run_episode`` evaluates the step_callback *before* each iteration's
    control+physics, so an episode that terminated via the probe ran its firing
    callback on the iteration *after* its last physics step. A replay budgeted
    to exactly the recorded step count never reaches that callback and would
    report every probe-terminated episode as TIMEOUT (surfaced by LAB-96's
    corpus, where ~half the baselines force-abort). The extra iteration only
    runs the callback — the probe fires before any extra physics, so the step
    count and trajectory are untouched. Timeout recordings get no extra
    callback: their generation run never evaluated one past the budget either.
    """
    return 0 if recorded_reason in (None, TerminalReason.TIMEOUT.value) else 1


def resolve_max_steps(config: EpisodeConfig) -> tuple[int, str]:
    """Resolve the step budget, returning ``(max_steps, budget_label)``.

    A replay plays back exactly the recorded commands, so it runs for the recorded length;
    a regenerated baseline (see ``replay_as_baseline``) runs for the recorded baseline length
    so it reproduces the scored human-only rollout; live input falls back to DEFAULT_MAX_STEPS.
    Probe-terminated recordings get one extra iteration — see
    :func:`_terminal_callback_extra`. Explicit 0/negative => run effectively forever
    (``range()`` is lazy, so ``sys.maxsize`` costs nothing).
    """
    args = config.args
    if args.max_steps is not None:
        requested_steps = args.max_steps
    elif config.replay_as_baseline:
        requested_steps = int(config.replay_meta["baseline_n_steps"]) + _terminal_callback_extra(
            config.replay_meta.get("baseline_terminal_reason")
        )
    elif config.is_replay:
        assert config.replay_columns is not None
        requested_steps = len(config.replay_columns["step"]) + _terminal_callback_extra(
            config.replay_meta.get("terminal_reason")
        )
    else:
        requested_steps = DEFAULT_MAX_STEPS
    max_steps = requested_steps if requested_steps > 0 else sys.maxsize
    budget = "unlimited" if requested_steps <= 0 else f"{requested_steps} steps"
    return max_steps, budget


def build_step_callback(
    config: EpisodeConfig,
    controller: Controller,
    observation: Observation,
    target_hole_index: int,
    environment: SimEnv,
) -> tuple[EpisodeLogger | TerminationProbe | None, EpisodeLogger | None, Path | None]:
    """Build the run's step_callback + record bookkeeping.

    The step_callback is the shared data-pipeline plumbing (``data/step_callbacks``), so a
    replay/record run ends exactly where its generated episode did and records identical
    rows: EpisodeLogger (record + terminate) for ``--record``, TerminationProbe (terminate
    only) for scripted/replay, None for vision free-play (runs until the viewer closes, so
    an incidental wall bump won't end it). run_episode uses the DEFAULT_* thresholds.

    Returns ``(step_callback, logger, record_path)``; ``logger``/``record_path`` are None
    unless recording (main reads them back to save the ``.npz``).
    """
    args = config.args
    if args.record is not None:
        record_commands = args.record in ("commands", "all")
        record_images = args.record in ("images", "all")
        record_path = _resolve_record_path(args.record_out)
        imgs_dir: Path | None = None
        if record_images:
            imgs_dir = record_path.parent / "imgs"
            imgs_dir.mkdir(parents=True, exist_ok=True)
        episode_logger = EpisodeLogger(
            observation.wrist_ft.copy(),  # ft_bias
            controller,
            target_hole_index=target_hole_index,
            success_depth=DEFAULT_SUCCESS_DEPTH,
            lateral_tolerance=DEFAULT_LATERAL_TOLERANCE,
            force_cap=DEFAULT_FORCE_CAP,
            render_fn=environment.render_wrist_camera if record_images else None,
            imgs_dir=imgs_dir,
            render_every=args.render_every,
        )
        log.info("Recording (%s) to: %s", args.record, record_path)
        # The logger always drives termination + frame rendering; main only writes the
        # .npz when commands are recorded (images-only leaves just imgs/).
        return episode_logger, (episode_logger if record_commands else None), record_path
    if args.input != "vision":
        probe = TerminationProbe(
            controller,
            target_hole_index=target_hole_index,
            success_depth=DEFAULT_SUCCESS_DEPTH,
            lateral_tolerance=DEFAULT_LATERAL_TOLERANCE,
            force_cap=DEFAULT_FORCE_CAP,
        )
        return probe, None, None
    return None, None, None


def _time_factor(value: str) -> float:
    """Parse --time-factor: a positive float, or 'inf'/'max' for uncapped (as-fast-as-possible)."""
    if value.lower() in ("inf", "max"):
        return math.inf
    factor = float(value)
    if factor <= 0:
        raise argparse.ArgumentTypeError("--time-factor must be > 0 (or 'inf'/'max' for uncapped).")
    return factor


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """Run-behaviour, recording, and controller knobs (headless, budget, clamps)."""
    group = parser.add_argument_group("run", "How the episode runs, records, and is clamped.")
    group.add_argument(
        "--headless", action="store_true", help="Skip the viewer; run the loop and print a summary."
    )
    group.add_argument(
        "--time-factor",
        type=_time_factor,
        default=None,
        metavar="RATIO",
        help="Cap the sim:wall-clock speed ratio (enforced by sleeping; never speeds up a slow "
        "box). Default: unbounded ('inf') when --headless, else 1.0 (real time). Use <1 for slow "
        "motion (e.g. 0.3), >1 to fast-forward, or 'inf'/'max' for as-fast-as-possible.",
    )
    group.add_argument(
        "--seed", type=int, default=0, help="Seed for the scripted human's noise and the SimEnv."
    )
    group.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode step budget (one step == one 2 ms sim tick). Default: the recorded "
        f"length when --input is an episode path (so a replay reproduces it to the step), "
        f"else {DEFAULT_MAX_STEPS}. Use 0 for no limit — run until you close the viewer or "
        "Ctrl-C (handy for free-play with --input vision).",
    )
    group.add_argument(
        "--cam",
        choices=["main", "wrist"],
        default="main",
        help="Which camera the interactive viewer opens with: 'main' (free camera, default) "
        "or 'wrist' (locked to the Panda's wrist camera, robot's-eye POV). Viewer keys still "
        "switch cameras live.",
    )
    group.add_argument(
        "--max-dpos",
        type=float,
        default=None,
        help="Controller command clamp in m/step (approach-speed / strictness knob). "
        "Default 0.025 (careful insertion); --input vision defaults to 0.3 for responsive "
        "mirror-like tracking (raise it further if the arm still lags your hand).",
    )
    group.add_argument(
        "--no-force-cap",
        action="store_true",
        help="Disable the force-cap watchdog (--input vision free-play). Normally a wrist "
        "force above ~30 N latches the arm into a HOLD lock that nothing in the vision path "
        "releases (e.g. after bumping the wall) — looks like the arm 'stops responding'. "
        "Use this to rule the watchdog in/out while debugging.",
    )
    group.add_argument(
        "--profile",
        action="store_true",
        help="Log a per-phase wall-time breakdown of the loop at the end (input / control / "
        "step / render / sleep, ...). To see the true per-step cost run with --time-factor max "
        "so pacing doesn't hide saturation in 'sleep'.",
    )
    group.add_argument(
        "--record",
        choices=["commands", "images", "all"],
        default=None,
        help="Record the episode (off by default). 'commands' saves the trajectory to "
        "episode.npz; 'images' saves wrist-camera PNG frames to an imgs/ folder (the vision "
        "stream the M7 policy is fed); 'all' saves both. Stops automatically on a successful "
        "insertion. Output dir is --record-out.",
    )
    group.add_argument(
        "--record-out",
        default=None,
        metavar="OUT",
        help="Output dir for --record (episode.npz and/or imgs/). Auto-numbered under data/recorded/ if omitted.",
    )
    group.add_argument(
        "--render-every",
        type=int,
        default=1,
        metavar="N",
        help="With --record images/all, save a frame every N recorded steps (cadence knob).",
    )


def add_scene_args(parser: argparse.ArgumentParser) -> None:
    """Scene selection: static task wall vs a freshly generated procedural wall."""
    group = parser.add_argument_group("scene", "Which wall/scene the episode runs on.")
    group.add_argument(
        "--wall-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Run on a freshly generated procedural wall from this seed. Omit for the static task scene.",
    )
    group.add_argument(
        "--distractors",
        type=int,
        default=None,
        help="Distractor-hole count when --wall-seed generates a wall.",
    )


def add_input_args(parser: argparse.ArgumentParser) -> None:
    """Base command source (--input) and its stereo-vision camera options."""
    group = parser.add_argument_group("input", "Where the base commands come from.")
    group.add_argument(
        "--input",
        default="scripted",
        metavar="MODE_OR_PATH",
        help="Base command source: 'scripted' noisy human (default), 'vision' two-webcam "
        "stereo hand tracking (needs the viewer + --stereo-calib), or a path to an episode "
        "folder / episode.npz to replay recorded commands.",
    )
    group.add_argument(
        "--stereo-calib",
        default=None,
        help="Path to a stereohand stereo_calib.json (required for --input vision). Camera sources are --cameras.",
    )
    group.add_argument(
        "--cameras",
        nargs=2,
        default=["0", "2"],
        metavar=("LEFT", "RIGHT"),
        help="Left and right camera sources for --input vision: device indices or stream URLs (default: 0 2).",
    )
    group.add_argument(
        "--no-cam-window",
        action="store_true",
        help="Hide the live stereo camera/3D-skeleton window (--input vision; shown by default).",
    )
    group.add_argument(
        "--max-fps",
        type=int,
        default=None,
        help="Cap hand-tracking to N fps (--input vision), even if the cameras run faster "
        "(~30). Fewer MediaPipe passes = less GIL pressure on the control loop. Default: "
        "no cap (process every new camera frame).",
    )
    group.add_argument(
        "--orientation",
        action="store_true",
        help="Enable 6-DoF orientation mirroring (--input vision): mirror the hand's "
        "roll/pitch/yaw too. Off by default — position-only is a calmer, round-peg baseline.",
    )
    group.add_argument(
        "--dropout-grace",
        type=float,
        default=0.2,
        metavar="SECONDS",
        help="Seconds the hand may vanish before the clutch releases (--input vision). The "
        "arm holds the last command through drops shorter than this; raise it (e.g. 0.4) if a "
        "flaky sensor keeps freezing the arm mid-motion. Default 0.2.",
    )
    group.add_argument(
        "--stereo-max-skew",
        type=float,
        default=0.02,
        metavar="SECONDS",
        help="Max capture-timestamp gap between the two cameras before a frame pair is "
        "dropped (--input vision). The cameras run on independent, uncoordinated capture "
        "threads, so a mismatched USB controller/camera pair can skew well past the default "
        "and get most pairs rejected on timing alone, before MediaPipe ever runs -- measured "
        "(scripts/dev/skew_rejection_probe.py) at 88%% rejected on one pair at the default "
        "vs 7%% at 0.05. If StereoHandSource's sensor-health log line (on close) shows high "
        "drop-out despite good lighting/positioning, measure your skew with that probe and "
        "raise this to match. Default 0.02 (stereohand's own default).",
    )


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    """The correction layer applied on top of the base commands (--policy)."""
    group = parser.add_argument_group("policy", "The correction layer under test.")
    group.add_argument(
        "--policy",
        choices=["noassist", "expert", "tf", "vision"],
        default="noassist",
        help="Correction layer applied on top of the base commands: 'noassist' (default, raw "
        "operator), 'expert' analytical residual, 'tf' trained residual (needs --checkpoint), "
        "'vision' Phase-2 vision-conditioned residual (not implemented). On replay the recorded "
        "commands are the same under every policy — only the correction differs.",
    )
    group.add_argument(
        "--checkpoint",
        default=None,
        help="Trained residual checkpoint .pt for --policy tf (e.g. runs/train/<run>/checkpoint.pt).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Assemble the CLI parser from the per-concern argument-group helpers."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser)
    add_scene_args(parser)
    add_input_args(parser)
    add_policy_args(parser)
    add_logging_arguments(parser)
    return parser


def _log_profile(result) -> None:
    """Log the per-phase wall-time breakdown from a run (--profile).

    Reports each phase's share and per-step cost, plus the achieved loop rate vs the 500 Hz
    physics target. A big 'sleep' share means the loop is keeping up (pacing is idling it) —
    run with --time-factor max to squeeze it out and see the real per-step cost.
    """
    timings = result.step_timings
    total = sum(timings.values())
    if total <= 0:
        return
    steps = result.n_steps
    lines = [
        f"  {phase:9s} {secs:7.3f}s  {secs / total * 100:5.1f}%  {secs / steps * 1e3:6.3f} ms/step"
        for phase, secs in sorted(timings.items(), key=lambda kv: -kv[1])
    ]
    realtime = (steps * SIM_DT) / total
    viewer_fps = result.render_count / total
    log.info(
        "Profile (%d steps, %.2fs wall, %.0f Hz loop, %.2fx real-time, viewer %.1f fps over %d frames):\n%s",
        steps,
        total,
        steps / total,
        realtime,
        viewer_fps,
        result.render_count,
        "\n".join(lines),
    )


def main() -> int:
    args = build_parser().parse_args()
    configure_from_args(args)

    config = build_config(args)
    env, observation, controller, scene_path = build_env(config)
    target_hole_index = resolve_target_hole_index(config)

    if not args.headless:
        env.launch_viewer(wrist_cam=args.cam == "wrist")
        # Mark the target hole for the human (viewer-only; never in the policy-facing
        # wrist-cam render). Lets the operator know which hole to aim at.
        env.highlight_target(observation.hole_poses[target_hole_index][:3])

    target_position = observation.hole_poses[target_hole_index][:3].copy()
    home_quaternion = controller.home_pose[3:]
    assist = _build_assist(args.policy, args.checkpoint)
    input_strategy, tracker = build_input(config, env, target_position, home_quaternion)

    start_dist = float(np.linalg.norm(observation.ee_pose[:3] - target_position))
    log.info(
        "Target hole %d at %s (%.0f mm from home EE)",
        target_hole_index,
        np.array2string(target_position, precision=3),
        start_dist * 1000,
    )
    max_steps, budget = resolve_max_steps(config)
    log.info(
        "Running %s with %s input + %s policy (Ctrl-C to stop)...",
        budget,
        args.input,
        args.policy,
    )

    step_callback, logger, record_path = build_step_callback(
        config, controller, observation, target_hole_index, env
    )

    # Pacing: unbounded headless, real time in the viewer, unless --time-factor overrides.
    # The loop is always physics-rate (one command per physics step), so a viewer replay
    # reproduces its recording to the step regardless of source or pacing.
    time_factor = (
        args.time_factor if args.time_factor is not None else (math.inf if args.headless else 1.0)
    )
    result = None
    try:
        result = run_episode(
            env,
            controller,
            input_strategy,
            assist,
            max_steps=max_steps,
            render=not args.headless,
            time_factor=time_factor,
            step_callback=step_callback,
        )
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        if tracker is not None:
            tracker.close()

    # The callback latched why the episode ended (TIMEOUT if none was set / vision).
    terminal_reason = (
        step_callback.terminal_reason if step_callback is not None else TerminalReason.TIMEOUT
    )

    if logger is not None and len(logger.recorder) > 0:
        assert record_path is not None
        logger.recorder.save(
            record_path,
            # Complete replay spec so the episode reproduces later: _rebuild_for_replay
            # reads these back to reconstruct the exact scene + controller. force_cap
            # None ⇒ the watchdog was off (--no-force-cap).
            metadata={
                "source": args.input,
                "policy": args.policy,
                "seed": args.seed,
                "target_hole_index": target_hole_index,
                "generated_wall": args.wall_seed is not None,
                "wall_seed": args.wall_seed,
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
        log.info(
            "Saved %d steps → %s  [%s]", len(logger.recorder), record_path, terminal_reason.value
        )

    if result is not None:
        final_dist = float(np.linalg.norm(result.final_observation.ee_pose[:3] - target_position))
        log.info(
            "Episode done: %d steps (%.2f s sim)  [%s]  final lock state = %s  EE-to-hole %.0f mm -> %.0f mm",
            result.n_steps,
            result.final_observation.sim_time,
            terminal_reason.value,
            result.lock_status.state.value,
            start_dist * 1000,
            final_dist * 1000,
        )
        if args.profile and result.n_steps > 0:
            _log_profile(result)
    # ponytail: tearing down the MuJoCo viewer / offscreen renderer GL context under
    # WSLg blocks ~10s after the run is already done. Everything durable (the .npz,
    # the logs) is flushed above, so hard-exit instead — see _fast_exit(). main() is
    # only the __main__ entrypoint, never imported. Drop to env.close()+return if
    # that changes.
    _fast_exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
