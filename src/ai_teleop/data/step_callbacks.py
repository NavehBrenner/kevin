"""Shared `run_episode` step-callbacks and the episode-outcome policy.

The per-tick loop (`ai_teleop.sim.runner.run_episode`) is deliberately
data-agnostic; callers bolt behavior on through its ``step_callback`` hook. Two
callbacks recur across the data pipeline and the ``kvn episode`` run CLI:

* :class:`EpisodeLogger` — record each row + detect termination (+ optional
  wrist-cam render), used by data generation's expert pass and by
  ``run_episode --record``.
* :class:`TerminationProbe` — score termination only (no recording), used by
  generation's paired human-only baseline and by the scripted/replay run path.

Both share :func:`episode_terminal_reason` — the single "why an episode ends"
policy — so generation and replay can't drift on when an episode is "over" (the
structural cause of replays not matching their generated episode).

**Stays in ``data/``, never ``sim/``.** ``EpisodeLogger`` needs
``EpisodeRecorder`` (``data.trajectory``); housing it in ``sim/`` would make
``sim → data`` and cycle against the existing ``data → sim.runner`` edge.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ai_teleop.common.observation import Observation
from ai_teleop.common.seating import SeatingGeometry
from ai_teleop.control import Controller, LockState
from ai_teleop.data.trajectory import EpisodeRecorder, TerminalReason
from ai_teleop.domain import Delta


class _SeatingMetrics:
    """Seating geometry plus the force read needed to decide termination.

    The geometry (penetration / lateral error / distance) comes from the shared
    ``common.seating`` definition so generation and the M6 eval harness cannot
    drift on what "seated" means; the force magnitude is data-generation's own
    concern. The termination *policy* is the module-level
    :func:`episode_terminal_reason` so any caller can reuse it without coupling to
    this private struct.
    """

    def __init__(self, observation: Observation, target_hole_index: int) -> None:
        geometry = SeatingGeometry.from_observation(observation, target_hole_index)
        self.hole_pose = geometry.target_hole_pose
        self.distance = geometry.distance
        self.lateral_error = geometry.lateral_error
        self.penetration = geometry.penetration
        self.force_magnitude = float(np.linalg.norm(observation.wrist_ft[:3]))


def episode_terminal_reason(
    *,
    penetration: float,
    lateral_error: float,
    force_magnitude: float,
    locked: bool,
    success_depth: float,
    lateral_tolerance: float,
    force_cap: float,
) -> TerminalReason | None:
    """The single episode-outcome policy — why a step ends the episode, or ``None``.

    Shared by data generation *and* the ``kvn episode`` run CLI so the two can't
    drift on when an episode is "over" (the structural cause of replays not
    matching their generated episode). SUCCESS once seated; FORCE_ABORT if the
    controller's force-cap watchdog has latched HOLD (``locked`` — the arm is
    frozen, so further steps are dead frames) or the raw wrist force exceeds
    ``force_cap``; else ``None`` to keep going. Takes primitives (not the private
    ``_SeatingMetrics``) so any caller can use it without that coupling.
    """
    if penetration >= success_depth and lateral_error < lateral_tolerance:
        return TerminalReason.SUCCESS
    if locked or force_magnitude > force_cap:
        return TerminalReason.FORCE_ABORT
    return None


_JPEG_QUALITY = 90  # visually near-lossless at 224x224 for this scene; ~55% of PNG size.


def _save_frame(imgs_dir: Path, step: int, frame: np.ndarray) -> None:
    """Write one wrist-camera frame as ``imgs/step_NNNNN.jpg``.

    PIL is imported lazily so the default (no-render) path needs nothing beyond
    numpy; only ``render_images`` pulls in the imaging stack.
    """
    from PIL import Image

    Image.fromarray(frame).save(imgs_dir / f"step_{step:05d}.jpg", quality=_JPEG_QUALITY)


class EpisodeLogger:
    """`run_episode` step_callback that records rows and detects termination.

    When ``render_fn`` and ``imgs_dir`` are supplied (``render_images``), it also
    renders the wrist camera every ``render_every`` recorded steps and saves the
    frame into the episode's ``imgs/`` folder. This is opt-in M7 plumbing: the
    F/T-only M5 corpus is generated with rendering off, and ``render_every`` is
    the cadence knob M7 will calibrate (1 ⇒ a frame per trajectory row).
    """

    def __init__(
        self,
        ft_bias: np.ndarray,
        controller: Controller,
        *,
        target_hole_index: int,
        success_depth: float,
        lateral_tolerance: float,
        force_cap: float,
        render_fn: Callable[[], np.ndarray] | None = None,
        imgs_dir: Path | None = None,
        render_every: int = 1,
    ) -> None:
        self.recorder = EpisodeRecorder()
        self.terminal_reason = TerminalReason.TIMEOUT
        self._ft_bias = ft_bias
        self._controller = controller
        self._target_hole_index = target_hole_index
        self._success_depth = success_depth
        self._lateral_tolerance = lateral_tolerance
        self._force_cap = force_cap
        self._render_fn = render_fn
        self._imgs_dir = imgs_dir
        self._render_every = render_every
        # Render throughput — offscreen rendering is ~500x a physics step
        # (project-wiki/entities/mujoco.md), so the full-corpus render cost is worth
        # tracking explicitly rather than inferred from total episode wall-time.
        self.frames_rendered = 0
        self.render_wall_time = 0.0

    def __call__(
        self,
        step: int,
        observation: Observation,
        base_command,
        delta: Delta,
        command,
    ) -> bool:
        metrics = _SeatingMetrics(observation, self._target_hole_index)
        reason = episode_terminal_reason(
            penetration=metrics.penetration,
            lateral_error=metrics.lateral_error,
            force_magnitude=metrics.force_magnitude,
            locked=self._controller.status.state is LockState.HOLD,
            success_depth=self._success_depth,
            lateral_tolerance=self._lateral_tolerance,
            force_cap=self._force_cap,
        )

        if (
            self._render_fn is not None
            and self._imgs_dir is not None
            and step % self._render_every == 0
        ):
            render_start = time.perf_counter()
            _save_frame(self._imgs_dir, step, self._render_fn())
            self.render_wall_time += time.perf_counter() - render_start
            self.frames_rendered += 1

        self.recorder.add(
            step=step,
            sim_time=observation.sim_time,
            wrist_ft=observation.wrist_ft - self._ft_bias,  # bias-subtracted
            joint_positions=observation.joint_positions,
            joint_velocities=observation.joint_velocities,
            ee_pose=observation.ee_pose,
            gripper_width=observation.gripper_width,
            cmd_position=base_command.target_position,
            cmd_quaternion=base_command.target_quaternion,
            cmd_grip=base_command.delta_grip_force,
            delta_position=delta.delta_position,
            delta_orientation=delta.delta_orientation,
            delta_grip=delta.delta_grip_force,
            peg_pose=observation.peg_pose,
            target_hole_pose=metrics.hole_pose,
            distance=metrics.distance,
            step_success=reason is TerminalReason.SUCCESS,
        )

        if reason is not None:
            self.terminal_reason = reason
            return True
        return False


class TerminationProbe:
    """`run_episode` step_callback that scores termination without recording.

    Used for the paired human-only baseline: it reuses the exact
    :class:`EpisodeLogger` seating logic but skips the trajectory recorder, so the
    baseline rollout is a cheap scoring pass over the same scene and operator
    stream.
    """

    def __init__(
        self,
        controller: Controller,
        *,
        target_hole_index: int,
        success_depth: float,
        lateral_tolerance: float,
        force_cap: float,
    ) -> None:
        self.terminal_reason = TerminalReason.TIMEOUT
        self._controller = controller
        self._target_hole_index = target_hole_index
        self._success_depth = success_depth
        self._lateral_tolerance = lateral_tolerance
        self._force_cap = force_cap

    def __call__(
        self, step: int, observation: Observation, base_command, delta: Delta, command
    ) -> bool:
        metrics = _SeatingMetrics(observation, self._target_hole_index)
        reason = episode_terminal_reason(
            penetration=metrics.penetration,
            lateral_error=metrics.lateral_error,
            force_magnitude=metrics.force_magnitude,
            locked=self._controller.status.state is LockState.HOLD,
            success_depth=self._success_depth,
            lateral_tolerance=self._lateral_tolerance,
            force_cap=self._force_cap,
        )
        if reason is not None:
            self.terminal_reason = reason
            return True
        return False
