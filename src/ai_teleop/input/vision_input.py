"""VisionInput — hand-pose readings → base EE Command (LAB-51).

The headline M8 teleop logic: turn the stereo sensor's :class:`HandReading`
stream (`hand_tracker.py`) into a base :class:`Command` behind the existing
:class:`~ai_teleop.domain.interfaces.InputStrategy` seam, so it drops into the
runner with no upstream/downstream change.

Four pieces, all the deferred "still open" calibration work from
`project-scope.md`:

- **Relative mapping + clutch.** Hand motion is mapped *incrementally*: while
  engaged, ``EE = anchor_EE + scale ⊙ remap(hand − anchor_hand)``. Lifting the
  hand out of frame (sensor ``present=False``) disengages and holds the last
  command; bringing it back re-anchors at the current EE pose. That re-anchoring
  *is* the clutch — lift out, reposition comfortably, drop back in, continue —
  and it means absolute camera origin never needs calibrating, only per-axis
  scale and axis remap/flip.
- **One-euro filter** on the mapped position to kill webcam tremor (a low-pass
  whose cutoff rises with speed: smooth when still, responsive when moving).
- **Grip.** The open/close scalar maps to ``Command.delta_grip_force``.
- **Orientation.** Stereo makes the hand orientation observable, so 6-DoF
  mirroring is usable (``track_orientation=True`` — what the stereo CLI path
  enables). The class default stays ``False`` (round peg ⇒ roll irrelevant, and a
  calmer translation-only baseline); when on, the held orientation follows the
  (filtered) hand orientation.

Live two-camera use is manual; the deterministic math here (mapping transform,
one-euro response, clutch/drop-out state machine) is unit-tested with synthetic
readings.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.geometry import quat_conjugate, quat_mul
from ai_teleop.common.log import console_stream, get_logger
from ai_teleop.common.observation import Observation

from .hand_tracker import HandReading

log = get_logger("vision_input")

# Braille spinner frames for the live centering line (TTY only).
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# calibrate_neutral's poll loop runs at ~200 Hz (poll_interval_s=0.005); on_tick (typically
# SimEnv.sync_viewer) is throttled to this wall-clock cadence rather than firing every poll,
# matching the run loop's own render-cadence throttle (sim/runner.py's render_fps). An
# unthrottled 200 Hz of viewer syncs floods MuJoCo's internal render thread and starves the
# stereohand tracker thread of GIL time right when the operator most needs a responsive
# calibration read — see project-wiki/concepts/realtime-teleop-loop.md.
_ON_TICK_INTERVAL_S = 1.0 / 30.0


class _HandSource(Protocol):
    """Anything that yields the latest hand reading — the live tracker or a fake."""

    def read(self) -> HandReading: ...


@dataclass(frozen=True)
class NeutralAnchor:
    """Operator-defined neutral captured at startup (see :func:`calibrate_neutral`).

    The hand pose that maps to the arm's home EE: feeding it to ``VisionInput`` as
    ``initial_anchor`` means the arm sits still at home until the operator moves their
    hand *relative* to this pose — no startup jump from anchoring on a bad first frame.
    """

    hand_position: np.ndarray
    hand_orientation: np.ndarray


def calibrate_neutral(
    source: _HandSource,
    *,
    hold_s: float = 3.0,
    move_tol: float = 0.02,
    pose_grace_s: float = 0.3,
    poll_interval_s: float = 0.005,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_tick: Callable[[], None] | None = None,
) -> NeutralAnchor:
    """Block until an open palm is held still for ``hold_s``; return the averaged neutral.

    The startup centering step: run *before the sim is stepped* (wall-clock timed, not
    ``sim_time``) so no command reaches the arm until a clean neutral exists. Reads ``source``
    in a loop, requiring ``reading.recenter_pose`` with under ``move_tol`` drift from the
    hold's start; drifting too far restarts. Returns the **mean** hand position over the hold
    (a stable zero, immune to single-frame jitter) and the settled orientation.

    A lost pose (sensor drop-out *or* a flickered ``recenter_pose``) only restarts the hold
    once it's been gone for ``pose_grace_s`` — within that window the hold is kept and the
    countdown keeps running, mirroring stereohand's renderer presence window so a single
    low-confidence frame doesn't make the operator start the 3 s hold over.

    ``on_tick`` runs every iteration — use it to pump the preview window / sync the viewer so
    both stay responsive while the operator poses. A per-second countdown is logged.
    """
    hold_start: float | None = None
    anchor: np.ndarray | None = None
    positions: list[np.ndarray] = []
    last_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    last_countdown: int | None = None
    last_good_time: float | None = None  # clock() of the last present open-palm frame
    last_tick_time: float | None = None  # clock() of the last on_tick fire

    # On a real terminal, draw one self-overwriting status line with a live
    # spinner so the operator sees it's working; otherwise fall back to the
    # per-state log lines (clean redirected output, and what the tests read).
    # Both paths go through console_stream(), not the live sys.stderr object: a vision
    # tracker is already constructed by the time this runs, and its HandLandmarker session
    # has the OS-level stderr fd redirected to a log file for its whole lifetime (see
    # console_stream()'s docstring) — writing straight to sys.stderr here would silently
    # vanish into that file instead of reaching the operator's terminal.
    console = console_stream()
    live = console.isatty()
    last_render = ""

    def show(now: float, text: str, *, done: bool = False) -> None:
        # The poll loop runs at ~200 Hz; the spinner frame only changes at 10 Hz.
        # Redraw only when the rendered line actually changes, or rewriting the
        # same content 200×/s flickers the terminal.
        nonlocal last_render
        frame = "✓" if done else _SPINNER_FRAMES[int(now * 10) % len(_SPINNER_FRAMES)]
        payload = f"\r{frame} {text}\x1b[K" + ("\n" if done else "")
        if payload == last_render:
            return
        last_render = payload
        console.write(payload)
        console.flush()

    if not live:
        log.info("centering — hold an open palm still for %.0fs to set neutral", hold_s)
    while True:
        reading = source.read()
        now = clock()
        if on_tick is not None and (
            last_tick_time is None or now - last_tick_time >= _ON_TICK_INTERVAL_S
        ):
            on_tick()
            last_tick_time = now
        if not (reading.present and reading.recenter_pose):
            # Tolerate a brief drop-out / pose flicker: only a loss sustained past
            # pose_grace_s resets an active hold; within the window keep counting.
            lost_too_long = last_good_time is not None and now - last_good_time > pose_grace_s
            if hold_start is not None and lost_too_long:
                if not live:
                    log.info(
                        "centering reset — open palm lost for >%.2fs; hold again", pose_grace_s
                    )
                hold_start, anchor, last_countdown = None, None, None
                positions.clear()
            # Only flip the line to "waiting" when no hold is active. During a
            # tolerated blip (hold still alive within the grace window) keep the
            # countdown on screen, or a single false-negative MediaPipe frame
            # flickers the text between "centering in Ns" and "waiting".
            if live and hold_start is None:
                show(now, "waiting for an open palm — hold still")
            sleep(poll_interval_s)
            continue
        last_good_time = now
        moved = anchor is not None and float(np.linalg.norm(reading.position - anchor)) > move_tol
        if hold_start is None or moved:
            hold_start = now
            anchor = reading.position.copy()
            positions = [reading.position.copy()]
            last_countdown = None
            if live:
                show(now, "open palm detected — keep holding still")
            sleep(poll_interval_s)
            continue

        positions.append(reading.position.copy())
        last_orientation = reading.orientation.copy()
        held_for = now - hold_start
        remaining = math.ceil(hold_s - held_for)
        if remaining > 0:
            if live:
                show(now, f"centering in {remaining}s — hold still")
            elif remaining != last_countdown:  # once per whole second
                last_countdown = remaining
                log.info("centering in %ds — hold still", remaining)
        if held_for >= hold_s:
            neutral_position = np.mean(positions, axis=0)
            if live:
                show(now, "centering complete — neutral set", done=True)
            else:
                log.info("centering complete — neutral set")
            return NeutralAnchor(neutral_position, last_orientation)
        sleep(poll_interval_s)


class _OneEuroVector:
    """One-euro filter over an N-vector (Casiez et al., 2012).

    Adaptive low-pass: ``cutoff = min_cutoff + beta·|speed|``. Low ``min_cutoff``
    smooths jitter at rest; ``beta`` buys back responsiveness during fast motion
    so the lag you'd get from a fixed low-pass doesn't show up as drag.
    """

    def __init__(self, *, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._prev_value: np.ndarray | None = None
        self._prev_derivative: np.ndarray | None = None
        self._prev_time: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._prev_value = None
        self._prev_derivative = None
        self._prev_time = None

    def __call__(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        if self._prev_value is None or self._prev_time is None:
            self._prev_value = value.copy()
            self._prev_derivative = np.zeros_like(value)
            self._prev_time = timestamp
            return value.copy()

        dt = timestamp - self._prev_time
        if dt <= 0.0:  # non-monotonic clock ⇒ pass through, don't divide by zero
            return self._prev_value.copy()

        derivative = (value - self._prev_value) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        derivative = a_d * derivative + (1.0 - a_d) * self._prev_derivative  # type: ignore[operator]

        cutoff = self._min_cutoff + self._beta * np.linalg.norm(derivative)
        a = self._alpha(float(cutoff), dt)
        filtered = a * value + (1.0 - a) * self._prev_value

        self._prev_value = filtered
        self._prev_derivative = derivative
        self._prev_time = timestamp
        return filtered.copy()


@dataclass(frozen=True)
class WorkspaceCalibration:
    """Metric camera-rig → robot-workspace mapping for the *relative* hand delta.

    Only scale and axis layout — the relative clutch handles the origin. Maps a
    camera-frame displacement ``(dx, dy, dz)`` in metres (the stereo-triangulated
    wrist delta per :class:`HandReading`) to a world EE displacement in metres.

    The camera-frame axes are (x-right, y-down, z-depth) of the rectified left
    camera, where z grows *away* from the camera.

    Attributes
    ----------
    scale:
        Metres of EE travel per metre of hand displacement, per *world* axis
        (x, y, z). A near-1:1 metric mirror gain — a comfortable hand sweep spans a
        good chunk of the workspace; the clutch tiles the rest. Scaled live by
        ``VisionInput(gain=...)``.
    axis_map:
        For each world axis, which camera axis (0=x-right, 1=y-down, 2=z-depth)
        drives it. Default maps depth→world-x, x→world-y, y→world-z — i.e. moving
        the hand toward/away from the camera pushes the EE forward/back, left/right
        pans it sideways, up/down raises it.
    axis_sign:
        ±1 per world axis to flip direction. The forward axis is negative because
        metric depth grows *away* from the camera (so hand-toward-camera ⇒ forward),
        and image y grows downward.

    ponytail: ``scale`` and the signs are rig-dependent tuning knobs — flip a sign
    if an axis feels inverted; raise ``scale`` to tile the workspace with fewer
    clutches.
    """

    scale: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5, 1.5]))
    axis_map: tuple[int, int, int] = (2, 0, 1)
    axis_sign: np.ndarray = field(default_factory=lambda: np.array([-1.0, 1.0, -1.0]))

    def map_delta(self, camera_delta: np.ndarray) -> np.ndarray:
        """Map a camera-space displacement to a world-frame EE displacement (m)."""
        remapped = camera_delta[list(self.axis_map)]
        return self.scale * self.axis_sign * remapped


class VisionInput:
    """Webcam hand-tracking :class:`InputStrategy` (relative clutched mapping).

    Parameters
    ----------
    hand_source:
        Anything with ``read() -> HandReading`` — typically a
        :class:`~ai_teleop.input.hand_tracker.StereoHandSource`; tests pass a
        fake. Read once per :meth:`get_command` (once per control tick).
    calibration:
        Metric camera-rig→workspace mapping. Defaults to :class:`WorkspaceCalibration`.
    grip_force:
        Newton magnitude the open/close scalar maps onto: a flat open hand
        commands ``-grip_force`` (release), a fist ``+grip_force`` (squeeze),
        additive on the baseline grip (see :class:`Command`).
    dropout_grace_s:
        Seconds the sensor may report ``present=False`` before the clutch actually
        releases. A debounce for the clutch: stereo
        triangulation drops out whenever the hand is briefly low-confidence in
        *either* view, and without this every single-frame miss would re-anchor and
        kill sustained motion. Within the grace window the last command is held and
        the anchor is kept, so motion resumes seamlessly; only a sustained loss
        releases the clutch (so deliberate lift-out-to-reposition still works).
    track_orientation:
        When True, the held orientation follows the filtered hand orientation;
        default False holds the start orientation (round peg ⇒ roll irrelevant).
    min_cutoff, beta:
        One-euro filter parameters for the mapped position. Tuned responsive
        (low lag) so the arm tracks the hand rather than lagging behind it.
    initial_anchor:
        Operator-set neutral from the startup centering (:func:`calibrate_neutral`).
        When given, the first engage anchors here instead of the live first frame, so
        the arm holds home until the hand moves *relative* to this pose — no startup jump.
    """

    def __init__(
        self,
        hand_source: _HandSource,
        *,
        calibration: WorkspaceCalibration | None = None,
        gain: float = 1.0,
        grip_force: float = 5.0,
        dropout_grace_s: float = 0.2,
        track_orientation: bool = False,
        min_cutoff: float = 2.0,
        beta: float = 1.5,
        initial_anchor: NeutralAnchor | None = None,
    ) -> None:
        self._source = hand_source
        base = calibration or WorkspaceCalibration()
        self._calibration = replace(base, scale=base.scale * gain) if gain != 1.0 else base
        self._grip_force = grip_force
        self._dropout_grace_s = dropout_grace_s
        self._track_orientation = track_orientation
        self._position_filter = _OneEuroVector(min_cutoff=min_cutoff, beta=beta)

        self._initial_anchor = initial_anchor  # operator-set neutral; consumed on first engage
        self._engaged = False
        self._last_present_time: float | None = None  # sim_time of the last present reading
        self._held: Command | None = None  # last commanded pose (held on disengage)
        self._hand_anchor: np.ndarray | None = None  # camera-space pose at engage
        self._ee_anchor: np.ndarray | None = None  # world EE position at engage
        self._hand_orientation_anchor: np.ndarray | None = None  # hand quat at engage
        self._ee_orientation_anchor: np.ndarray | None = None  # EE quat at engage

    def get_command(self, observation: Observation) -> Command:
        # Seed the held pose from the current EE pose on the first tick, so a
        # disengaged start simply holds where the arm already is.
        if self._held is None:
            self._held = Command(
                observation.ee_pose[:3].copy(), observation.ee_pose[3:].copy(), 0.0
            )

        reading = self._source.read()

        # Drop-out handling, debounced. Stereo triangulation blinks out whenever the
        # hand is briefly low-confidence in either view; without a grace window every
        # single-frame miss would disengage and re-anchor, killing sustained motion.
        if not reading.present:
            within_grace = (
                self._engaged
                and self._last_present_time is not None
                and observation.sim_time - self._last_present_time <= self._dropout_grace_s
            )
            if within_grace:
                # Brief blip: hold the last command, keep the anchor, stay engaged so
                # motion resumes from the same reference when the hand reappears.
                return self._held
            # Sustained loss: actually release the clutch. Freeze at the arm's real
            # pose (never drifts home); re-acquiring will re-anchor from here.
            was_engaged = self._engaged
            self._engaged = False
            self._held = Command(
                observation.ee_pose[:3].copy(),
                observation.ee_pose[3:].copy(),
                self._held.delta_grip_force,
            )
            if was_engaged:  # transition-only: not every absent tick
                log.info("clutch released — hand left frame; holding EE pose")
            return self._held

        self._last_present_time = observation.sim_time

        # Fresh engage (sustained drop-out → present, or first detection): re-anchor
        # so the current hand position maps to wherever the EE currently is — no jump.
        if not self._engaged:
            self._engaged = True
            if self._initial_anchor is not None:
                # Operator-set neutral from the startup centering: anchor here (not the live
                # first frame, which may catch the hand still entering the frame), so the arm
                # holds home until the hand moves *relative* to this pose.
                self._hand_anchor = self._initial_anchor.hand_position.copy()
                self._hand_orientation_anchor = self._initial_anchor.hand_orientation.copy()
                self._initial_anchor = None  # consume once; clutch re-engages use live frames
            else:
                self._hand_anchor = reading.position.copy()
                self._hand_orientation_anchor = reading.orientation.copy()
            self._ee_anchor = self._held.target_position.copy()
            self._ee_orientation_anchor = self._held.target_quaternion.copy()
            self._position_filter.reset()
            log.info("clutch engaged — anchored to current hand/EE pose")

        assert self._hand_anchor is not None and self._ee_anchor is not None

        # Plain mirror: absolute position, EE = anchor + scale·(hand − hand_anchor).
        camera_delta = reading.position - self._hand_anchor
        raw_target = self._ee_anchor + self._calibration.map_delta(camera_delta)

        target_position = self._position_filter(raw_target, observation.sim_time)
        target_quaternion = self._oriented(reading)
        # Fist → squeeze, open hand → release (open=1 ⇒ release, fist=0 ⇒ squeeze).
        delta_grip_force = (2.0 * reading.open_close - 1.0) * self._grip_force

        self._held = Command(target_position, target_quaternion, delta_grip_force)
        return self._held

    def _oriented(self, reading: HandReading) -> np.ndarray:
        """The held EE quaternion for this tick.

        When not tracking orientation, hold the start orientation. When tracking, mirror the
        hand *relatively*: rotate the anchored EE orientation by the hand's rotation since the
        anchor. At neutral (hand at the anchor pose) this is exactly the anchored EE
        orientation, so engaging never snaps the wrist to the hand's absolute orientation.
        """
        assert self._held is not None
        if not self._track_orientation:
            return self._held.target_quaternion
        assert self._hand_orientation_anchor is not None
        assert self._ee_orientation_anchor is not None
        hand_delta = quat_mul(reading.orientation, quat_conjugate(self._hand_orientation_anchor))
        target = quat_mul(hand_delta, self._ee_orientation_anchor)
        return target / np.linalg.norm(target)
