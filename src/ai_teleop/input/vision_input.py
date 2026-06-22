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
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation

from .hand_tracker import HandReading

ControlMode = Literal["mirror", "expo", "rate"]
"""How a hand displacement drives the EE:

- ``mirror`` — absolute position: ``EE = anchor + scale·(hand − hand_anchor)``.
  Direct, but one linear gain can't be both "cross the workspace" and "nudge
  2 mm", and hand tremor maps straight in.
- ``expo`` — same position control, but the camera delta passes through a
  dead-zone (kills resting drift) + a cubic expo curve (tiny near rest ⇒
  precision, near-linear at full sweep ⇒ reach). The intuitive default.
- ``rate`` — "point to steer": an open hand sets the in-plane EE *velocity* from
  the direction it points, plus a gentle forward creep scaled by how much it
  angles into the camera; a fist drives slowly backward; a half-closed hand
  locks. Position-independent and low-fatigue; unlimited range (no camera-FoV
  ceiling). See :class:`~ai_teleop.input.hand_tracker.HandReading`.
"""

# rate-mode tuning — hand-tuned calibration knobs; adjust from operator feedback.
_RATE_FWD_DEADZONE = 0.01  # forwardness ignored below this (keeps flat pointing from creeping)
_RATE_OPEN_THRESHOLD = 0.6  # open_close above this ⇒ open hand ⇒ steer (+ forward)
_RATE_FIST_THRESHOLD = 0.1  # open_close below this ⇒ fist ⇒ drive back; between the two ⇒ lock


def _deadzone(value: np.ndarray, width: float) -> np.ndarray:
    """Per-axis dead-zone: zero inside ±width, shifted-linear outside (no step)."""
    return np.sign(value) * np.maximum(np.abs(value) - width, 0.0)


def _expo(value: np.ndarray, amount: float) -> np.ndarray:
    """Cubic expo blend in [0,1]: 0 = linear, 1 = pure cubic (soft centre)."""
    return (1.0 - amount) * value + amount * value**3


class _HandSource(Protocol):
    """Anything that yields the latest hand reading — the live tracker or a fake."""

    def read(self) -> HandReading: ...


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
    gain:
        Scalar multiplier on the calibration's per-axis scale — the live "how
        much the arm mirrors the hand" knob. >1 amplifies hand motion, <1 damps.
        In ``rate`` mode it scales the velocities (``rate_speed``/``pitch_gain``).
    mode:
        Control mapping — see :data:`ControlMode`. Default ``expo``.
    deadzone:
        In ``expo`` mode, the camera-space half-width zeroed around the anchor so
        a still hand commands nothing. In ``rate`` mode, the pitch dead-zone.
    expo:
        Cubic expo amount in [0,1] for ``expo`` mode (0 = linear, 1 = soft).
    rate_speed:
        In-plane EE speed (m/s) while the two-finger drive gesture is held
        (``rate`` mode). Scaled by ``gain``.
    forward_gain:
        Forward EE speed (m/s) per unit of forwardness (``rate`` mode) — the
        gentle creep as you angle the fingers into the camera. Kept small so it
        stays minor next to the in-plane motion. Scaled by ``gain``.
    back_speed:
        Backward EE speed (m/s) while a fist is held (``rate`` mode). Scaled by
        ``gain``.
    leash:
        Max distance (m, per axis) the velocity target may lead the actual EE in
        ``rate`` mode. Bounds run-away when the arm can't keep up and sets the
        effective top speed (bigger ⇒ larger impedance error ⇒ faster). On lock
        the target snaps back to the arm, so there's no catch-up drift.
    lock_delay:
        Seconds the drive gesture may drop out before the arm locks (``rate``
        mode) — a debounce so brief tracking glitches don't freeze you.
    dropout_grace_s:
        Seconds the sensor may report ``present=False`` before the clutch actually
        releases (``mirror``/``expo`` modes). A debounce for the clutch: stereo
        triangulation drops out whenever the hand is briefly low-confidence in
        *either* view, and without this every single-frame miss would re-anchor and
        kill sustained motion. Within the grace window the last command is held and
        the anchor is kept, so motion resumes seamlessly; only a sustained loss
        releases the clutch (so deliberate lift-out-to-reposition still works).
    track_orientation:
        When True, the held orientation follows the filtered hand orientation;
        default False holds the start orientation (round peg ⇒ roll irrelevant).
    recenter:
        When True, an open palm held square to the camera and still for
        ``recenter_hold_s`` seconds re-anchors ("set neutral here") — a gesture
        alternative to the lift-out-of-frame clutch, mirroring stereohand's
        recenter. Off by default; meaningful only in ``mirror``/``expo`` modes (the
        velocity-driven ``rate`` mode has no anchor). The grip command is held
        steady during the hold so recentering doesn't release the gripper.
    recenter_hold_s:
        Seconds the open-palm pose must be held still to trigger a recenter.
    recenter_lock_s:
        Seconds into a recenter hold after which the arm is *locked* (frozen in
        place) for the rest of the hold — well before ``recenter_hold_s`` fires — so
        the arm doesn't drift toward the posed hand while you wait out the countdown.
    recenter_move_tol:
        How far (m) the hand may drift during the hold and still count as "still";
        exceeding it restarts the timer.
    recenter_pose_grace_s:
        Seconds the noisy ``recenter_pose`` flag may flicker off mid-hold without
        restarting the countdown (a debounce mirroring stereohand's renderer); only a
        sustained loss of the pose releases the hold.
    min_cutoff, beta:
        One-euro filter parameters for the mapped position. Tuned responsive
        (low lag) so the arm tracks the hand rather than lagging behind it.
    """

    def __init__(
        self,
        hand_source: _HandSource,
        *,
        calibration: WorkspaceCalibration | None = None,
        gain: float = 1.0,
        grip_force: float = 5.0,
        mode: ControlMode = "expo",
        deadzone: float = 0.03,
        expo: float = 0.6,
        rate_speed: float = 4.0,
        forward_gain: float = 6.0,
        back_speed: float = 0.3,
        leash: float = 0.2,
        lock_delay: float = 0.2,
        dropout_grace_s: float = 0.2,
        track_orientation: bool = False,
        recenter: bool = False,
        recenter_hold_s: float = 3.0,
        recenter_lock_s: float = 0.5,
        recenter_move_tol: float = 0.02,
        recenter_pose_grace_s: float = 0.15,
        min_cutoff: float = 2.0,
        beta: float = 1.5,
    ) -> None:
        self._source = hand_source
        base = calibration or WorkspaceCalibration()
        # rate mode is direction-driven, not offset-driven: `gain` scales the
        # velocities (below), not the position scale, so leave the calibration be.
        self._calibration = (
            replace(base, scale=base.scale * gain) if gain != 1.0 and mode != "rate" else base
        )
        self._grip_force = grip_force
        self._mode = mode
        self._deadzone = deadzone
        self._expo = expo
        rate_scale = gain if mode == "rate" else 1.0
        self._rate_speed = rate_speed * rate_scale
        self._forward_gain = forward_gain * rate_scale
        self._back_speed = back_speed * rate_scale
        self._leash = leash
        self._lock_delay = lock_delay
        self._dropout_grace_s = dropout_grace_s
        self._track_orientation = track_orientation
        self._recenter = recenter
        self._recenter_hold_s = recenter_hold_s
        self._recenter_lock_s = recenter_lock_s
        self._recenter_move_tol = recenter_move_tol
        self._recenter_pose_grace_s = recenter_pose_grace_s
        self._position_filter = _OneEuroVector(min_cutoff=min_cutoff, beta=beta)

        self._recenter_hold_start: float | None = None  # sim_time the open-palm hold began
        self._recenter_anchor: np.ndarray | None = None  # hand position at hold start
        self._last_recenter_pose_time: float | None = None  # last tick the pose was detected
        self._recentered = False  # latched after a hold fires, until the pose releases
        self._in_recenter_hold = False  # pose held this tick ⇒ freeze grip
        self._recenter_locked = False  # pose held > recenter_lock_s ⇒ freeze the arm

        self._engaged = False
        self._last_present_time: float | None = None  # sim_time of the last present reading
        self._held: Command | None = None  # last commanded pose (held on disengage)
        self._hand_anchor: np.ndarray | None = None  # camera-space pose at engage
        self._ee_anchor: np.ndarray | None = None  # world EE position at engage
        self._prev_time: float | None = None  # for rate-mode integration
        self._driving = False  # rate-mode drive/lock state (with lock_delay debounce)
        self._last_drive_time = 0.0  # last tick a drive gesture (point or fist) was seen

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
            self._engaged = False
            self._held = Command(
                observation.ee_pose[:3].copy(),
                observation.ee_pose[3:].copy(),
                self._held.delta_grip_force,
            )
            return self._held

        self._last_present_time = observation.sim_time

        # Fresh engage (sustained drop-out → present, or first detection): re-anchor
        # so the current hand position maps to wherever the EE currently is — no jump.
        if not self._engaged:
            self._engaged = True
            self._hand_anchor = reading.position.copy()
            self._ee_anchor = self._held.target_position.copy()
            self._prev_time = observation.sim_time
            self._position_filter.reset()

        assert self._hand_anchor is not None and self._ee_anchor is not None
        assert self._prev_time is not None

        if self._mode == "rate":
            return self._rate_command(reading, observation)

        if self._recenter:
            self._maybe_recenter(reading, observation)
            if self._recenter_locked:
                # Holding the calibration pose: freeze the arm so it doesn't drift
                # toward the posed hand while you wait out the recenter countdown.
                return self._held

        camera_delta = reading.position - self._hand_anchor
        if self._mode == "expo":
            shaped = _expo(_deadzone(camera_delta, self._deadzone), self._expo)
            raw_target = self._ee_anchor + self._calibration.map_delta(shaped)
        else:  # mirror
            raw_target = self._ee_anchor + self._calibration.map_delta(camera_delta)

        target_position = self._position_filter(raw_target, observation.sim_time)
        target_quaternion = (
            reading.orientation.copy() if self._track_orientation else self._held.target_quaternion
        )
        # Fist → squeeze, open hand → release (open=1 ⇒ release, fist=0 ⇒ squeeze).
        delta_grip_force = (2.0 * reading.open_close - 1.0) * self._grip_force
        if self._in_recenter_hold:
            # The recenter pose is an open palm; don't read it as "release the grip".
            delta_grip_force = self._held.delta_grip_force

        self._held = Command(target_position, target_quaternion, delta_grip_force)
        return self._held

    def _maybe_recenter(self, reading: HandReading, observation: Observation) -> None:
        """Open-palm-held-still gesture → re-anchor, without lifting out of frame.

        Mirrors stereohand's recenter: holding the open-palm pose
        (``reading.recenter_pose``) still for ``recenter_hold_s`` seconds sets the
        current hand position as the new neutral mapped to the current EE — the arm
        stays put and subsequent motion is measured from here. Fires once per hold
        (re-arms when the pose releases). Sets ``_in_recenter_hold`` so the caller
        can freeze the grip during the hold.
        """
        assert self._held is not None
        if not reading.recenter_pose:
            # Debounce like stereohand's renderer: ``recenter_pose`` comes from a noisy
            # per-frame landmark test, so a single flicker shouldn't restart a hold in
            # progress. Within the grace window, hold all recenter state as-is; only a
            # sustained loss of the pose actually releases.
            within_grace = (
                self._recenter_hold_start is not None
                and self._last_recenter_pose_time is not None
                and observation.sim_time - self._last_recenter_pose_time
                <= self._recenter_pose_grace_s
            )
            if within_grace:
                return
            self._recenter_hold_start = None
            self._recenter_anchor = None
            self._recentered = False
            self._in_recenter_hold = False
            self._recenter_locked = False
            return

        self._last_recenter_pose_time = observation.sim_time
        self._in_recenter_hold = True
        if self._recentered:
            return  # already fired for this hold; wait for the pose to release

        moved = (
            self._recenter_anchor is not None
            and float(np.linalg.norm(reading.position - self._recenter_anchor))
            > self._recenter_move_tol
        )
        if self._recenter_hold_start is None or moved:
            # (Re)starting the hold — the operator is still positioning, so let the arm
            # follow until the pose settles.
            self._recenter_anchor = reading.position.copy()
            self._recenter_hold_start = observation.sim_time
            self._recenter_locked = False
            return

        held_for = observation.sim_time - self._recenter_hold_start
        # Lock the arm once the pose has been held still a moment, well before the full
        # recenter fires — so the countdown wait doesn't drift the arm toward the pose.
        self._recenter_locked = held_for >= self._recenter_lock_s
        if held_for >= self._recenter_hold_s:
            # Re-anchor like the clutch does: map the current hand to where the arm
            # actually *is* (not the last commanded target), so any accumulated
            # target-vs-actual drift is cleared and the arm holds put.
            self._hand_anchor = reading.position.copy()
            self._ee_anchor = observation.ee_pose[:3].copy()
            self._position_filter.reset()
            self._recentered = True

    def _rate_command(self, reading: HandReading, observation: Observation) -> Command:
        """`rate` "point to steer": an open hand → in-plane velocity (where it
        points) + gentle forward creep (angle into camera); a fist → slow
        backward; a half-closed hand → lock. Open vs fist comes from open_close,
        which is foreshortening-robust, so pointing into the camera stays "open"."""
        assert self._held is not None and self._prev_time is not None
        dt = float(np.clip(observation.sim_time - self._prev_time, 0.0, 0.1))
        self._prev_time = observation.sim_time
        ee_position = observation.ee_pose[:3]

        # Open hand steers (+ creeps forward); a fist drives backward; in between
        # locks. Either drive gesture is "driving".
        is_open = reading.open_close > _RATE_OPEN_THRESHOLD
        is_fist = reading.open_close < _RATE_FIST_THRESHOLD

        # Drive/lock with a debounce: engage instantly on a drive gesture, but only
        # lock once it's been gone for `lock_delay` (brief glitches don't freeze).
        if is_open or is_fist:
            self._driving = True
            self._last_drive_time = observation.sim_time
        elif observation.sim_time - self._last_drive_time > self._lock_delay:
            self._driving = False

        # grip is parked for now (see project notes) — rate mode holds it steady.
        if not self._driving:
            # Locked: freeze at the arm's actual pose (no catch-up drift).
            self._held = Command(
                ee_position.copy(), observation.ee_pose[3:].copy(), self._held.delta_grip_force
            )
            return self._held

        sign = self._calibration.axis_sign
        world_velocity = np.zeros(3)
        if is_open:
            # point_direction magnitude shrinks as the hand angles into the camera,
            # so in-plane motion fades out exactly as the forward component grows.
            point = reading.point_direction
            world_velocity[1] = sign[1] * self._rate_speed * float(point[0])  # L/R ← image-x
            world_velocity[2] = sign[2] * self._rate_speed * float(point[1])  # U/D ← image-y
            forward = max(reading.forwardness - _RATE_FWD_DEADZONE, 0.0)  # one-sided, gentle
            # Forward/back is a world-frame robot direction (+world-x = forward), driven
            # by the gesture's own "into-camera" signal — independent of axis_sign[0],
            # which only flips the *mirror* depth-delta mapping.
            world_velocity[0] = self._forward_gain * forward  # forward ← angle into camera
        elif is_fist:
            world_velocity[0] = -self._back_speed  # backward ← fist
        # else (grace window): velocity 0, just pause

        raw_target = self._held.target_position + world_velocity * dt
        # Leash the target to the arm so it can't run far ahead (anti-runaway; also
        # sets the effective top speed via the steady-state impedance error).
        raw_target = ee_position + np.clip(raw_target - ee_position, -self._leash, self._leash)
        target_position = self._position_filter(raw_target, observation.sim_time)

        target_quaternion = (
            reading.orientation.copy() if self._track_orientation else self._held.target_quaternion
        )
        self._held = Command(target_position, target_quaternion, self._held.delta_grip_force)
        return self._held
