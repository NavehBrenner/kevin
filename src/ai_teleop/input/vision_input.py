"""VisionInput — hand-pose readings → base EE Command (LAB-51).

The headline M8 teleop logic: turn the MediaPipe sensor's :class:`HandReading`
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
- **Orientation.** Off by default (``track_orientation=False``): the peg is
  round so roll is irrelevant, and the MediaPipe orientation estimate is the
  jitteriest signal — tracking it tends to fight the controller. When enabled,
  the held orientation follows the (filtered) hand orientation.

Live webcam use is manual; the deterministic math here (mapping transform,
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
- ``rate`` — "point to steer": the two-finger gesture sets the in-plane EE
  *velocity* from the direction it points, plus a gentle forward creep scaled by
  how much it angles into the camera; a fist drives slowly backward; an open /
  relaxed hand locks. Position-independent and low-fatigue; unlimited range (no
  camera-FoV ceiling). See :class:`~ai_teleop.input.hand_tracker.HandReading`.
"""

# rate-mode tuning. ponytail: hand-tuned calibration knobs the live feel needs —
# adjust from operator feedback, not magic numbers.
_RATE_FWD_DEADZONE = 0.04  # forwardness ignored below this (keeps in-plane pointing from creeping)
_RATE_FIST_THRESHOLD = 0.25  # open_close below this (with no point gesture) ⇒ fist ⇒ drive back


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
    """Camera-space → robot-workspace mapping for the *relative* hand delta.

    Only scale and axis layout — the relative clutch handles the origin. Maps a
    camera-space displacement ``(dx, dy, dz)`` (image-normalized) to a world EE
    displacement in metres.

    The camera-space axes are (image_x, image_y, depth) per :class:`HandReading`,
    where depth is the apparent-hand-size proxy (larger ⇒ closer to camera).

    Attributes
    ----------
    scale:
        Metres of EE travel per unit of camera displacement, per *world* axis
        (x, y, z). Sized so the hand *mirrors* the arm — a partial hand sweep
        spans the workspace rather than slowly nudging it. The forward/back axis
        is driven by the hand-size proxy, whose usable swing is smaller, so
        world-x gets a bigger gain. Scaled live by ``VisionInput(gain=...)``.
    axis_map:
        For each world axis, which camera axis (0=image_x, 1=image_y, 2=depth)
        drives it. Default maps depth→world-x, image_x→world-y, image_y→world-z —
        i.e. moving the hand toward/away from the camera pushes the EE forward/
        back, left/right pans it sideways, up/down raises it.
    axis_sign:
        ±1 per world axis to flip direction (image y grows downward, etc.).
        Chosen so natural hand motion reads as intuitive EE motion; flip an entry
        if an axis feels inverted.
    """

    scale: np.ndarray = field(default_factory=lambda: np.array([3.0, 1.2, 1.2]))
    axis_map: tuple[int, int, int] = (2, 0, 1)
    axis_sign: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, -1.0]))

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
        :class:`~ai_teleop.input.hand_tracker.MediaPipeHandTracker`; tests pass a
        fake. Read once per :meth:`get_command` (once per control tick).
    calibration:
        Camera→workspace mapping. Defaults to :class:`WorkspaceCalibration`.
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
    track_orientation:
        When True, the held orientation follows the filtered hand orientation;
        default False holds the start orientation (round peg ⇒ roll irrelevant).
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
        forward_gain: float = 2.0,
        back_speed: float = 0.3,
        leash: float = 0.2,
        lock_delay: float = 0.2,
        track_orientation: bool = False,
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
        self._track_orientation = track_orientation
        self._position_filter = _OneEuroVector(min_cutoff=min_cutoff, beta=beta)

        self._engaged = False
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

        # Drop-out (or disengaged): freeze the arm exactly where it physically is
        # right now — command the current EE pose (keeping the last grip). Holds
        # static, never drifts toward home, and re-acquiring re-anchors from here.
        if not reading.present:
            self._engaged = False
            self._held = Command(
                observation.ee_pose[:3].copy(),
                observation.ee_pose[3:].copy(),
                self._held.delta_grip_force,
            )
            return self._held

        # Fresh engage (drop-out → present, or first detection): re-anchor so the
        # current hand position maps to wherever the EE currently is — no jump.
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

        self._held = Command(target_position, target_quaternion, delta_grip_force)
        return self._held

    def _rate_command(self, reading: HandReading, observation: Observation) -> Command:
        """`rate` "point to steer": two-finger gesture → in-plane velocity + gentle
        forward creep (angle into camera); fist → slow backward; else lock."""
        assert self._held is not None and self._prev_time is not None
        dt = float(np.clip(observation.sim_time - self._prev_time, 0.0, 0.1))
        self._prev_time = observation.sim_time
        ee_position = observation.ee_pose[:3]

        # A fist (no point gesture, fingers curled) drives backward; the two-finger
        # gesture steers. Either one is "driving"; anything else locks.
        is_fist = not reading.is_pointing and reading.open_close < _RATE_FIST_THRESHOLD

        # Drive/lock with a debounce: engage instantly on a drive gesture, but only
        # lock once it's been gone for `lock_delay` (brief glitches don't freeze).
        if reading.is_pointing or is_fist:
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
        if reading.is_pointing:
            point = reading.point_direction
            world_velocity[1] = sign[1] * self._rate_speed * float(point[0])  # L/R ← image-x
            world_velocity[2] = sign[2] * self._rate_speed * float(point[1])  # U/D ← image-y
            forward = max(reading.forwardness - _RATE_FWD_DEADZONE, 0.0)  # one-sided, gentle
            world_velocity[0] = (
                sign[0] * self._forward_gain * forward
            )  # forward ← angle into camera
        elif is_fist:
            world_velocity[0] = -sign[0] * self._back_speed  # backward ← fist
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
