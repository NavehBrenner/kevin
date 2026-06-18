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
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation

from .hand_tracker import HandReading


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

    Attributes
    ----------
    scale:
        Metres of EE travel per unit of camera displacement, per *world* axis
        (x, y, z). A full hand sweep is ~0.5 in normalized image space, so
        ~0.6 puts the ~0.3 m workspace within a comfortable sweep.
    axis_map:
        For each world axis, which camera axis (0=x, 1=y, 2=z) drives it. Default
        maps camera-x→world-y, camera-y→world-z, camera-z→world-x — i.e. moving
        the hand left/right pans the EE sideways, up/down raises it, and toward/
        away from the camera pushes toward/away from the wall.
    axis_sign:
        ±1 per world axis to flip direction (image y grows downward; depth z
        grows away). Chosen so natural hand motion reads as intuitive EE motion.
    """

    scale: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.6, 0.6]))
    axis_map: tuple[int, int, int] = (2, 0, 1)
    axis_sign: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0, -1.0]))

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
    track_orientation:
        When True, the held orientation follows the filtered hand orientation;
        default False holds the start orientation (round peg ⇒ roll irrelevant).
    min_cutoff, beta:
        One-euro filter parameters for the mapped position.
    """

    def __init__(
        self,
        hand_source: _HandSource,
        *,
        calibration: WorkspaceCalibration | None = None,
        grip_force: float = 5.0,
        track_orientation: bool = False,
        min_cutoff: float = 1.0,
        beta: float = 0.7,
    ) -> None:
        self._source = hand_source
        self._calibration = calibration or WorkspaceCalibration()
        self._grip_force = grip_force
        self._track_orientation = track_orientation
        self._position_filter = _OneEuroVector(min_cutoff=min_cutoff, beta=beta)

        self._engaged = False
        self._held: Command | None = None  # last commanded pose (held on disengage)
        self._hand_anchor: np.ndarray | None = None  # camera-space pose at engage
        self._ee_anchor: np.ndarray | None = None  # world EE position at engage

    def get_command(self, observation: Observation) -> Command:
        # Seed the held pose from the current EE pose on the first tick, so a
        # disengaged start simply holds where the arm already is.
        if self._held is None:
            self._held = Command(
                observation.ee_pose[:3].copy(), observation.ee_pose[3:].copy(), 0.0
            )

        reading = self._source.read()

        # Drop-out (or disengaged): hold the last command verbatim.
        if not reading.present:
            self._engaged = False
            return self._held

        # Fresh engage (drop-out → present, or first detection): re-anchor so the
        # current hand position maps to wherever the EE currently is — no jump.
        if not self._engaged:
            self._engaged = True
            self._hand_anchor = reading.position.copy()
            self._ee_anchor = self._held.target_position.copy()
            self._position_filter.reset()

        assert self._hand_anchor is not None and self._ee_anchor is not None
        world_delta = self._calibration.map_delta(reading.position - self._hand_anchor)
        target_position = self._position_filter(self._ee_anchor + world_delta, observation.sim_time)

        target_quaternion = (
            reading.orientation.copy() if self._track_orientation else self._held.target_quaternion
        )
        # open=1 → release (−force); fist=0 → squeeze (+force).
        delta_grip_force = (1.0 - 2.0 * reading.open_close) * self._grip_force

        self._held = Command(target_position, target_quaternion, delta_grip_force)
        return self._held
