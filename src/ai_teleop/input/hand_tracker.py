"""Stereo hand sensor — metric 3-D landmarks → hand-pose readings (LAB-50/74).

The off-the-shelf **sensor** layer per `project-scope.md`: two calibrated webcams
feed the :mod:`stereohand` package, which triangulates each frame into 21 *metric*
3-D landmarks, and we distill those into a small typed :class:`HandReading` (wrist
position in metres, an orientation estimate, an open/close grip proxy, and a
``present`` flag). It is deliberately *pure sensing*: no robot, no
:class:`Command`, no calibration, no clutch — all of that lives one layer up in
:class:`ai_teleop.input.vision_input.VisionInput`.

Two halves:

- :func:`reading_from_landmarks` — the deterministic landmark→reading math. No
  camera, no stereohand import; this is what the unit tests exercise with
  synthetic landmark sets.
- :class:`StereoHandSource` — the live path. It adapts
  :class:`stereohand.StereoHandTracker` (capture + MediaPipe-per-view +
  triangulation on a background thread) to the non-blocking ``read() ->
  HandReading`` seam, returning ``present=False`` when the hand is missing in
  either view (never raises mid-stream).

``stereohand`` is imported lazily inside the live class so the pure function (and
its tests) work without the ``stereo-input`` extra installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from ai_teleop.common.geometry import mat3_to_quat
from ai_teleop.common.log import get_logger

log = get_logger("hand_tracker")

# MediaPipe Hands landmark indices we use (of the 21-point model).
_WRIST = 0
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_MIDDLE_FINGERTIP = 12
_PINKY_MCP = 17
_FINGERTIPS = (8, 12, 16, 20)  # index, middle, ring, pinky tips (thumb excluded)
_FINGER_MCPS = (5, 9, 13, 17)  # their knuckles (for the open-palm recenter test)

# Empirical open/close bounds for the fingertip-spread ratio (tip→wrist distance
# over hand scale). Fist ≈ 1.0, flat open hand ≈ 2.4 — hand-tuned; recalibrate
# from operator feel if the grip proxy reads hot or cold.
_GRIP_RATIO_CLOSED = 1.0
_GRIP_RATIO_OPEN = 2.4

# The cv2 preview only needs ~30 Hz, and its imshow/waitKey is an expensive WSLg round-trip
# on the main thread that steals CPU/GIL from the tracking thread. Throttle by *wall-clock
# time*, not by a read-count stride: read() is called at wildly different rates (≈200 Hz in the
# centering loop, ≈100-160 Hz in the render-paced control loop), so a fixed stride yields ~10 fps
# in some phases and starves tracking in others. A time gate gives a stable preview rate
# regardless of caller cadence. The hand reading itself is taken every call (cheap — it just
# returns the background thread's latest).
_WINDOW_PUMP_INTERVAL_S = 1.0 / 30.0


@dataclass(frozen=True)
class HandReading:
    """One frame of hand sensing, in the metric stereo-rig (left-camera) frame.

    Lives here (not in ``common/``) because only the input layer consumes it.

    Attributes
    ----------
    position:
        Shape (3,) — the wrist landmark's true metric xyz (metres) in the
        rectified left-camera frame: x right, y down, z = depth (away from the
        camera). Real triangulated depth, no proxy. The strategy layer maps this
        to robot workspace.
    orientation:
        Shape (4,) unit quaternion (w, x, y, z) — a hand-frame estimate fit to the
        3-D palm landmarks. Observable from two views; the strategy may still filter it.
    open_close:
        Grip proxy in [0, 1]: 0 = closed fist, 1 = flat open hand.
    present:
        False when no hand was detected this frame (drop-out).
    point_direction:
        Shape (2,) — the in-plane direction the hand points (wrist → middle
        fingertip), in the camera xy-plane (x right, y down), scaled by hand size
        so its *magnitude* shrinks as the hand angles into the camera
        (foreshortening). Position-independent: it's *where the hand points*, not
        where it sits. Drives in-plane steering in ``rate`` mode; the shrinking
        magnitude blends in-plane motion out as the forward component takes over.
    forwardness:
        How much the hand points *into* the camera (≈ "forward"): ~0 when pointing
        across the image plane, larger as the fingertips angle toward the lens.
        From the fingertips' depth (negative z = toward camera). The strategy
        dead-zones it and drives a gentle forward creep.
    recenter_pose:
        True when this frame is an open palm held square to the camera — the pose
        the startup centering (:func:`ai_teleop.input.calibrate_neutral`) requires the
        operator to hold still to set the neutral anchor. The hold timing lives in the
        calibration routine; this is just the per-frame pose test.

    The ``rate`` mode reads the open/close grip to pick a gesture — an open hand
    steers (+ creeps forward), a fist drives back — so it is robust where a
    finger-extension test fails: ``open_close`` is built from 3-D landmark
    distances and stays "open" even when the hand foreshortens toward the camera.
    """

    position: np.ndarray
    orientation: np.ndarray
    open_close: float
    present: bool
    point_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    forwardness: float = 0.0
    recenter_pose: bool = False


_ABSENT = HandReading(
    position=np.zeros(3),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    open_close=0.0,
    present=False,
)


def reading_from_landmarks(landmarks: np.ndarray) -> HandReading:
    """Convert a (21, 3) metric hand landmark array to a :class:`HandReading`.

    Pure and camera-free — the unit-tested core. ``landmarks`` is the hand's 21
    points as real metric (x, y, z) rows in the rectified left-camera frame (the
    triangulated output of :mod:`stereohand`). ``position`` is the true wrist xyz;
    the derived signals (grip, pointing, orientation) are scale-invariant ratios.
    """
    if landmarks.shape != (21, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {landmarks.shape}")

    wrist = landmarks[_WRIST]

    # Apparent hand size (wrist→middle-finger MCP) — normalizes the grip ratio and
    # the pointing vector so both are distance-invariant.
    hand_scale = float(np.linalg.norm(landmarks[_MIDDLE_MCP] - wrist))
    position = wrist[:3].copy()
    if hand_scale < 1e-6:
        return HandReading(position, np.array([1.0, 0.0, 0.0, 0.0]), 0.0, True)

    tip_spread = float(np.mean([np.linalg.norm(landmarks[t] - wrist) for t in _FINGERTIPS]))
    ratio = tip_spread / hand_scale
    open_close = (ratio - _GRIP_RATIO_CLOSED) / (_GRIP_RATIO_OPEN - _GRIP_RATIO_CLOSED)
    open_close = float(np.clip(open_close, 0.0, 1.0))

    orientation = _palm_orientation(landmarks)

    # In-plane pointing: wrist → middle fingertip, scaled by hand size so the
    # vector is distance-invariant and *shrinks* as the hand angles into the
    # camera (the fingertip foreshortens toward the wrist). Kept un-normalized on
    # purpose: the magnitude blends in-plane steering out as forwardness rises.
    point_direction = (landmarks[_MIDDLE_FINGERTIP, :2] - wrist[:2]) / hand_scale
    # Forwardness: fingertips angled toward the camera ⇒ negative tip z ⇒ positive.
    # Wrist-relative, to remove the camera-frame depth offset.
    forwardness = -float(np.mean(landmarks[list(_FINGERTIPS), 2] - wrist[2]))

    return HandReading(
        position,
        orientation,
        open_close,
        present=True,
        point_direction=point_direction,
        forwardness=forwardness,
        recenter_pose=_palm_open_facing(landmarks),
    )


def _palm_open_facing(landmarks: np.ndarray) -> bool:
    """True when the hand is open and roughly square to the camera (the recenter pose).

    Ported from stereohand's renderer: ≥3 fingers extended (tip→wrist clearly longer
    than knuckle→wrist) and the palm-plane normal roughly aligned with the camera's
    z-axis (the squareness test reads metric landmark depth).
    """
    wrist = landmarks[_WRIST]
    extended = sum(
        np.linalg.norm(landmarks[tip] - wrist) > 1.4 * np.linalg.norm(landmarks[mcp] - wrist)
        for tip, mcp in zip(_FINGERTIPS, _FINGER_MCPS, strict=True)
    )
    if extended < 3:
        return False
    normal = np.cross(landmarks[_INDEX_MCP] - wrist, landmarks[_PINKY_MCP] - wrist)
    norm = float(np.linalg.norm(normal))
    return norm > 0 and abs(float(normal[2])) > 0.7 * norm


def _palm_orientation(landmarks: np.ndarray) -> np.ndarray:
    """Estimate a hand-frame quaternion from the palm landmarks.

    Builds an orthonormal frame: +y points wrist→middle-MCP (finger direction),
    +x points across the knuckles (index→pinky MCP), +z is the palm normal. Rough
    — adequate for the coarse teleop signal, and the strategy filters/ignores it.
    """
    wrist = landmarks[_WRIST]
    forward = landmarks[_MIDDLE_MCP] - wrist
    across = landmarks[_PINKY_MCP] - landmarks[_INDEX_MCP]

    y_axis = forward / (np.linalg.norm(forward) + 1e-9)
    z_axis = np.cross(across, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    z_axis = z_axis / z_norm
    x_axis = np.cross(y_axis, z_axis)

    quat = mat3_to_quat(np.column_stack([x_axis, y_axis, z_axis]))
    mujoco.mju_normalize4(quat)
    return quat


class StereoHandSource:
    """Two-webcam stereo tracker → metric :class:`HandReading` (the live sensor).

    Adapts :class:`stereohand.StereoHandTracker` (which triangulates metric
    ``(21, 3)`` landmarks from two calibrated webcams) to the non-blocking
    ``read() -> HandReading`` seam by running :func:`reading_from_landmarks` on the
    triangulated landmarks — so :class:`~ai_teleop.input.vision_input.VisionInput`
    gets real metric depth and an observable orientation (enable
    ``track_orientation`` for true 6-DoF mirroring).

    ``stereohand`` is imported lazily (the ``stereo-input`` extra) so the pure
    landmark math and its tests don't need it installed.
    """

    def __init__(
        self,
        calibration_path: str,
        *,
        left: int | str = 0,
        right: int | str = 2,
        show_window: bool = False,
        max_fps: int | Literal["cam"] = "cam",
    ) -> None:
        from stereohand import RenderConfig, StereoCalibration, StereoHandTracker

        calibration = StereoCalibration.load(calibration_path)
        self._show_window = show_window
        self._last_pump = 0.0  # wall-clock of the last cv2 window pump (see read())
        # recenter=True only drives the renderer's open-palm countdown HUD, a handy visual
        # while the operator holds the startup-centering pose; kevin times the hold itself in
        # calibrate_neutral, and the renderer's origin offset is a no-op for us.
        self._tracker = StereoHandTracker.open(
            calibration,
            left=left,
            right=right,
            max_fps=max_fps,
            render=show_window,
            render_config=RenderConfig() if show_window else None,
        )
        log.info(
            "stereo hand tracker started (calib %s, cameras %r/%r)",
            calibration_path,
            left,
            right,
        )

    def read(self) -> HandReading:
        # cv2 GUI must be pumped from the main (control-loop) thread, but only needs ~30 Hz —
        # pump on a wall-clock interval, not every call (see the interval constant). The hand
        # reading below is taken every call regardless (cheap — the background thread's latest).
        if self._show_window:
            now = time.monotonic()
            if now - self._last_pump >= _WINDOW_PUMP_INTERVAL_S:
                self._last_pump = now
                # stereohand's split renderer draws in render_step() but flushes the imshow
                # buffer to screen only in poll(); without the poll() the window stays blank.
                self._tracker.render_step()
                self._tracker.poll()
        reading = self._tracker.read()
        if not reading.present:
            return _ABSENT
        return reading_from_landmarks(reading.landmarks)

    def set_renderer_origin(self, origin: np.ndarray) -> None:
        """Center the preview's 3-D skeleton view on ``origin`` (metric left-camera frame).

        No-op when the preview window is off (there's no renderer to update). Used to pin the
        renderer's origin to the operator-set neutral from startup centering.
        """
        if self._show_window:
            self._tracker.set_renderer_origin((
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
            ))

    def close(self) -> None:
        self._tracker.close()
        log.info("stereo hand tracker stopped")

    def __enter__(self) -> StereoHandSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
