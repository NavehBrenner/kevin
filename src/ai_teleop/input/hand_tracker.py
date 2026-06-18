"""MediaPipe Hands sensor wrapper — webcam frames → hand-pose readings (LAB-50).

The off-the-shelf **sensor** layer per `project-scope.md`: OpenCV captures
frames, MediaPipe Hands turns each into 21 landmarks, and we distill those into
a small typed :class:`HandReading` (palm position, a rough orientation estimate,
an open/close grip proxy, and a ``present`` flag). It is deliberately *pure
sensing*: no robot, no :class:`Command`, no calibration, no clutch — all of that
lives one layer up in :class:`ai_teleop.input.vision_input.VisionInput`.

Two halves:

- :func:`reading_from_landmarks` — the deterministic landmark→reading math. No
  camera, no MediaPipe import; this is what the unit tests exercise with
  synthetic landmark sets.
- :class:`MediaPipeHandTracker` — the live path. OpenCV + MediaPipe run on a
  background thread (camera ~30 fps decoupled from the 500 Hz control loop);
  :meth:`MediaPipeHandTracker.read` returns the latest reading non-blocking, and
  returns ``present=False`` when no hand is in frame (never raises mid-stream).

``cv2`` and ``mediapipe`` are imported lazily inside the live class so the pure
function (and its tests) work without the ``vision-input`` extra installed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import mujoco
import numpy as np

from ai_teleop.common.log import get_logger

log = get_logger("hand_tracker")

# MediaPipe Hands landmark indices we use (of the 21-point model).
_WRIST = 0
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_PINKY_MCP = 17
_FINGERTIPS = (8, 12, 16, 20)  # index, middle, ring, pinky tips (thumb excluded)

# Empirical open/close bounds for the fingertip-spread ratio (tip→wrist distance
# over hand scale). Fist ≈ 1.0, flat open hand ≈ 2.4. ponytail: hand-tuned
# constants — the calibration knob the physical signal needs, not magic numbers.
_GRIP_RATIO_CLOSED = 1.0
_GRIP_RATIO_OPEN = 2.4


@dataclass(frozen=True)
class HandReading:
    """One frame of hand sensing, in camera/image-normalized space.

    Lives here (not in ``common/``) because only the input layer consumes it.

    Attributes
    ----------
    position:
        Shape (3,) — wrist landmark in MediaPipe's image-normalized frame:
        x,y in [0, 1] (origin top-left), z is relative depth (smaller ⇒ closer
        to camera). The strategy layer maps this to robot workspace.
    orientation:
        Shape (4,) unit quaternion (w, x, y, z) — a rough hand-frame estimate
        from the palm landmarks. Noisy; the strategy may ignore it.
    open_close:
        Grip proxy in [0, 1]: 0 = closed fist, 1 = flat open hand.
    present:
        False when no hand was detected this frame (drop-out).
    """

    position: np.ndarray
    orientation: np.ndarray
    open_close: float
    present: bool


_ABSENT = HandReading(
    position=np.zeros(3),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    open_close=0.0,
    present=False,
)


def reading_from_landmarks(landmarks: np.ndarray) -> HandReading:
    """Convert a (21, 3) MediaPipe landmark array to a :class:`HandReading`.

    Pure and camera-free — the unit-tested core. ``landmarks`` is the hand's 21
    points as (x, y, z) rows in MediaPipe's image-normalized frame.
    """
    if landmarks.shape != (21, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {landmarks.shape}")

    wrist = landmarks[_WRIST]
    position = wrist.copy()

    # Hand scale: wrist→middle-finger MCP. Used to normalize the grip ratio so
    # it is roughly distance-invariant (closer hand ⇒ bigger pixels, same grip).
    hand_scale = float(np.linalg.norm(landmarks[_MIDDLE_MCP] - wrist))
    if hand_scale < 1e-6:
        return HandReading(position, np.array([1.0, 0.0, 0.0, 0.0]), 0.0, True)

    tip_spread = float(np.mean([np.linalg.norm(landmarks[t] - wrist) for t in _FINGERTIPS]))
    ratio = tip_spread / hand_scale
    open_close = (ratio - _GRIP_RATIO_CLOSED) / (_GRIP_RATIO_OPEN - _GRIP_RATIO_CLOSED)
    open_close = float(np.clip(open_close, 0.0, 1.0))

    orientation = _palm_orientation(landmarks)
    return HandReading(position, orientation, open_close, present=True)


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

    rotation = np.column_stack([x_axis, y_axis, z_axis]).reshape(9)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation)
    mujoco.mju_normalize4(quat)
    return quat


class MediaPipeHandTracker:
    """Live webcam → :class:`HandReading` via OpenCV + MediaPipe Hands.

    Capture + inference run on a background thread so :meth:`read` is a cheap,
    non-blocking grab of the most recent reading — the 500 Hz control loop must
    not stall on a ~30 fps camera. ``read`` never raises; on drop-out (or before
    the first frame) it returns a reading with ``present=False``.

    ponytail: one daemon grabber thread + a lock. Fine for a single demo camera;
    if multi-camera or recording is ever needed, swap for a proper capture queue.
    """

    def __init__(
        self,
        *,
        camera_index: int = 0,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        target_fps: float = 30.0,
    ) -> None:
        # Lazy import: only the live path needs the vision-input extra.
        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open camera index {camera_index}")
        self._capture.set(cv2.CAP_PROP_FPS, target_fps)

        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self._latest = _ABSENT
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self._thread.start()
        log.info("MediaPipe hand tracker started (camera %d, ~%.0f fps)", camera_index, target_fps)

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                continue
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
            hand_landmarks = getattr(results, "multi_hand_landmarks", None)
            if not hand_landmarks:
                reading = _ABSENT
            else:
                points = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks[0].landmark], dtype=np.float64
                )
                reading = reading_from_landmarks(points)
            with self._lock:
                self._latest = reading

    def read(self) -> HandReading:
        """Latest hand reading (non-blocking). ``present=False`` on drop-out."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._capture.release()
        self._hands.close()
        log.info("MediaPipe hand tracker stopped")

    def __enter__(self) -> MediaPipeHandTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
