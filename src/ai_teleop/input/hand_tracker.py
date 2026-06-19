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
from dataclasses import dataclass, field

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
# (tip, pip) per finger — extension test for the two-finger "drive" gesture.
_INDEX = (8, 6)
_MIDDLE = (12, 10)
_RING = (16, 14)
_PINKY = (20, 18)

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
        Shape (3,) — (image_x, image_y, depth). ``image_x, image_y`` are the
        wrist landmark in MediaPipe's image-normalized frame (origin top-left,
        [0, 1]); ``depth`` is an apparent-hand-size proxy (wrist→middle-MCP
        distance) — *larger ⇒ hand closer to camera*. The size proxy replaces
        MediaPipe's unreliable landmark z, giving the strategy a usable
        forward/back axis. The strategy layer maps this to robot workspace.
    orientation:
        Shape (4,) unit quaternion (w, x, y, z) — a rough hand-frame estimate
        from the palm landmarks. Noisy; the strategy may ignore it.
    open_close:
        Grip proxy in [0, 1]: 0 = closed fist, 1 = flat open hand.
    present:
        False when no hand was detected this frame (drop-out).
    point_direction:
        Shape (2,) — unit in-plane direction the extended fingers point, in image
        coords (x right, y down), or zeros if not pointing. Position-independent:
        it's *where the hand points*, not where it sits. Drives the ``rate``
        "point to steer" mode. ``[0, 0]`` when degenerate.
    is_pointing:
        True for the two-finger drive gesture (index + middle extended, ring +
        pinky curled). In ``rate`` mode the arm moves only while this holds; any
        other hand shape freezes it (the "lock"). Ignored by mirror/expo.
    pitch:
        Forward/back signal: hand-pitch proxy, positive when the fingers tilt
        *toward* the camera (≈ "forward"). Derived from the extended fingertips'
        depth; noisy, so the strategy dead-zones it.
    """

    position: np.ndarray
    orientation: np.ndarray
    open_close: float
    present: bool
    point_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    is_pointing: bool = False
    pitch: float = 0.0


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

    # Apparent hand size (wrist→middle-finger MCP). Serves two purposes: it
    # normalizes the grip ratio (distance-invariant), and it *is* the depth proxy
    # for the forward/back axis — bigger ⇒ hand closer to the camera. We use it
    # instead of MediaPipe's raw landmark z, which is far too noisy to teleop with.
    hand_scale = float(np.linalg.norm(landmarks[_MIDDLE_MCP] - wrist))
    position = np.array([wrist[0], wrist[1], hand_scale])
    if hand_scale < 1e-6:
        return HandReading(position, np.array([1.0, 0.0, 0.0, 0.0]), 0.0, True)

    tip_spread = float(np.mean([np.linalg.norm(landmarks[t] - wrist) for t in _FINGERTIPS]))
    ratio = tip_spread / hand_scale
    open_close = (ratio - _GRIP_RATIO_CLOSED) / (_GRIP_RATIO_OPEN - _GRIP_RATIO_CLOSED)
    open_close = float(np.clip(open_close, 0.0, 1.0))

    orientation = _palm_orientation(landmarks)

    # Two-finger "drive" gesture + pointing direction (rate "point to steer").
    # A finger is "extended" when its tip is farther from the wrist than its PIP
    # joint, measured in the image plane — robust to hand orientation and dodges
    # the noisy landmark z.
    def extended(tip: int, pip: int) -> bool:
        return float(np.linalg.norm(landmarks[tip, :2] - wrist[:2])) > float(
            np.linalg.norm(landmarks[pip, :2] - wrist[:2])
        )

    is_pointing = (
        extended(*_INDEX) and extended(*_MIDDLE) and not extended(*_RING) and not extended(*_PINKY)
    )
    # In-plane pointing direction: knuckle midpoint → fingertip midpoint of the
    # two driving fingers, unit-normalized. Independent of hand position in frame.
    finger_base = (landmarks[_INDEX_MCP, :2] + landmarks[_MIDDLE_MCP, :2]) / 2.0
    finger_tip = (landmarks[_INDEX[0], :2] + landmarks[_MIDDLE[0], :2]) / 2.0
    point_vec = finger_tip - finger_base
    point_norm = float(np.linalg.norm(point_vec))
    point_direction = point_vec / point_norm if point_norm > 1e-6 else np.zeros(2)
    # Forward/back: MediaPipe z is wrist-origin, negative toward the camera, so
    # fingers tilted toward the camera ⇒ negative tip z ⇒ positive ("forward").
    pitch = -float(np.mean([landmarks[_INDEX[0], 2], landmarks[_MIDDLE[0], 2]]))

    return HandReading(
        position,
        orientation,
        open_close,
        present=True,
        point_direction=point_direction,
        is_pointing=is_pointing,
        pitch=pitch,
    )


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
        camera: int | str = 0,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        target_fps: float = 30.0,
        show_window: bool = False,
        mode: str = "mirror",
    ) -> None:
        # Lazy import: only the live path needs the vision-input extra. The legacy
        # `solutions` API (pinned mediapipe==0.10.21) bundles the landmark drawing
        # helpers the debug window uses.
        import cv2
        from mediapipe.python.solutions import drawing_styles, drawing_utils, hands

        self._cv2 = cv2
        self._mp_hands = hands
        self._mp_drawing = drawing_utils
        self._mp_styles = drawing_styles
        self._show_window = show_window
        self._mode = mode
        # `camera` is an int device index, or a string OpenCV can open: a stream
        # URL (e.g. an MJPEG/RTSP feed) or a device path. WSL2 has no UVC driver
        # for a host webcam, so there it must be a stream URL — see docs/cli.md.
        self._capture = cv2.VideoCapture(camera)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open camera source {camera!r}")
        self._capture.set(cv2.CAP_PROP_FPS, target_fps)

        self._hands = hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self._latest = _ABSENT
        # Annotated display frame produced by the capture thread; rendered from
        # read() (main thread) — see _render_window for why.
        self._display_frame: np.ndarray | None = None
        self._frame_id = 0
        self._last_rendered_id = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self._thread.start()
        log.info("MediaPipe hand tracker started (camera %r, ~%.0f fps)", camera, target_fps)

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
            display = self._annotate(frame, hand_landmarks, reading) if self._show_window else None
            with self._lock:
                self._latest = reading
                if display is not None:
                    self._display_frame = display
                    self._frame_id += 1

    def _annotate(
        self, frame: np.ndarray, hand_landmarks: object, reading: HandReading
    ) -> np.ndarray:
        """Draw MediaPipe landmarks + a teleop HUD onto the frame (no GUI calls)."""
        cv2 = self._cv2
        if hand_landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                hand_landmarks[0],  # type: ignore[index]
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )
        # Mirror for a natural selfie view (after drawing, so overlays match).
        frame = cv2.flip(frame, 1)

        # High-contrast semi-transparent HUD panel, pinned to the top-left corner.
        font = cv2.FONT_HERSHEY_SIMPLEX
        panel_w, panel_h = 470, 134
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        green, red, white, amber = (0, 230, 0), (60, 60, 255), (255, 255, 255), (0, 200, 255)
        if not reading.present:
            status, colour = "NO HAND - holding (clutched)", red
        elif self._mode == "rate" and not reading.is_pointing:
            status, colour = "LOCKED - point 2 fingers to drive", amber
        elif self._mode == "rate":
            status, colour = "DRIVING", green
        else:
            status, colour = f"TRACKING   hand {reading.open_close * 100:3.0f}% open", green
        cv2.putText(frame, status, (12, 26), font, 0.62, colour, 2)

        if self._mode == "rate":
            lines = [
                "Point index+middle: steer where you point",
                "Tilt fingers toward camera: forward / back",
                "Relax hand: lock    Fist/open (locked): grip",
            ]
        else:
            lines = [
                "Move hand: drive arm     Lift out of frame: clutch",
                "Toward camera: forward   Away from camera: back",
                "Open hand: release       Make a fist: grip",
            ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (12, 58 + i * 24), font, 0.48, white, 1)
        return frame

    def read(self) -> HandReading:
        """Latest hand reading (non-blocking). ``present=False`` on drop-out.

        Also pumps the debug window when enabled: OpenCV HighGUI only paints from
        the thread that owns the event loop, and read() is called from the main
        (control-loop) thread, so the actual imshow/waitKey happen here, not on
        the capture thread (which would render a blank window under WSLg/Qt).
        """
        with self._lock:
            reading = self._latest
            frame = self._display_frame
            frame_id = self._frame_id
        if self._show_window and frame is not None and frame_id != self._last_rendered_id:
            self._cv2.imshow("ai_teleop - hand tracking", frame)
            self._cv2.waitKey(1)  # required for HighGUI to actually render
            self._last_rendered_id = frame_id
        return reading

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._capture.release()
        self._hands.close()
        if self._show_window:
            self._cv2.destroyAllWindows()
        log.info("MediaPipe hand tracker stopped")

    def __enter__(self) -> MediaPipeHandTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
