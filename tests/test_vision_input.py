"""Unit tests for the M8 vision teleop: hand-tracking sensor + VisionInput.

Covers only the deterministic, camera-free core (LAB-50 landmark→reading math,
LAB-51 calibration / one-euro / clutch state machine). The live webcam path is
exercised manually — see `uv run kvn episode --input vision`.
"""

from __future__ import annotations

import numpy as np

from ai_teleop.common.observation import Observation
from ai_teleop.domain import InputStrategy
from ai_teleop.input import VisionInput, WorkspaceCalibration
from ai_teleop.input.hand_tracker import HandReading, reading_from_landmarks
from ai_teleop.input.vision_input import _OneEuroVector

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _flat_open_hand() -> np.ndarray:
    """A synthetic 21-landmark flat open hand: fingers splayed out from wrist."""
    points = np.zeros((21, 3))
    points[9] = [0.0, 0.3, 0.0]  # middle MCP — sets hand scale
    points[5] = [-0.15, 0.3, 0.0]  # index MCP
    points[17] = [0.15, 0.3, 0.0]  # pinky MCP
    for tip in (8, 12, 16, 20):  # fingertips far from wrist ⇒ open
        points[tip] = [0.0, 0.9, 0.0]
    return points


def _closed_fist() -> np.ndarray:
    points = _flat_open_hand()
    for tip in (8, 12, 16, 20):  # tips curled back near the MCPs ⇒ closed
        points[tip] = [0.0, 0.33, 0.0]
    return points


def _observation(sim_time: float = 0.0) -> Observation:
    return Observation(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_pose=np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        wrist_ft=np.zeros(6),
        gripper_width=0.08,
        peg_pose=np.zeros(7),
        hole_poses=np.zeros((1, 7)),
        target_hole_index=0,
        sim_time=sim_time,
    )


class _FakeSource:
    """Replays a scripted list of HandReadings, one per read()."""

    def __init__(self, readings: list[HandReading]) -> None:
        self._readings = readings
        self._i = 0

    def read(self) -> HandReading:
        reading = self._readings[min(self._i, len(self._readings) - 1)]
        self._i += 1
        return reading


# ---------------------------------------------------------------------------
# LAB-50 — landmark → reading
# ---------------------------------------------------------------------------


def test_open_close_scalar_distinguishes_open_and_fist():
    open_reading = reading_from_landmarks(_flat_open_hand())
    fist_reading = reading_from_landmarks(_closed_fist())
    assert open_reading.present and fist_reading.present
    assert open_reading.open_close > 0.8
    assert fist_reading.open_close < 0.2


def test_position_xy_is_wrist_and_z_is_depth_proxy():
    points = _flat_open_hand()
    points[0] = [0.42, 0.61, -0.1]  # move the wrist
    reading = reading_from_landmarks(points)
    assert np.allclose(reading.position[:2], [0.42, 0.61])  # x,y = wrist image coords
    assert reading.position[2] > 0.0  # z = apparent-hand-size depth proxy, not raw landmark z


def test_depth_proxy_grows_as_hand_appears_larger():
    small = _flat_open_hand()
    big = small * 2.0  # all landmarks twice as far apart ⇒ hand looks closer
    assert reading_from_landmarks(big).position[2] > reading_from_landmarks(small).position[2]


def test_reading_orientation_is_unit_quaternion():
    quat = reading_from_landmarks(_flat_open_hand()).orientation
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_point_direction_follows_where_the_hand_points():
    # Fingers below the wrist (image +y) ⇒ the hand points "down" in image coords.
    assert reading_from_landmarks(_flat_open_hand()).point_direction[1] > 0.5


def test_forwardness_positive_when_fingertips_angle_toward_camera():
    points = _flat_open_hand()
    for tip in (8, 12, 16, 20):
        points[tip, 2] = -0.2  # tips toward camera (negative MediaPipe z)
    assert reading_from_landmarks(points).forwardness > 0.0


# ---------------------------------------------------------------------------
# LAB-51 — calibration transform
# ---------------------------------------------------------------------------


def test_calibration_remaps_and_scales_axes():
    calib = WorkspaceCalibration(
        scale=np.array([1.0, 2.0, 3.0]),
        axis_map=(2, 0, 1),
        axis_sign=np.array([-1.0, 1.0, 1.0]),
    )
    # camera delta (cx, cy, cz) = (0.1, 0.2, 0.3)
    out = calib.map_delta(np.array([0.1, 0.2, 0.3]))
    # world x ← -1 * scale_x * cz; world y ← +2 * cx; world z ← +3 * cy
    assert np.allclose(out, [-1.0 * 0.3, 2.0 * 0.1, 3.0 * 0.2])


# ---------------------------------------------------------------------------
# LAB-51 — one-euro filter
# ---------------------------------------------------------------------------


def test_one_euro_passes_first_sample_through():
    f = _OneEuroVector(min_cutoff=1.0, beta=0.5)
    first = f(np.array([1.0, 2.0, 3.0]), timestamp=0.0)
    assert np.allclose(first, [1.0, 2.0, 3.0])


def test_one_euro_smooths_a_jump():
    f = _OneEuroVector(min_cutoff=0.5, beta=0.0)  # pure low-pass, no speed term
    f(np.zeros(3), timestamp=0.0)
    filtered = f(np.array([1.0, 0.0, 0.0]), timestamp=0.02)
    assert 0.0 < filtered[0] < 1.0  # lags the step ⇒ jitter suppressed


# ---------------------------------------------------------------------------
# LAB-51 — clutch / drop-out state machine
# ---------------------------------------------------------------------------


def test_visioninput_conforms_to_protocol():
    assert isinstance(VisionInput(_FakeSource([])), InputStrategy)


def test_holds_ee_pose_on_startup_dropout():
    """No hand yet ⇒ command holds the current EE pose, no jump."""
    absent = HandReading(np.zeros(3), np.array([1.0, 0, 0, 0]), 0.0, present=False)
    vision = VisionInput(_FakeSource([absent]))
    command = vision.get_command(_observation())
    assert np.allclose(command.target_position, [0.5, 0.0, 0.5])


def test_relative_mapping_moves_from_engage_anchor():
    """Engage anchors at current EE; subsequent hand motion is relative to it."""
    # Identity-ish calibration so we can predict the world delta.
    calib = WorkspaceCalibration(
        scale=np.array([1.0, 1.0, 1.0]), axis_map=(0, 1, 2), axis_sign=np.array([1.0, 1.0, 1.0])
    )
    anchor = HandReading(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.5, present=True)
    moved = HandReading(np.array([0.6, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.5, present=True)
    vision = VisionInput(
        _FakeSource([anchor, moved]), calibration=calib, mode="mirror", min_cutoff=50.0
    )

    vision.get_command(_observation(sim_time=0.0))  # engage, anchors at EE x=0.5
    command = vision.get_command(_observation(sim_time=0.02))  # +0.1 in camera x
    assert command.target_position[0] > 0.5 + 0.05  # moved in +x (filter lag aside)


def test_gain_amplifies_mapped_motion():
    """A larger gain moves the EE further for the same hand displacement."""
    calib = WorkspaceCalibration(
        scale=np.array([1.0, 1.0, 1.0]), axis_map=(0, 1, 2), axis_sign=np.array([1.0, 1.0, 1.0])
    )
    anchor = HandReading(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.5, present=True)
    moved = HandReading(np.array([0.6, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.5, present=True)

    def travel(gain: float) -> float:
        v = VisionInput(_FakeSource([anchor, moved]), calibration=calib, gain=gain, min_cutoff=50.0)
        v.get_command(_observation(0.0))
        return float(v.get_command(_observation(0.02)).target_position[0] - 0.5)

    assert travel(2.0) > 1.8 * travel(1.0)  # ~2x motion (filter lag aside)


def test_expo_softens_small_motion_vs_mirror():
    """Near the anchor, expo mode moves the EE *less* than plain mirror (precision)."""
    calib = WorkspaceCalibration(
        scale=np.array([1.0, 1.0, 1.0]), axis_map=(0, 1, 2), axis_sign=np.array([1.0, 1.0, 1.0])
    )
    anchor = HandReading(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.5, present=True)
    small = HandReading(np.array([0.55, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.5, present=True)

    def travel(mode: str) -> float:
        v = VisionInput(_FakeSource([anchor, small]), calibration=calib, mode=mode, min_cutoff=50.0)
        v.get_command(_observation(0.0))
        return float(v.get_command(_observation(0.02)).target_position[0] - 0.5)

    assert 0.0 < travel("expo") < travel("mirror")  # soft centre ⇒ smaller for small input


def _open_hand(
    direction: np.ndarray, *, forwardness: float = 0.0, open_close: float = 1.0
) -> HandReading:
    """An open hand (drives in rate mode) pointing `direction` in image coords."""
    return HandReading(
        np.array([0.5, 0.5, 0.0]),
        np.array([1.0, 0, 0, 0]),
        open_close,
        present=True,
        point_direction=direction,
        forwardness=forwardness,
    )


def _half_closed(open_close: float = 0.4) -> HandReading:
    """Neither open nor fist ⇒ the rate-mode lock state."""
    return HandReading(
        np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), open_close, present=True
    )


def test_rate_mode_steers_in_pointed_direction():
    """Pointing up (image −y) drives the EE up tick after tick (velocity)."""
    up = _open_hand(np.array([0.0, -1.0]))  # image y grows down ⇒ −y = up
    vision = VisionInput(_FakeSource([up, up, up, up]), mode="rate", leash=10.0, min_cutoff=50.0)
    vision.get_command(_observation(0.00))  # engage
    z1 = vision.get_command(_observation(0.02)).target_position[2]
    z2 = vision.get_command(_observation(0.04)).target_position[2]
    assert z2 > z1 > 0.5  # default axis_sign maps pointing-up → world +z


def test_rate_mode_pointing_into_camera_creeps_forward():
    """Forwardness above the dead-zone eases the EE forward (+x, default sign)."""
    at_cam = _open_hand(np.array([0.0, -1.0]), forwardness=0.3)
    vision = VisionInput(
        _FakeSource([at_cam, at_cam, at_cam]), mode="rate", leash=10.0, min_cutoff=50.0
    )
    vision.get_command(_observation(0.00))  # engage
    x1 = vision.get_command(_observation(0.02)).target_position[0]
    x2 = vision.get_command(_observation(0.04)).target_position[0]
    assert x2 > x1 > 0.5  # angling into the camera drives forward


def test_rate_mode_flat_pointing_does_not_creep_forward():
    """Pointing across the plane (forwardness ~0) stays within the dead-zone ⇒ no creep."""
    flat = _open_hand(np.array([1.0, 0.0]), forwardness=0.0)  # point sideways, no forward angle
    vision = VisionInput(_FakeSource([flat, flat, flat]), mode="rate", leash=10.0, min_cutoff=50.0)
    vision.get_command(_observation(0.00))
    vision.get_command(_observation(0.02))
    assert np.isclose(vision.get_command(_observation(0.04)).target_position[0], 0.5)


def test_rate_mode_fist_drives_backward():
    """A fist (curled, no point gesture) eases the EE backward (−x, default sign)."""
    fist = HandReading(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.0, present=True)
    vision = VisionInput(_FakeSource([fist, fist, fist]), mode="rate", leash=10.0, min_cutoff=50.0)
    vision.get_command(_observation(0.00))  # engage (fist is a drive gesture)
    x1 = vision.get_command(_observation(0.02)).target_position[0]
    x2 = vision.get_command(_observation(0.04)).target_position[0]
    assert x2 < x1 < 0.5  # fist retracts


def test_rate_mode_locks_and_freezes_at_arm_after_delay():
    """Relax the gesture past lock_delay ⇒ arm locks, frozen at its actual pose."""
    up = _open_hand(np.array([0.0, -1.0]))
    vision = VisionInput(
        _FakeSource([up, up, _half_closed(), _half_closed()]),
        mode="rate",
        lock_delay=0.0,  # lock immediately once the gesture is gone
        leash=10.0,
        min_cutoff=50.0,
    )
    vision.get_command(_observation(0.00))  # engage
    vision.get_command(_observation(0.02))  # drive up
    locked = vision.get_command(_observation(0.04)).target_position
    assert np.allclose(locked, _observation().ee_pose[:3])  # snapped to the arm, no catch-up


def test_rate_mode_holds_grip_constant():
    """Grip is parked in rate mode — it holds the seed across drive and lock."""
    vision = VisionInput(
        _FakeSource([_open_hand(np.array([0.0, -1.0])), _half_closed()]),
        mode="rate",
        grip_force=5.0,
        lock_delay=0.0,
        min_cutoff=50.0,
    )
    assert vision.get_command(_observation(0.00)).delta_grip_force == 0.0  # steering (open)
    assert vision.get_command(_observation(0.02)).delta_grip_force == 0.0  # locked (half-closed)


def test_dropout_freezes_at_current_ee_pose_then_reengage_no_jump():
    """Lift hand out ⇒ arm freezes at its current EE pose; re-entry re-anchors."""
    calib = WorkspaceCalibration(
        scale=np.array([1.0, 1.0, 1.0]), axis_map=(0, 1, 2), axis_sign=np.array([1.0, 1.0, 1.0])
    )
    present_a = HandReading(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.5, present=True)
    moved = HandReading(np.array([0.7, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.5, present=True)
    absent = HandReading(np.zeros(3), np.array([1.0, 0, 0, 0]), 0.0, present=False)
    # far-away hand position on re-entry — would jump if mapping were absolute
    reentry = HandReading(np.array([0.9, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.5, present=True)
    vision = VisionInput(
        _FakeSource([present_a, moved, absent, reentry, reentry]),
        calibration=calib,
        min_cutoff=50.0,
    )

    vision.get_command(_observation(0.00))  # engage
    vision.get_command(_observation(0.02))  # drive the arm
    held = vision.get_command(_observation(0.04))  # drop-out ⇒ freeze at current EE pose
    assert np.allclose(
        held.target_position, _observation().ee_pose[:3]
    )  # frozen at the arm, not target
    # re-entry far away re-anchors from the frozen pose ⇒ no jump
    vision.get_command(_observation(0.06))  # re-engage tick
    settled = vision.get_command(_observation(0.08))
    assert abs(settled.target_position[0] - held.target_position[0]) < 0.05


def test_grip_open_and_fist_are_opposite_and_open_releases():
    open_hand = HandReading(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 1.0, present=True)
    fist = HandReading(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.0, present=True)
    open_cmd = VisionInput(_FakeSource([open_hand]), grip_force=5.0).get_command(_observation())
    fist_cmd = VisionInput(_FakeSource([fist]), grip_force=5.0).get_command(_observation())
    # Opposite directions; on this gripper's convention open hand releases (+),
    # a fist squeezes (−). (Flip the sign in VisionInput if hardware disagrees.)
    assert open_cmd.delta_grip_force > 0  # release
    assert fist_cmd.delta_grip_force < 0  # squeeze
    assert open_cmd.delta_grip_force == -fist_cmd.delta_grip_force
