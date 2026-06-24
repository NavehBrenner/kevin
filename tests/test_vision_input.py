"""Unit tests for the M8 vision teleop: hand-tracking sensor + VisionInput.

Covers only the deterministic, camera-free core (LAB-50 landmark→reading math,
LAB-51 calibration / one-euro / clutch state machine). The live webcam path is
exercised manually — see `uv run kvn episode --input vision`.
"""

from __future__ import annotations

import numpy as np

from ai_teleop.common.observation import Observation
from ai_teleop.domain import InputStrategy
from ai_teleop.input import (
    NeutralAnchor,
    VisionInput,
    WorkspaceCalibration,
    calibrate_neutral,
)
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


def test_position_is_real_metric_wrist_xyz():
    points = _flat_open_hand()
    points[0] = [0.10, 0.20, 0.50]  # metric wrist xyz (metres in the rig frame)
    assert np.allclose(reading_from_landmarks(points).position, [0.10, 0.20, 0.50])


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
# Stereo upgrade — metric landmarks → reading
# ---------------------------------------------------------------------------


def test_scale_invariant_signals_present():
    """Grip + orientation are ratios/directions computed off the metric landmarks."""
    assert reading_from_landmarks(_flat_open_hand()).open_close > 0.8
    assert reading_from_landmarks(_closed_fist()).open_close < 0.2
    quat = reading_from_landmarks(_flat_open_hand()).orientation
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_metric_calibration_maps_camera_depth_to_robot_forward():
    calib = WorkspaceCalibration()  # the (now sole) metric default
    toward_camera = calib.map_delta(np.array([0.0, 0.0, -0.1]))  # metric depth shrinks
    away_from_camera = calib.map_delta(np.array([0.0, 0.0, 0.1]))
    assert toward_camera[0] > 0.0  # hand toward camera ⇒ robot forward
    assert away_from_camera[0] < 0.0  # hand away ⇒ robot back


# ---------------------------------------------------------------------------
# Open-palm pose sensor signal (consumed by the startup centering)
# ---------------------------------------------------------------------------


def test_open_palm_facing_sets_recenter_pose():
    assert reading_from_landmarks(_flat_open_hand()).recenter_pose
    assert not reading_from_landmarks(_closed_fist()).recenter_pose


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
    vision = VisionInput(_FakeSource([anchor, moved]), calibration=calib, min_cutoff=50.0)

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
    # Sustained drop-out (past the default 0.2 s grace) ⇒ clutch releases, freeze at EE.
    held = vision.get_command(_observation(0.30))
    assert np.allclose(
        held.target_position, _observation().ee_pose[:3]
    )  # frozen at the arm, not target
    # re-entry far away re-anchors from the frozen pose ⇒ no jump
    vision.get_command(_observation(0.32))  # re-engage tick
    settled = vision.get_command(_observation(0.34))
    assert abs(settled.target_position[0] - held.target_position[0]) < 0.05


def test_brief_dropout_within_grace_keeps_anchor():
    """A single-frame stereo miss must NOT re-anchor — motion resumes from the same
    reference, so sustained hand motion survives the constant stereo blips."""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    p0 = np.zeros(3)
    p1 = np.array([0.10, 0.0, 0.0])  # hand moved 10 cm from the engage anchor
    present0 = HandReading(p0, quat, 0.5, present=True)
    present1 = HandReading(p1, quat, 0.5, present=True)
    absent = HandReading(np.zeros(3), quat, 0.0, present=False)
    vision = VisionInput(
        _FakeSource([present0, present1, absent, present1]),
        dropout_grace_s=0.2,
        min_cutoff=1e6,  # defeat the one-euro filter for an exact comparison
    )
    ee = _observation().ee_pose[:3]
    vision.get_command(_observation(0.00))  # engage, anchor = p0
    moved = vision.get_command(_observation(0.05)).target_position  # tracking p1
    vision.get_command(_observation(0.10))  # one missed frame, within grace ⇒ hold
    after = vision.get_command(_observation(0.15)).target_position  # reappear at p1
    assert float(np.linalg.norm(moved - ee)) > 0.05  # it really did move off home
    assert np.allclose(after, moved, atol=1e-6)  # same anchor ⇒ no re-anchor snap-back


def test_sustained_dropout_past_grace_reanchors():
    """Beyond the grace window the clutch releases; reappearing re-anchors (snap to EE)."""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    p1 = np.array([0.10, 0.0, 0.0])
    present0 = HandReading(np.zeros(3), quat, 0.5, present=True)
    present1 = HandReading(p1, quat, 0.5, present=True)
    absent = HandReading(np.zeros(3), quat, 0.0, present=False)
    vision = VisionInput(
        _FakeSource([present0, present1, absent, present1]),
        dropout_grace_s=0.05,
        min_cutoff=1e6,
    )
    ee = _observation().ee_pose[:3]
    vision.get_command(_observation(0.00))  # engage
    vision.get_command(_observation(0.10))  # tracking p1
    vision.get_command(_observation(0.30))  # absent, 0.20 s > grace ⇒ release
    after = vision.get_command(_observation(0.35)).target_position  # reappear ⇒ re-anchor
    assert np.allclose(after, ee, atol=1e-6)  # re-anchored: hand maps to current EE


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


def test_clutch_transitions_are_logged(caplog):
    """Engage on first detection, release on a sustained drop-out — logged once each
    (not per tick), so the operator sees the clutch state in the terminal."""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    present = HandReading(np.zeros(3), quat, 0.5, present=True)
    absent = HandReading(np.zeros(3), quat, 0.0, present=False)
    strategy = VisionInput(_FakeSource([present, absent, absent]))
    with caplog.at_level("INFO", logger="ai_teleop.vision_input"):
        strategy.get_command(_observation(0.0))  # first present ⇒ engage
        strategy.get_command(_observation(0.1))  # absent, within grace ⇒ silent
        strategy.get_command(_observation(1.0))  # absent, past grace ⇒ release
    messages = [r.message for r in caplog.records]
    assert sum("clutch engaged" in m for m in messages) == 1
    assert sum("clutch released" in m for m in messages) == 1


def test_calibrate_neutral_averages_held_position():
    """A held open palm sets neutral; the anchor is the mean position over the hold window."""
    pos = np.array([0.1, 0.2, 0.3])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    reading = HandReading(pos, quat, 1.0, present=True, recenter_pose=True)
    pumped = []
    clock = iter(np.arange(0.0, 100.0, 1.0))
    anchor = calibrate_neutral(
        _FakeSource([reading]),
        hold_s=3.0,
        clock=lambda: float(next(clock)),
        sleep=lambda _s: None,
        on_tick=lambda: pumped.append(1),
    )
    assert np.allclose(anchor.hand_position, pos)
    assert np.allclose(anchor.hand_orientation, quat)
    assert pumped  # the window/viewer pump callback ran while waiting


def test_calibrate_neutral_restarts_when_pose_drops():
    """Losing the open-palm pose mid-hold restarts the countdown (no premature neutral)."""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    held = HandReading(np.zeros(3), quat, 1.0, present=True, recenter_pose=True)
    gap = HandReading(np.zeros(3), quat, 0.0, present=False)  # pose lost ⇒ restart
    # held, held (1s), gap (restart), then held*4 ⇒ a full 3 s hold from the restart.
    readings = [held, held, gap, held, held, held, held]
    clock = iter(np.arange(0.0, 100.0, 1.0))
    anchor = calibrate_neutral(
        _FakeSource(readings),
        hold_s=3.0,
        clock=lambda: float(next(clock)),
        sleep=lambda _s: None,
    )
    assert np.allclose(anchor.hand_position, np.zeros(3))  # completed only after the restart


def test_initial_anchor_holds_home_orientation_without_snapping():
    """With a startup neutral, a hand already at the anchor orientation keeps the wrist at
    home — orientation is mirrored relatively, so engaging never snaps to the hand's pose."""
    home_quat = _observation().ee_pose[3:]  # [1, 0, 0, 0]
    hand_quat = np.array([0.7071, 0.0, 0.7071, 0.0])  # a 90° hand rotation
    hand_quat = hand_quat / np.linalg.norm(hand_quat)
    pos = np.array([0.1, 0.0, 0.0])
    reading = HandReading(pos, hand_quat, 0.5, present=True)
    vision = VisionInput(
        _FakeSource([reading]),
        track_orientation=True,
        initial_anchor=NeutralAnchor(pos.copy(), hand_quat.copy()),
    )
    cmd = vision.get_command(_observation())
    assert np.allclose(cmd.target_quaternion, home_quat, atol=1e-6)  # no snap


def test_calibrate_neutral_tolerates_brief_pose_flicker():
    """A single low-confidence frame within pose_grace_s must not restart the hold —
    mirrors stereohand's presence window (a blink shouldn't reset the countdown)."""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    held = HandReading(np.zeros(3), quat, 1.0, present=True, recenter_pose=True)
    flicker = HandReading(np.zeros(3), quat, 1.0, present=True)  # pose flag dropped one frame
    readings = [held, held, flicker, held, held, held, held]
    clock = iter(np.arange(0.0, 100.0, 0.05))  # 0.05 s/iter < pose_grace_s ⇒ blink survives
    anchor = calibrate_neutral(
        _FakeSource(readings),
        hold_s=0.3,
        pose_grace_s=0.15,
        clock=lambda: float(next(clock)),
        sleep=lambda _s: None,
    )
    assert np.allclose(anchor.hand_position, np.zeros(3))  # completed despite the flicker
