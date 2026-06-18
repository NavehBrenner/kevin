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


def test_position_is_the_wrist_landmark():
    points = _flat_open_hand()
    points[0] = [0.42, 0.61, -0.1]
    assert np.allclose(reading_from_landmarks(points).position, [0.42, 0.61, -0.1])


def test_reading_orientation_is_unit_quaternion():
    quat = reading_from_landmarks(_flat_open_hand()).orientation
    assert np.isclose(np.linalg.norm(quat), 1.0)


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


def test_dropout_holds_then_reengage_reanchors_without_jump():
    """Lift hand out (hold), bring it back at a new spot ⇒ no jump (re-anchor)."""
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

    vision.get_command(_observation(0.00))  # engage at EE x=0.5
    after_move = vision.get_command(_observation(0.02))  # hand +0.2 ⇒ EE ~0.7
    held = vision.get_command(_observation(0.04))  # drop-out ⇒ hold
    assert np.allclose(held.target_position, after_move.target_position)
    # re-entry at x=0.9 re-anchors to the held EE; staying put ⇒ no further jump
    vision.get_command(_observation(0.06))  # re-engage tick
    settled = vision.get_command(_observation(0.08))
    assert abs(settled.target_position[0] - held.target_position[0]) < 0.05


def test_grip_maps_open_to_release_and_fist_to_squeeze():
    open_hand = HandReading(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 1.0, present=True)
    fist = HandReading(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0]), 0.0, present=True)
    open_cmd = VisionInput(_FakeSource([open_hand]), grip_force=5.0).get_command(_observation())
    fist_cmd = VisionInput(_FakeSource([fist]), grip_force=5.0).get_command(_observation())
    assert open_cmd.delta_grip_force < 0  # release
    assert fist_cmd.delta_grip_force > 0  # squeeze
