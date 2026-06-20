"""Eval-trace schema + offline-replay tests (LAB-37).

The load-bearing property: a :class:`TrialObserver` driven **live** over a stream
and the same observer driven over a saved-then-loaded trace of that stream produce
the *identical* KPI record — i.e. offline replay recomputes KPIs with no re-run.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation
from ai_teleop.domain.delta import Delta
from ai_teleop.eval.ablation import replay_kpis
from ai_teleop.eval.observer import TrialObserver
from ai_teleop.eval.trace import EvalTraceRecorder, load_eval_trace, replay_trace

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
DT = 0.002


def _observation(peg_position, sim_time, *, wrist_ft=None, ee_position=None, hole_position=None):
    """Identity-orientation Observation (penetration reduces to tip_x − hole_x)."""
    peg_position = np.asarray(peg_position, dtype=float)
    hole_position = np.zeros(3) if hole_position is None else np.asarray(hole_position, float)
    ee = peg_position if ee_position is None else np.asarray(ee_position, float)
    return Observation(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_pose=np.concatenate([ee, IDENTITY_QUAT]),
        wrist_ft=np.zeros(6) if wrist_ft is None else np.asarray(wrist_ft, dtype=float),
        gripper_width=0.04,
        peg_pose=np.concatenate([peg_position, IDENTITY_QUAT]),
        hole_poses=np.concatenate([hole_position, IDENTITY_QUAT]).reshape(1, 7),
        target_hole_index=0,
        sim_time=sim_time,
    )


def _base_command() -> Command:
    return Command(np.array([0.1, 0.2, 0.3]), IDENTITY_QUAT.copy(), 1.5)


def _delta() -> Delta:
    return Delta(np.array([0.001, -0.002, 0.003]), np.array([0.01, 0.0, -0.01]), 0.5)


def _seating_stream():
    """A trajectory that approaches, makes contact, then sustains seating."""
    stream = []
    seated_peg = np.array([0.02, 0.0, -0.030])  # tip at +0.02 penetration, centred
    miss_peg = np.array([-0.05, 0.0, -0.030])
    # two pre-contact steps, then 40 seated steps (well past the 0.05 s sustain window)
    for index in range(2):
        ee = np.array([0.001 * index, 0.0, 0.0])
        stream.append(
            _observation(miss_peg, index * DT, wrist_ft=[6.0, 0, 0, 0, 0, 0], ee_position=ee)
        )
    for index in range(2, 42):
        ee = np.array([0.002 * index, 0.001 * index, 0.0])  # moving ⇒ non-zero jerk
        stream.append(
            _observation(seated_peg, index * DT, wrist_ft=[7.0, 0, 0, 0, 0, 0], ee_position=ee)
        )
    return stream


def _drive(observer, stream):
    for step, observation in enumerate(stream):
        if observer(step, observation, _base_command(), _delta(), None):
            break
    return observer.result()


def test_schema_round_trip(tmp_path):
    recorder = EvalTraceRecorder()
    for observation in _seating_stream():
        recorder.record(observation, _base_command(), _delta())
    path = tmp_path / "trace.npz"
    recorder.save(path, {"episode_index": 3, "config": "human_only"})

    columns, metadata = load_eval_trace(path)
    assert metadata["episode_index"] == 3
    assert metadata["config"] == "human_only"
    assert metadata["schema_version"] == "1.0"
    assert columns["wrist_ft"].shape == (len(recorder), 6)
    # Raw wrench survives unmodified (no bias subtraction in the producer).
    assert columns["wrist_ft"][-1, 0] == pytest.approx(7.0)


def test_replay_reconstructs_stream():
    stream = _seating_stream()
    recorder = EvalTraceRecorder()
    for observation in stream:
        recorder.record(observation, _base_command(), _delta())
    columns = {key: np.stack([row[key] for row in recorder._rows]) for key in recorder._rows[0]}

    reconstructed = list(replay_trace(columns))
    assert len(reconstructed) == len(stream)
    step, observation, base_command, delta = reconstructed[5]
    assert step == 5
    # Seating geometry inputs round-trip exactly.
    np.testing.assert_allclose(observation.peg_pose, stream[5].peg_pose)
    np.testing.assert_allclose(observation.wrist_ft, stream[5].wrist_ft)
    np.testing.assert_allclose(base_command.target_position, _base_command().target_position)
    np.testing.assert_allclose(delta.delta_position, _delta().delta_position)


def test_offline_replay_equals_live(tmp_path):
    """The headline property: replayed KPIs == live KPIs, bit-for-bit."""
    stream = _seating_stream()

    live = _drive(TrialObserver(seed=3, config_label="human_only"), stream)

    recorder = EvalTraceRecorder()
    for observation in stream:
        recorder.record(observation, _base_command(), _delta())
    path = tmp_path / "trace.npz"
    recorder.save(path, {"episode_index": 3, "config": "human_only"})
    replayed = replay_kpis(path)

    assert live.to_dict() == replayed.to_dict()
    assert live.success  # the stream was constructed to seat


def test_empty_trace_save_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty trace"):
        EvalTraceRecorder().save(tmp_path / "x.npz", {})


def test_record_validates_shapes():
    recorder = EvalTraceRecorder()
    bad = _observation([0.02, 0.0, -0.030], 0.0)
    object.__setattr__(bad, "ee_pose", np.zeros(3))  # wrong shape (7,)→(3,)
    with pytest.raises(ValueError, match="expected shape"):
        recorder.record(bad, _base_command(), _delta())
