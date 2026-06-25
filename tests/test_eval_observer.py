"""Unit tests for the M6 passive-observer harness (LAB-36).

The observer is driven exactly as ``run_episode`` drives it — called with the
``(step, observation, base_command, delta, command)`` signature — so these feed
hand-built ``Observation`` sequences and assert the resulting ``TrialKPIs``.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from ai_teleop.common.observation import Observation
from ai_teleop.common.seating import PEG_HALF_LENGTH, SeatingGeometry
from ai_teleop.eval.observer import TrialObserver
from ai_teleop.eval.schema import TrialKPIs, TrialOutcome

_SCENE_PATH = pathlib.Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "full_scene.xml"

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
SIM_DT = 0.002  # one control tick, matches sim.runner.SIM_DT


def _observation(
    *,
    peg_position: np.ndarray,
    sim_time: float,
    wrist_ft: np.ndarray | None = None,
    ee_position: np.ndarray | None = None,
    hole_position: np.ndarray | None = None,
) -> Observation:
    """An Observation with identity peg/hole orientation.

    With identity quaternions the peg's +z axis is world +z (tip =
    ``peg_position + 0.030·ẑ``) and the hole's insertion axis is world +x, so
    penetration reduces to ``tip_x − hole_x`` — easy to reason about by hand.
    """
    peg_position = np.asarray(peg_position, dtype=float)
    hole_position = np.zeros(3) if hole_position is None else np.asarray(hole_position, dtype=float)
    return Observation(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_pose=np.concatenate([
            peg_position if ee_position is None else np.asarray(ee_position, float),
            IDENTITY_QUAT,
        ]),
        wrist_ft=np.zeros(6) if wrist_ft is None else np.asarray(wrist_ft, dtype=float),
        gripper_width=0.08,
        peg_pose=np.concatenate([peg_position, IDENTITY_QUAT]),
        hole_poses=np.concatenate([hole_position, IDENTITY_QUAT]).reshape(1, 7),
        target_hole_index=0,
        sim_time=sim_time,
    )


# A peg whose tip is 0.02 m past the hole entry, perfectly centred laterally:
# tip = [0.02, 0, 0] once the +0.030 z offset is cancelled by peg_z = −0.030.
_SEATED_PEG = np.array([0.02, 0.0, -PEG_HALF_LENGTH])
# A peg short of the hole: tip_x = −0.05 → negative penetration.
_MISS_PEG = np.array([-0.05, 0.0, -PEG_HALF_LENGTH])


def _drive(observer: TrialObserver, observations: list[Observation]) -> TrialKPIs:
    """Feed a sequence through the observer (stopping early if it signals end)."""
    for step, observation in enumerate(observations):
        if observer(step, observation, None, None, None):
            break
    return observer.result()


# ---------------------------------------------------------------------------
# Seating geometry — hand-computed penetration / lateral error
# ---------------------------------------------------------------------------


def test_seating_geometry_penetration_matches_hand_value():
    geometry = SeatingGeometry.from_observation(
        _observation(peg_position=_SEATED_PEG, sim_time=0.0)
    )
    assert geometry.penetration == pytest.approx(0.02)
    assert geometry.lateral_error == pytest.approx(0.0, abs=1e-12)


def test_seating_geometry_lateral_error_is_off_axis_distance():
    # Shift the peg 0.004 m in y → that becomes pure lateral error.
    peg = _SEATED_PEG + np.array([0.0, 0.004, 0.0])
    geometry = SeatingGeometry.from_observation(_observation(peg_position=peg, sim_time=0.0))
    assert geometry.penetration == pytest.approx(0.02)
    assert geometry.lateral_error == pytest.approx(0.004)


# ---------------------------------------------------------------------------
# Success / miss / force-abort classification
# ---------------------------------------------------------------------------


def test_sustained_seating_classified_success():
    observer = TrialObserver(sustained_duration_s=0.02)
    # 0.02 s at 0.002 s/step = held seated for 10+ steps.
    observations = [_observation(peg_position=_SEATED_PEG, sim_time=i * SIM_DT) for i in range(20)]
    kpis = _drive(observer, observations)
    assert kpis.outcome is TrialOutcome.SUCCESS
    assert kpis.success is True
    assert kpis.time_to_insert_s == pytest.approx(0.0, abs=1e-9)


def test_never_seated_classified_timeout():
    observer = TrialObserver()
    observations = [_observation(peg_position=_MISS_PEG, sim_time=i * SIM_DT) for i in range(20)]
    kpis = _drive(observer, observations)
    assert kpis.outcome is TrialOutcome.TIMEOUT
    assert kpis.success is False
    assert kpis.time_to_insert_s is None


def test_transient_overshoot_is_not_success():
    """Seated for one step then popping back out must not count as insertion."""
    observer = TrialObserver(sustained_duration_s=0.05)
    observations = [
        _observation(peg_position=_MISS_PEG, sim_time=0 * SIM_DT),
        _observation(peg_position=_SEATED_PEG, sim_time=1 * SIM_DT),  # brief touch
        _observation(peg_position=_MISS_PEG, sim_time=2 * SIM_DT),  # back out
        _observation(peg_position=_MISS_PEG, sim_time=3 * SIM_DT),
    ]
    kpis = _drive(observer, observations)
    assert kpis.outcome is TrialOutcome.TIMEOUT


def test_force_over_cap_classified_force_abort():
    observer = TrialObserver(force_cap=50.0)
    bias = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # static offset tared on step 0
    observations = [
        _observation(peg_position=_MISS_PEG, sim_time=0.0, wrist_ft=bias),
        _observation(
            peg_position=_MISS_PEG, sim_time=SIM_DT, wrist_ft=bias + np.array([60.0, 0, 0, 0, 0, 0])
        ),
    ]
    kpis = _drive(observer, observations)
    assert kpis.outcome is TrialOutcome.FORCE_ABORT


# ---------------------------------------------------------------------------
# KPIs — peak contact force, contact events, smoothness
# ---------------------------------------------------------------------------


def test_peak_contact_force_is_bias_subtracted():
    observer = TrialObserver()
    bias = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    observations = [
        _observation(peg_position=_MISS_PEG, sim_time=0.0, wrist_ft=bias),  # tare here
        _observation(
            peg_position=_MISS_PEG,
            sim_time=SIM_DT,
            wrist_ft=bias + np.array([3.0, 4.0, 0, 0, 0, 0]),
        ),
    ]
    kpis = _drive(observer, observations)
    # |(3,4,0)| = 5 after removing the 2 N static bias.
    assert kpis.peak_contact_force == pytest.approx(5.0)


def test_contact_events_counted_with_hysteresis():
    observer = TrialObserver(contact_force_floor=5.0, contact_release_floor=2.5)
    forces = [0.0, 6.0, 7.0, 1.0, 0.0, 8.0]  # two distinct rising edges above 5 N
    observations = [
        _observation(
            peg_position=_MISS_PEG, sim_time=i * SIM_DT, wrist_ft=np.array([f, 0, 0, 0, 0, 0])
        )
        for i, f in enumerate(forces)
    ]
    kpis = _drive(observer, observations)
    assert kpis.contact_events == 2


def test_straight_constant_velocity_path_has_near_zero_jerk():
    observer = TrialObserver()
    observations = [
        _observation(
            peg_position=_MISS_PEG, sim_time=i * SIM_DT, ee_position=np.array([0.1 * i, 0.0, 0.0])
        )
        for i in range(10)
    ]
    kpis = _drive(observer, observations)
    assert kpis.jerk_integral == pytest.approx(0.0, abs=1e-6)


def test_jerk_integral_zero_for_short_trace():
    observer = TrialObserver()
    observations = [_observation(peg_position=_MISS_PEG, sim_time=i * SIM_DT) for i in range(3)]
    kpis = _drive(observer, observations)
    assert kpis.jerk_integral == 0.0
    assert kpis.n_steps == 3


# ---------------------------------------------------------------------------
# Trial-boundary detection (reuse across episodes)
# ---------------------------------------------------------------------------


def test_sim_time_reset_starts_a_fresh_trial():
    observer = TrialObserver(sustained_duration_s=0.0)
    # First (incomplete) episode, then sim_time jumps back → a clean second trial.
    first = [_observation(peg_position=_MISS_PEG, sim_time=i * SIM_DT) for i in range(5)]
    for step, observation in enumerate(first):
        observer(step, observation, None, None, None)
    # New episode: a single seated step at sim_time 0 should now register success
    # with a clean accumulator (n_steps == 1), not carry the previous 5 steps.
    observer(0, _observation(peg_position=_SEATED_PEG, sim_time=0.0), None, None, None)
    kpis = observer.result()
    assert kpis.outcome is TrialOutcome.SUCCESS
    assert kpis.n_steps == 1


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


def test_trial_kpis_dict_round_trip():
    kpis = TrialKPIs(
        outcome=TrialOutcome.SUCCESS,
        time_to_insert_s=1.5,
        peak_contact_force=12.0,
        contact_events=3,
        jerk_integral=0.4,
        n_steps=750,
        duration_s=1.5,
        seed=99,
        config_label="residual",
    )
    record = kpis.to_dict()
    assert record["outcome"] == "success"
    assert record["success"] is True
    assert TrialKPIs.from_dict(record) == kpis


# ---------------------------------------------------------------------------
# Integration: the observer plugs into the real run_episode as a step_callback
# ---------------------------------------------------------------------------


def test_observer_runs_as_step_callback_through_run_episode():
    """Drive the observer through the real composed loop; assert a coherent record.

    Proves the observer consumes live ``Observation``s with the runner's exact
    callback signature and emits a well-formed ``TrialKPIs`` — the integration
    contract. (Whether a real episode *seats* is a 72%-expert / stochastic-scene
    question covered by LAB-37's ablation, not asserted here.)
    """
    if not _SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {_SCENE_PATH}")

    from ai_teleop.control import Controller
    from ai_teleop.domain import NoAssist
    from ai_teleop.input import ScriptedNoisyHuman
    from ai_teleop.sim.runner import run_episode
    from ai_teleop.sim.scene import SimEnv

    environment = SimEnv(str(_SCENE_PATH), render_mode="headless")
    try:
        controller = Controller(environment)
        start = environment.reset()
        target = np.concatenate([
            start.hole_poses[start.target_hole_index][:3],
            controller.home_pose[3:],
        ])
        human = ScriptedNoisyHuman(target, seed=0)
        observer = TrialObserver(seed=0, config_label="human_only")
        run_episode(
            environment, controller, human, NoAssist(), max_steps=400, step_callback=observer
        )
    finally:
        environment.close()

    kpis = observer.result()
    assert kpis.outcome in tuple(TrialOutcome)
    assert 1 <= kpis.n_steps <= 400
    assert kpis.peak_contact_force >= 0.0
    assert kpis.jerk_integral >= 0.0
    assert kpis.duration_s >= 0.0
    assert kpis.seed == 0 and kpis.config_label == "human_only"


# ---------------------------------------------------------------------------
# Dependency-Inversion: eval/ and control/ do not import each other
# ---------------------------------------------------------------------------


def _module_dir(module_name: str) -> pathlib.Path:
    import importlib

    return pathlib.Path(importlib.import_module(module_name).__file__).parent


# The eval *measurement core* — the observer + its record/trace contracts — must
# never reach into the controller (the LAB-36 DIP pillar: trial/success/KPI concepts
# stay independent of the control stack). `ablation.py` is deliberately exempt: it is
# the experiment *orchestrator* and, exactly like `data/generate.py`, composes
# SimEnv + Controller + operator + assist to drive `run_episode`. The load-bearing
# direction — control/ never importing eval/ — is asserted in full below and still
# holds; the controller stays mode-less and trial-unaware.
_MEASUREMENT_CORE = ("observer.py", "schema.py", "trace.py")


def test_eval_measurement_core_does_not_import_control():
    eval_dir = _module_dir("ai_teleop.eval")
    for filename in _MEASUREMENT_CORE:
        source = (eval_dir / filename).read_text()
        matches = re.findall(r"^\s*(?:import|from)\s+ai_teleop\.control", source, re.MULTILINE)
        assert not matches, f"control import found in eval/{filename}"


def test_control_does_not_import_eval():
    control_dir = _module_dir("ai_teleop.control")
    for python_file in control_dir.rglob("*.py"):
        source = python_file.read_text()
        matches = re.findall(r"^\s*(?:import|from)\s+ai_teleop\.eval", source, re.MULTILINE)
        assert not matches, f"eval import found in control/{python_file.name}"
