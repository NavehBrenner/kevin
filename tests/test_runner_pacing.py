"""Unit tests for `_substeps` — the physics-steps-per-tick pacing math (LAB-88).

Physics-rate control (the default, `allow_catchup=False`) must always advance exactly one
step per tick so a replay reproduces its recording deterministically; catch-up substepping
(`allow_catchup=True`, for expensive live vision input) advances a bounded, time_factor-
scaled burst to keep sim-time tracking wall-time. Pure function, no real sleeping.
"""

from __future__ import annotations

import math

from ai_teleop.sim.runner import _MAX_CATCHUP_STEPS, SIM_DT, _substeps


def test_physics_rate_control_is_always_one_step():
    # allow_catchup=False: one step per tick regardless of how far behind wall-time — this is
    # the determinism contract (replay consumes one recorded command per physics step).
    assert _substeps(elapsed_wall=10.0, sim_steps=0, time_factor=1.0, allow_catchup=False) == 1
    assert _substeps(elapsed_wall=0.0, sim_steps=999, time_factor=0.3, allow_catchup=False) == 1


def test_uncapped_time_factor_is_always_one_step():
    # inf time_factor = no wall-clock to track, so nothing to catch up to — even with catchup on.
    assert _substeps(elapsed_wall=10.0, sim_steps=0, time_factor=math.inf, allow_catchup=True) == 1


def test_catchup_advances_to_pin_sim_time_to_wall_time():
    # 20 ms of wall-time elapsed at real time, 0 steps done → catch up ~10 physics steps (2 ms each).
    assert _substeps(elapsed_wall=0.020, sim_steps=0, time_factor=1.0, allow_catchup=True) == 10
    # On pace (elapsed == sim_steps * SIM_DT) → the floor of 1 step.
    assert (
        _substeps(elapsed_wall=100 * SIM_DT, sim_steps=100, time_factor=1.0, allow_catchup=True)
        == 1
    )


def test_catchup_scales_with_time_factor():
    # Same wall-time, but a 2x fast-forward asks for twice the physics steps.
    assert _substeps(elapsed_wall=0.010, sim_steps=0, time_factor=1.0, allow_catchup=True) == 5
    assert _substeps(elapsed_wall=0.010, sim_steps=0, time_factor=2.0, allow_catchup=True) == 10


def test_catchup_is_capped():
    # A long stall can't spiral into an unbounded burst.
    assert (
        _substeps(elapsed_wall=100.0, sim_steps=0, time_factor=1.0, allow_catchup=True)
        == _MAX_CATCHUP_STEPS
    )
