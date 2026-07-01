"""M3 end-to-end runner — the reusable per-episode composition loop.

Wires the M3 stack together for one episode:

    InputStrategy → base Command → AssistProvider → apply_delta
        → Controller.compute → SimEnv.step

`run_episode(...)` is the single place the per-tick composition lives; the
`scripts/run_episode.py` CLI and the M4 data-generation pipeline
(`ai_teleop.data.generate`) both import it. It is deliberately free of console
/ logging side effects (no printing) so callers can wrap it — the data-gen
rollout drives it through `step_callback` to log trajectories.

The seam composes *around* the controller, not inside it: `Controller.compute`
still receives a single `Command` and knows nothing about a Δ source. Swapping
`NoAssist` for the expert (M4) or the learned residual (M5) is a one-argument
change here with no edit to the input strategy or the controller — the
dependency-inversion property M3 exists to establish.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ai_teleop.common.observation import Observation
from ai_teleop.control import LockStatus
from ai_teleop.control.backbone import Controller
from ai_teleop.domain import apply_delta
from ai_teleop.domain.interfaces import AssistProvider, InputStrategy
from ai_teleop.sim.scene import SimEnv

# Sim runs at 500 Hz (dt=2 ms in the MJCF). One control tick == one sim step (physics-rate
# control) unless catch-up substepping is enabled — see run_episode + _substeps below.
SIM_DT = 0.002
DEFAULT_MAX_STEPS = 2000  # ~4 s of sim time — a full M3 episode budget.

# Catch-up pacing: cap how many physics steps one control tick may run to catch sim-time
# up to wall-time, so a one-off stall (GC, window resize) can't spiral into a long burst.
# If this binds repeatedly, physics genuinely can't hit the target rate on this box.
_MAX_CATCHUP_STEPS = 25  # 50 ms of sim time


def _substeps(elapsed_wall: float, sim_steps: int, time_factor: float, allow_catchup: bool) -> int:
    """Physics steps this control tick should advance.

    Physics-rate control (``allow_catchup=False``, the default) always returns 1: one
    command + one controller recompute per physics step, exactly as headless generation
    ran — so a replay reproduces its recording tick-for-tick regardless of pacing. Pacing
    is then done purely by sleeping (see ``run_episode``).

    Catch-up substepping (``allow_catchup=True``) is for *expensive* live input (stereo
    vision) that can't produce a command every 2 ms tick: advance enough steps to pin
    sim-time to ``time_factor`` × elapsed wall-time, holding torque (ZOH) across them. This
    is non-deterministic (step count depends on machine load) — fine for live teleop, never
    for replay. ``time_factor=inf`` (uncapped) also collapses to 1: no wall-clock to track.
    """
    if not allow_catchup or math.isinf(time_factor):
        return 1
    behind = round(elapsed_wall * time_factor / SIM_DT) - sim_steps
    return min(max(behind, 1), _MAX_CATCHUP_STEPS)


@dataclass(frozen=True)
class EpisodeResult:
    """What one episode leaves behind for the caller to inspect."""

    final_observation: Observation
    lock_status: LockStatus
    n_steps: int


def run_episode(
    environment: SimEnv,
    controller: Controller,
    input_strategy: InputStrategy,
    assist: AssistProvider,
    *,
    max_steps: int,
    render: bool = False,
    time_factor: float = math.inf,
    allow_catchup: bool = False,
    step_callback=None,
) -> EpisodeResult:
    """Run one episode of the composed M3 loop; return the terminal state.

    The single per-tick composition — base command, correction Δ, combine,
    control, step — lives here and nowhere else. Anything that wants a
    different Δ source (NoAssist now, expert/residual later) passes a different
    `assist`; nothing else changes. Deliberately free of console/logging side
    effects so M4 can wrap it for data generation.

    Args:
        environment: a `SimEnv` (anything with reset/step/get_observation).
        controller: the M2 `Controller` (or any object exposing
            `compute(obs, command)` and a `status` property).
        input_strategy: an `InputStrategy` producing the base `Command`.
        assist: an `AssistProvider` producing the correction `Delta`.
        max_steps: episode budget in *physics* steps (sim-time = max_steps * SIM_DT).
        render: when True, sync the interactive viewer each tick (~50 Hz). Pacing
            is separate — see `time_factor`. Leave False for headless/batch.
        time_factor: pacing cap — the max sim:wall speed ratio, enforced by
            sleeping (never speeds up a slow machine). `inf` (default) = uncapped,
            as-fast-as-possible (headless/batch). `1.0` = real time. `<1` = slow
            motion, `>1` = fast-forward (up to what the box can step).
        allow_catchup: when True, expensive live input (stereo vision, which can't
            emit a command every 2 ms) may advance multiple physics steps per tick
            with torque held (ZOH) to keep sim-time tracking wall-time. This is
            non-deterministic (step count depends on load) — fine for live teleop,
            never for replay. Leave False (the default) for physics-rate control:
            exactly one command + controller recompute per physics step, so a replay
            reproduces its recording tick-for-tick at any `time_factor`.
        step_callback: optional `f(step, observation, base_command, delta,
            command) -> bool`. Called each tick with the *pre-step* observation
            the assist acted on; this is the M4 data-generation logging hook
            (the loop stays logging-free itself). Returning a truthy value ends
            the episode early — how the driver signals a terminal condition.
    """
    observation = environment.reset()
    sim_steps = 0
    control_ticks = 0
    wall_start = time.monotonic()
    while sim_steps < max_steps:
        base_command = input_strategy.get_command(observation)
        delta = assist.get_delta(observation, base_command)
        command = apply_delta(base_command, delta)

        stop = False
        if step_callback is not None:
            stop = bool(step_callback(control_ticks, observation, base_command, delta, command))

        controller.compute(observation, command)

        # How many physics steps this control tick advances (see _substeps): 1 for
        # physics-rate control (deterministic — replay reproduces its recording), or a
        # catch-up burst for expensive live input (vision). `elapsed` is absolute (no
        # incremental drift).
        n_substeps = _substeps(time.monotonic() - wall_start, sim_steps, time_factor, allow_catchup)

        for _ in range(n_substeps):
            environment.step()
            sim_steps += 1
            if sim_steps >= max_steps:
                break
        observation = environment.get_observation()
        control_ticks += 1

        if render:
            environment.sync_viewer()  # self-throttled to ~50 Hz
        if not math.isinf(time_factor):
            # Sleep to hold the sim:wall speed cap (time_factor). Absolute target, so a slow
            # tick is absorbed by the next rather than accumulating drift.
            # ponytail: ~1 ms sleep granularity at 500 Hz means a heavy box just lags real
            # time slightly (still one step per tick, still deterministic) — fine for
            # eyeballing. No fix until sub-ms pacing is actually needed.
            ahead = wall_start + sim_steps * SIM_DT / time_factor - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
        if stop:
            break

    return EpisodeResult(
        final_observation=observation,
        lock_status=controller.status,
        n_steps=sim_steps,
    )
