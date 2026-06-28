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

import time
from dataclasses import dataclass

from ai_teleop.common.observation import Observation
from ai_teleop.control import LockStatus
from ai_teleop.control.backbone import Controller
from ai_teleop.control.lock import LockState
from ai_teleop.domain import apply_delta
from ai_teleop.domain.interfaces import AssistProvider, InputStrategy
from ai_teleop.sim.scene import SimEnv

# Sim runs at 500 Hz (dt=2 ms in the MJCF). Headless: one control tick == one sim step.
# Render: one control tick advances physics enough steps to track wall-clock (see below).
SIM_DT = 0.002
DEFAULT_MAX_STEPS = 2000  # ~4 s of sim time — a full M3 episode budget.

# Render-path real-time pacing: cap how many physics steps one control tick may run to
# catch sim-time up to wall-time, so a one-off stall (GC, window resize) can't spiral into
# a long burst. If this binds repeatedly, physics genuinely can't hit 500 Hz on this box.
_MAX_CATCHUP_STEPS = 25  # 50 ms of sim time


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
    reset_episode_index: int | None = None,
    step_callback=None,
    stop_on_hold_lock: bool = False,
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
        render: when True, run the sim at wall-clock real time via catch-up
            substepping — each control tick advances physics enough steps to pin
            sim-time to elapsed wall-time, so the sim stays real-time despite
            per-tick GUI/vision cost (and the viewer refreshes at ~50 Hz). The
            controller's torque is held (ZOH) across a tick's substeps; control
            runs at the loop rate, physics at 500 Hz. Leave False for
            headless/batch: exactly one step per tick, deterministic, no sleep.
        reset_episode_index: forwarded to `environment.reset(...)` for the M4
            coverage-randomized reset (None ⇒ the deterministic home pose).
        step_callback: optional `f(step, observation, base_command, delta,
            command) -> bool`. Called each tick with the *pre-step* observation
            the assist acted on; this is the M4 data-generation logging hook
            (the loop stays logging-free itself). Returning a truthy value ends
            the episode early — how the driver signals a terminal condition.
        stop_on_hold_lock: when True, end the episode as soon as the controller's
            force-cap watchdog latches the HOLD lock. Once HOLD-latched the arm is
            frozen (nothing in this loop releases it), so every further step is a
            dead, identical frame — terminating avoids padding the trajectory with
            them. Off by default so interactive free-play (a wall bump shouldn't
            close the viewer) is unaffected; the data-gen and replay paths opt in.
    """
    observation = environment.reset(reset_episode_index)
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

        # How many physics steps this control tick advances. Headless/data-gen: exactly
        # one (deterministic). Render: enough to pin sim-time to elapsed wall-time, so the
        # sim runs real-time regardless of per-tick GUI/vision cost; torque is held across
        # the substeps. `behind` is computed absolutely (no incremental drift).
        if render:
            behind = round((time.monotonic() - wall_start) / SIM_DT) - sim_steps
            n_substeps = min(max(behind, 1), _MAX_CATCHUP_STEPS)
        else:
            n_substeps = 1

        for _ in range(n_substeps):
            environment.step()
            sim_steps += 1
            if sim_steps >= max_steps:
                break
        observation = environment.get_observation()
        control_ticks += 1

        if render:
            environment.sync_viewer()  # self-throttled to ~50 Hz
            # If we've caught up to wall-time, sleep until the next physics step is due
            # instead of busy-spinning the control loop (this is the metronome when work
            # is light; when work is heavy `ahead` is negative and the next tick catches up).
            ahead = (wall_start + sim_steps * SIM_DT) - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
        if stop:
            break
        if stop_on_hold_lock and controller.status.state is LockState.HOLD:
            break

    return EpisodeResult(
        final_observation=observation,
        lock_status=controller.status,
        n_steps=sim_steps,
    )
