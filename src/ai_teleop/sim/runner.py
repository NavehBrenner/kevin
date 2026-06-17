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
from ai_teleop.domain import apply_delta

# Sim runs at 500 Hz (dt=2 ms in the MJCF). One control tick == one sim step.
SIM_DT = 0.002
DEFAULT_MAX_STEPS = 2000  # ~4 s of sim time — a full M3 episode budget.


@dataclass(frozen=True)
class EpisodeResult:
    """What one episode leaves behind for the caller to inspect."""

    final_observation: Observation
    lock_status: LockStatus
    n_steps: int


def run_episode(
    environment,
    controller,
    input_strategy,
    assist,
    *,
    max_steps: int,
    render: bool = False,
    reset_episode_index: int | None = None,
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
        max_steps: episode step budget (one step == one sim/control tick).
        render: when True, sleep one timestep per tick so a `viewer`-mode env
            runs at roughly real time. Leave False for headless/batch.
        reset_episode_index: forwarded to `environment.reset(...)` for the M4
            coverage-randomized reset (None ⇒ the deterministic home pose).
        step_callback: optional `f(step, observation, base_command, delta,
            command) -> bool`. Called each tick with the *pre-step* observation
            the assist acted on; this is the M4 data-generation logging hook
            (the loop stays logging-free itself). Returning a truthy value ends
            the episode early — how the driver signals a terminal condition.
    """
    observation = environment.reset(reset_episode_index)
    steps = 0
    for step_index in range(max_steps):
        base_command = input_strategy.get_command(observation)
        delta = assist.get_delta(observation, base_command)
        command = apply_delta(base_command, delta)
        stop = False
        if step_callback is not None:
            stop = bool(step_callback(step_index, observation, base_command, delta, command))
        controller.compute(observation, command)
        environment.step()
        observation = environment.get_observation()
        steps += 1
        if render:
            time.sleep(SIM_DT)
        if stop:
            break

    return EpisodeResult(
        final_observation=observation,
        lock_status=controller.status,
        n_steps=steps,
    )
