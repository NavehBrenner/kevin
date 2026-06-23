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
from ai_teleop.domain import apply_delta
from ai_teleop.domain.interfaces import AssistProvider, InputStrategy
from ai_teleop.sim.scene import SimEnv

# Sim runs at 500 Hz (dt=2 ms in the MJCF). One control tick == one sim step.
SIM_DT = 0.002
DEFAULT_MAX_STEPS = 2000  # ~4 s of sim time — a full M3 episode budget.

# Passive-viewer refresh rate, decoupled from the 500 Hz sim (render path only). The
# window only needs ~30-60 Hz; syncing every 2 ms step saturates WSLg's GUI pipe.
VIEWER_FPS = 50.0
VIEWER_FRAME_PERIOD = 1.0 / VIEWER_FPS


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
        render: when True, pace the loop to wall-clock (deadline sleep per tick)
            so a `viewer`-mode env runs at real time, and drive the viewer at
            VIEWER_FPS. Leave False for headless/batch (no sleep, no sync).
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
    # Real-time pacing (render path only). Each tick should consume exactly SIM_DT
    # of wall-clock, so we sleep the *remainder* of the budget after this tick's
    # work — not a fixed sleep(SIM_DT), which ignores work time and overshoots
    # (~0.69x real-time on WSL, turning sim-time-anchored teleop gestures sluggish).
    next_tick = time.monotonic() + SIM_DT
    next_frame_time = 0.0  # wall-clock deadline for the next viewer sync (0 ⇒ sync now)
    start_time = time.monotonic()
    lost_time = 0.0
    frozen_ticks = 0
    for step_index in range(max_steps):
        base_command = input_strategy.get_command(observation)
        delta = assist.get_delta(observation, base_command)
        command = apply_delta(base_command, delta)

        stop = False
        if step_callback is not None:
            stop = bool(step_callback(step_index, observation, base_command, delta, command))

        controller.compute(observation, command)
        # Viewer frame clock (render path only): sync at VIEWER_FPS off a wall-clock
        # deadline, not every step — decoupled from the sim rate so the display holds
        # ~50 Hz whether the sim is real-time, slow, or bursting through a catch-up.
        now = time.monotonic()
        sync_viewer = render and now >= next_frame_time
        if sync_viewer:
            next_frame_time = now + VIEWER_FRAME_PERIOD
        environment.step(sync_viewer=sync_viewer)
        observation = environment.get_observation()
        steps += 1
        if render:
            # Sleep only the time left in this 2 ms budget. If the tick overran it
            # (e.g. a viewer sync — SimEnv.step throttles those to ~50 Hz), drop the
            # debt rather than banking it: one-step-per-tick can't claw lost sim-time
            # back, only avoid compounding it.
            # ponytail: deadline pacing fixes sleep *overshoot* (the headless 0.69x),
            # not a chronically over-budget tick. If the live loop still runs slow,
            # the upgrade is a catch-up substep loop (step round(behind/SIM_DT) times).
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
                next_tick += SIM_DT
            else:
                lost_time += now - next_tick
                total_time = now - start_time
                frozen_ticks += 1
                lost_tick_rate = frozen_ticks / total_time
                print(
                    f"lost {lost_time:.3f}s so far over {total_time:.3f}s total time - {frozen_ticks} lost ticks toatl, at {lost_tick_rate:.3f} lost ticks/sec"
                )

                next_tick = now + SIM_DT
        if stop:
            break

    return EpisodeResult(
        final_observation=observation,
        lock_status=controller.status,
        n_steps=steps,
    )
