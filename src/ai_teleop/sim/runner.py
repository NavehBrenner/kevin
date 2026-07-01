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

# Sim runs at 500 Hz (dt=2 ms in the MJCF). The loop is always physics-rate: exactly one
# base command + one controller recompute + one mj_step per iteration. This is what makes a
# replay reproduce its recording tick-for-tick — no wall-clock-dependent substepping.
SIM_DT = 0.002
DEFAULT_MAX_STEPS = 2000  # ~4 s of sim time — a full M3 episode budget.
DEFAULT_RENDER_FPS = 50.0  # target viewer frames per *sim* second when there's spare time.
DEFAULT_MIN_RENDER_FPS = 15.0  # floor — render at least this often even if it costs wall-time.


def _should_render(
    steps_since_render: int, frame_interval: int, floor_interval: int, slack: float
) -> bool:
    """Whether to sync the viewer this step. Never faster than the target rate
    (`frame_interval`); at the target when there's spare wall-time (`slack > 0`); and always
    at least at the floor rate (`floor_interval`) even when behind — the floor wins.
    """
    if steps_since_render < frame_interval:
        return False
    return slack > 0 or steps_since_render >= floor_interval


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
    render_fps: float = DEFAULT_RENDER_FPS,
    min_render_fps: float = DEFAULT_MIN_RENDER_FPS,
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
        render: when True, sync the interactive viewer at `render_fps`. Pacing is
            separate — see `time_factor`. Leave False for headless/batch.
        time_factor: pacing cap — the max sim:wall speed ratio, enforced by
            sleeping (never speeds up a slow machine). `inf` (default) = uncapped,
            as-fast-as-possible (headless/batch). `1.0` = real time. `<1` = slow
            motion, `>1` = fast-forward (up to what the box can step).
        render_fps: *target* viewer frames per second — the loop syncs the viewer
            at most this often (every `1/(render_fps·SIM_DT)` physics steps), and only
            when there's spare wall-time (we're ahead of the `time_factor` pace). When
            the box can't render this fast without slipping behind real time, the rate
            drops toward `min_render_fps` to give physics the time back. Ignored when
            `render=False`.
        min_render_fps: *floor* viewer frames per sim-second — render at least this
            often even when behind wall-time (so the viewer never freezes on a slow
            box; it just falls further behind — smooth slow-motion, not dropped frames).
            The floor wins over the spare-time gate. Keep it ≤ `render_fps`.
        step_callback: optional `f(step, observation, base_command, delta,
            command) -> bool`. Called each tick with the *pre-step* observation
            the assist acted on; this is the M4 data-generation logging hook
            (the loop stays logging-free itself). Returning a truthy value ends
            the episode early — how the driver signals a terminal condition.
    """
    observation = environment.reset()
    sim_steps = 0
    sim_time = 0.0
    control_ticks = 0
    wall_start = time.monotonic()
    # Viewer sync cadence, in physics steps: at most one frame per `frame_interval` steps
    # (the render_fps target), and a guaranteed one per `floor_interval` steps (the
    # min_render_fps floor). Both count sim-steps, so the floor can't starve; the wall
    # clock only *gates* the target rate, never physics — determinism is untouched.
    frame_interval = max(1, round(1.0 / (render_fps * SIM_DT)))
    floor_interval = max(frame_interval, round(1.0 / (min_render_fps * SIM_DT)))
    last_render_step = 0
    while sim_steps < max_steps:
        base_command = input_strategy.get_command(observation)
        delta = assist.get_delta(observation, base_command)
        command = apply_delta(base_command, delta)

        stop = False
        if step_callback is not None:
            stop = bool(step_callback(control_ticks, observation, base_command, delta, command))

        controller.compute(observation, command)

        environment.step()
        sim_steps += 1
        sim_time += SIM_DT

        observation = environment.get_observation()
        control_ticks += 1

        if render:
            # slack > 0 ⇒ ahead of the wall-clock cap (spare time to render); < 0 ⇒ behind.
            # Uncapped (time_factor=inf) has no wall target — always treat as spare.
            slack = (
                math.inf
                if math.isinf(time_factor)
                else wall_start + sim_time / time_factor - time.monotonic()
            )
            if _should_render(sim_steps - last_render_step, frame_interval, floor_interval, slack):
                environment.sync_viewer()
                last_render_step = sim_steps

        if not math.isinf(time_factor):
            # Sleep to hold the sim:wall speed cap (time_factor). Absolute target (recomputed
            # post-render), so a slow tick is absorbed by the next rather than drifting.
            # ponytail: ~1 ms sleep granularity at 500 Hz means a heavy box just lags real
            # time slightly (still one step per tick, still deterministic) — fine for
            # eyeballing. No fix until sub-ms pacing is actually needed.
            ahead = wall_start + sim_time / time_factor - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
        if stop:
            break

    return EpisodeResult(
        final_observation=observation,
        lock_status=controller.status,
        n_steps=sim_steps,
    )
