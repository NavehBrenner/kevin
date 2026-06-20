"""Paired-seed ablation runner — the mechanism behind the M6 head-to-head (LAB-37).

One *trial* is a fixed ``(master_seed, episode_index)`` pair: it pins the
randomized scene (``SimEnv.reset`` derives it from exactly that pair) and the
scripted operator (a same-seeded :class:`ScriptedNoisyHuman`, which is **open-loop**
— its command stream depends only on its seed and tick, never on the realized
observation). Running that trial once per *config* (e.g. ``NoAssist`` vs the learned
residual) therefore changes **only the assist layer** — identical scene, identical
operator command stream — which is the zero-operator-variance pairing that gives the
ablation its statistical power.

Each trial is observed live by a :class:`TrialObserver` (computes the KPIs and ends
the episode on seating / force-abort) and, optionally, persisted as an
:class:`EvalTraceRecorder` trace so the KPIs can be recomputed offline without
re-running (see :mod:`ai_teleop.eval.trace`). This module owns the *mechanism*; the
~100-trial run against the fine-tuned residual and the difficulty pin are LAB-53.

This stays a pure consumer of the M3 runner and the M5 assist seam — no controller
edit, mirroring the data-gen rollout. The configs are supplied by the caller as
``(label, assist_factory)`` so ``eval/`` never imports a concrete policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_teleop.control import Controller
from ai_teleop.domain import NoAssist
from ai_teleop.domain.interfaces import AssistProvider
from ai_teleop.eval.observer import TrialObserver
from ai_teleop.eval.schema import TrialKPIs
from ai_teleop.eval.trace import TRACE_NPZ_NAME, EvalTraceRecorder
from ai_teleop.input.scripted_noisy_human import ScriptedNoisyHuman
from ai_teleop.sim.runner import DEFAULT_MAX_STEPS, run_episode
from ai_teleop.sim.scene import SimEnv
from ai_teleop.sim.scene_source import STATIC_TASK_SCENE

# Controller command clamp (m/step). Matches the run_episode / data-gen default;
# it is a difficulty knob the calibration sweep may vary.
DEFAULT_MAX_DPOS = 0.025


@dataclass(frozen=True)
class Config:
    """One assist configuration to evaluate.

    ``assist_factory`` is called once per trial to build a fresh provider (so any
    per-episode state — e.g. the residual's GRU hidden state — starts clean). It is
    a factory, not an instance, so ``eval/`` need not import any concrete policy.
    """

    label: str
    assist_factory: Callable[[], AssistProvider]


# The always-available human-only baseline config (needs no checkpoint).
HUMAN_ONLY = Config(label="human_only", assist_factory=NoAssist)


def _human_seed(master_seed: int, episode_index: int) -> int:
    """Deterministic per-trial operator seed from ``(master_seed, episode_index)``."""
    return int(np.random.SeedSequence([master_seed, episode_index]).generate_state(1)[0])


def run_trial(
    episode_index: int,
    config: Config,
    *,
    master_seed: int = 0,
    scene_path: str | Path = STATIC_TASK_SCENE,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_dpos: float = DEFAULT_MAX_DPOS,
    trace_path: str | Path | None = None,
    **observer_kwargs: Any,
) -> TrialKPIs:
    """Run one trial of ``config`` and return its KPI record.

    When ``trace_path`` is given, the realized-state trace is written there so the
    KPIs can later be recomputed offline via :func:`replay_kpis`.
    """
    environment = SimEnv(str(scene_path), render_mode="headless", seed=master_seed, randomize=True)
    try:
        controller = Controller(environment, max_dpos_per_step=max_dpos)
        observation = environment.reset(episode_index)
        target_position = observation.hole_poses[observation.target_hole_index][:3].copy()
        home_quaternion = controller.home_pose[3:]
        target_pose = np.concatenate([target_position, home_quaternion])
        human = ScriptedNoisyHuman(target_pose, seed=_human_seed(master_seed, episode_index))
        assist = config.assist_factory()

        observer = TrialObserver(seed=episode_index, config_label=config.label, **observer_kwargs)
        recorder = EvalTraceRecorder() if trace_path is not None else None

        def step_callback(step, obs, base_command, delta, command) -> bool:
            if recorder is not None:
                recorder.record(obs, base_command, delta)
            return observer(step, obs, base_command, delta, command)

        run_episode(
            environment,
            controller,
            human,
            assist,
            max_steps=max_steps,
            reset_episode_index=episode_index,
            step_callback=step_callback,
        )

        if recorder is not None and trace_path is not None:
            recorder.save(
                trace_path,
                {
                    "master_seed": master_seed,
                    "episode_index": episode_index,
                    "config": config.label,
                },
            )
        return observer.result()
    finally:
        environment.close()


def run_paired(
    episode_index: int,
    configs: list[Config],
    *,
    master_seed: int = 0,
    out_dir: str | Path | None = None,
    scene_path: str | Path = STATIC_TASK_SCENE,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_dpos: float = DEFAULT_MAX_DPOS,
    **observer_kwargs: Any,
) -> dict[str, TrialKPIs]:
    """Run one paired trial — the same ``episode_index`` under each config.

    Returns ``{config.label: TrialKPIs}``. When ``out_dir`` is set, each config's
    trace is written to ``out_dir/<label>/episode_<NNNNN>/trace.npz``.
    """
    results: dict[str, TrialKPIs] = {}
    for config in configs:
        trace_path: Path | None = None
        if out_dir is not None:
            trace_path = (
                Path(out_dir) / config.label / f"episode_{episode_index:05d}" / TRACE_NPZ_NAME
            )
        results[config.label] = run_trial(
            episode_index,
            config,
            master_seed=master_seed,
            scene_path=scene_path,
            max_steps=max_steps,
            max_dpos=max_dpos,
            trace_path=trace_path,
            **observer_kwargs,
        )
    return results


def replay_kpis(trace_path: str | Path, **observer_kwargs: Any) -> TrialKPIs:
    """Recompute a trial's KPIs offline from a saved trace — no episode re-run.

    Drives a fresh :class:`TrialObserver` over the reconstructed observation stream,
    so the result equals what the live observer produced (the observer reads only the
    observation, so live and replay are the same calculator).
    """
    from ai_teleop.eval.trace import load_eval_trace, replay_trace

    columns, metadata = load_eval_trace(trace_path)
    observer = TrialObserver(
        seed=metadata.get("episode_index"),
        config_label=metadata.get("config"),
        **observer_kwargs,
    )
    for step, observation, base_command, delta in replay_trace(columns):
        command = None  # the observer does not read `command`; nothing to reconstruct
        if observer(step, observation, base_command, delta, command):
            break
    return observer.result()
