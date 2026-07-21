"""Paired-seed ablation runner — the mechanism behind the M6 head-to-head (LAB-37).

One *trial* is a fixed ``(master_seed, episode_index)`` pair: it pins the
procedural wall (built from a seed derived from exactly that pair, mirroring data
generation) and the scripted operator (a same-seeded :class:`ScriptedNoisyHuman`,
which is **open-loop**
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
from ai_teleop.data.generate import (
    DEFAULT_JOINT_DAMPING,
    DEFAULT_SPEED_LOGNORMAL_MEDIAN,
    DEFAULT_SPEED_LOGNORMAL_SIGMA,
)
from ai_teleop.data.generate import DEFAULT_MAX_DPOS as _DATAGEN_MAX_DPOS
from ai_teleop.domain import NoAssist
from ai_teleop.domain.interfaces import AssistProvider
from ai_teleop.eval.observer import DEFAULT_FORCE_CAP, TrialObserver
from ai_teleop.eval.schema import TrialKPIs
from ai_teleop.eval.trace import TRACE_NPZ_NAME, EvalTraceRecorder
from ai_teleop.input.scripted_noisy_human import (
    DEFAULT_DRIFT_POSITION_STD,
    DEFAULT_POSITION_BIAS_STD,
    ScriptedNoisyHuman,
)
from ai_teleop.sim.config import EnvConfig, episode_wall_seed
from ai_teleop.sim.env_setup import make_env
from ai_teleop.sim.runner import run_episode
from ai_teleop.sim.scene import DEFAULT_WRIST_RENDER_EVERY

# Both generated walls and the static escape hatch place the goal at hole_0; the
# expert/observer/operator are all aimed there (the env stays target-agnostic).
_TARGET_HOLE_INDEX = 0

# Controller command clamp (m/step). Re-anchored to the data-gen / deployment
# (teleop) config by LAB-98: eval trials must sample the same contact dynamics
# and operator distribution the corpus trains on, or the difficulty pin measures
# a different task than the policy learned (LAB-96 moved the corpus; the pin
# follows). The Controller's own careful-insertion default (0.025) is still
# reachable per-trial via the ``max_dpos`` argument.
DEFAULT_MAX_DPOS = _DATAGEN_MAX_DPOS

# Per-episode step budget for an insertion trial (~18 s of sim @ 500 Hz). Moves in
# lockstep with data.generate.DEFAULT_MAX_STEPS (6000 → 9000 by LAB-100: the operator
# speed draw's slow tail needs the extra clock to finish seating); eval must use the
# same task budget as data-gen or timeout rates measure the budget, not the policy.
# Pre-LAB-100 corpora (dataset_8 and earlier) were generated at 6000.
INSERTION_MAX_STEPS = 9000

# Difficulty knob for the LAB-53 pin: a multiplier on the scripted operator's lateral
# error (per-episode bias + OU drift) relative to the M5 training distribution.
# 1.0 == the σ's the corpus was generated at (where contact lands on the flat wall, far
# outside the ~chamfer-width capture band); < 1.0 shrinks the error toward that band so
# the F/T residual has a lever and human-only sits below ceiling with headroom.
DEFAULT_OPERATOR_ERROR_SCALE = 1.0

# Wrist-camera capture cadence for a vision trial — the shared env default
# (`sim.scene.DEFAULT_WRIST_RENDER_EVERY`). Only a vision policy triggers capture;
# F/T-only and human-only render nothing.


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
    generated_walls: bool = True,
    # Default is the insertion budget (9000), NOT the generic sim budget: eval must
    # use the same task budget as data-gen or timeouts measure the budget, not the
    # policy. LAB-107: this default silently disagreed with scripts/evaluate.py's
    # (which always passed INSERTION_MAX_STEPS), so callers that omit max_steps —
    # e.g. dagger._reablate — under-budgeted at 5000 and depressed human-only.
    max_steps: int = INSERTION_MAX_STEPS,
    max_dpos: float = DEFAULT_MAX_DPOS,
    joint_damping: float = DEFAULT_JOINT_DAMPING,
    operator_error_scale: float = DEFAULT_OPERATOR_ERROR_SCALE,
    speed_lognormal_median: float = DEFAULT_SPEED_LOGNORMAL_MEDIAN,
    speed_lognormal_sigma: float = DEFAULT_SPEED_LOGNORMAL_SIGMA,
    force_cap: float = DEFAULT_FORCE_CAP,
    wrist_render_every: int = DEFAULT_WRIST_RENDER_EVERY,
    trace_path: str | Path | None = None,
    **observer_kwargs: Any,
) -> TrialKPIs:
    """Run one trial of ``config`` and return its KPI record.

    Mirrors data generation: with ``generated_walls`` (the default) the trial runs
    on its own procedural wall seeded from ``(master_seed, episode_index)``, so eval
    matches the per-episode-wall training distribution; ``generated_walls=False`` runs
    on the static wall instead (no ``scenegen``/CadQuery — for fast tests). The
    controller config (``max_dpos``, ``joint_damping``) and the operator's
    per-episode approach-speed draw (``speed_lognormal_*``) default to the
    data-gen deployment config (LAB-96/98), so eval trials sample the same task
    distribution the corpus trains on.

    ``operator_error_scale`` multiplies the scripted operator's lateral-error σ's
    (bias + drift) off their training defaults — the difficulty knob the LAB-53 pin
    sweeps. ``force_cap`` feeds **both** the controller's watchdog (``force_cap_n``)
    and the observer's own FORCE_ABORT threshold — they must match (LAB-94: the
    controller freezes the arm at its threshold first, so a higher observer
    threshold is never reached and FORCE_ABORT silently never fires). When
    ``trace_path`` is given, the realized-state trace is written there so the KPIs
    can later be recomputed offline via :func:`replay_kpis`.
    """
    wall_seed = episode_wall_seed(master_seed, episode_index) if generated_walls else None
    environment = make_env(EnvConfig(wall_seed=wall_seed), render_mode="headless")
    try:
        controller = Controller(
            environment,
            max_dpos_per_step=max_dpos,
            joint_damping=joint_damping,
            force_cap_n=force_cap,
        )
        observation = environment.reset()
        target_position = observation.hole_poses[_TARGET_HOLE_INDEX][:3].copy()
        home_quaternion = controller.home_pose[3:]
        target_pose = np.concatenate([target_position, home_quaternion])
        human = ScriptedNoisyHuman(
            target_pose,
            position_bias_std=DEFAULT_POSITION_BIAS_STD * operator_error_scale,
            drift_position_std=DEFAULT_DRIFT_POSITION_STD * operator_error_scale,
            speed_lognormal_median=speed_lognormal_median,
            speed_lognormal_sigma=speed_lognormal_sigma,
            seed=_human_seed(master_seed, episode_index),
        )
        assist = config.assist_factory()
        # A vision policy needs a live wrist frame on each Observation; enable the
        # env's rate-limited capture for it (duck-typed so eval/ imports no policy).
        # F/T-only and human-only leave `use_vision` False → the env renders nothing.
        if getattr(assist, "use_vision", False):
            environment.enable_wrist_capture(wrist_render_every)

        observer = TrialObserver(
            target_hole_index=_TARGET_HOLE_INDEX,
            seed=episode_index,
            config_label=config.label,
            force_cap=force_cap,
            **observer_kwargs,
        )
        recorder = (
            EvalTraceRecorder(target_hole_index=_TARGET_HOLE_INDEX)
            if trace_path is not None
            else None
        )

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
    out_dir: str | Path | None = None,
    **trial_kwargs: Any,
) -> dict[str, TrialKPIs]:
    """Run one paired trial — the same ``episode_index`` under each config.

    Returns ``{config.label: TrialKPIs}``. When ``out_dir`` is set, each config's
    trace is written to ``out_dir/<label>/episode_<NNNNN>/trace.npz``. Everything
    else (``master_seed``, ``max_steps``, the controller/operator knobs, observer
    kwargs) is forwarded verbatim to `run_trial` — see its signature for the
    defaults.

    It forwards rather than re-declares deliberately: LAB-107 was caused by this
    function carrying its own copy of ``max_steps``'s default, which drifted from
    `run_trial`'s and silently under-budgeted the DAgger eval path by 4000 steps.
    One definition of each default is the fix (audit finding C-3).
    """
    results: dict[str, TrialKPIs] = {}
    for config in configs:
        trace_path: Path | None = None
        if out_dir is not None:
            trace_path = (
                Path(out_dir) / config.label / f"episode_{episode_index:05d}" / TRACE_NPZ_NAME
            )
        results[config.label] = run_trial(
            episode_index, config, trace_path=trace_path, **trial_kwargs
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
    # The observer does not read `command`; nothing to reconstruct from the trace.
    for step, observation, base_command, delta in replay_trace(columns):
        if observer(step, observation, base_command, delta, command=None):
            break
    return observer.result()
