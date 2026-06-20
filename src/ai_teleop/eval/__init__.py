"""Evaluation harness — the M6 measurement layer.

A **passive observer** that watches the runtime ``Observation`` stream, decides
when trials start/end, classifies success/failure, and computes the KPIs defined
in ``project-scope.md`` / ``docs/design/evaluation-protocol.md``. It has no
dependency on the controller in either direction (Dependency Inversion): trial,
success, and KPI concepts live only here, so the controller stays mode-less.

Public surface:

* :class:`TrialObserver` — the ``run_episode`` ``step_callback`` that produces a
  per-trial record (LAB-36).
* :class:`TrialKPIs` / :class:`TrialOutcome` — the behavior-free record contract
  the ablation runner (LAB-37) and reporting (LAB-38) consume.
* :class:`Config` / :func:`run_trial` / :func:`run_paired` / :func:`replay_kpis` —
  the paired-seed ablation mechanism + offline trace replay (LAB-37).
* :class:`EvalTraceRecorder` — the realized-state eval-log producer (LAB-37).
"""

from ai_teleop.eval.ablation import (
    HUMAN_ONLY,
    Config,
    replay_kpis,
    run_paired,
    run_trial,
)
from ai_teleop.eval.observer import TrialObserver
from ai_teleop.eval.schema import TrialKPIs, TrialOutcome
from ai_teleop.eval.trace import EvalTraceRecorder, load_eval_trace

__all__ = [
    "TrialObserver",
    "TrialKPIs",
    "TrialOutcome",
    "Config",
    "HUMAN_ONLY",
    "run_trial",
    "run_paired",
    "replay_kpis",
    "EvalTraceRecorder",
    "load_eval_trace",
]
