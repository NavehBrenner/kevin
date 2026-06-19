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
"""

from ai_teleop.eval.observer import TrialObserver
from ai_teleop.eval.schema import TrialKPIs, TrialOutcome

__all__ = [
    "TrialObserver",
    "TrialKPIs",
    "TrialOutcome",
]
