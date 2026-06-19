"""Per-trial KPI record — the behavior-free on-disk contract for `eval/`.

This is the unit the passive-observer harness (LAB-36) emits and the ablation
runner (LAB-37) / reporting (LAB-38) consume. Kept implementation-free (no sim,
control, or numpy import) so any layer can depend on it without a cycle —
mirrors the rationale for ``data/schema.py``.

The five KPIs are the evaluation-protocol set:

================================  =====  =================================
field                             type   role
================================  =====  =================================
``outcome`` / ``success``         enum   **headline** — did the peg seat?
``time_to_insert_s``              s      supporting (None unless success)
``peak_contact_force``            N      safety proxy — bounded by design
``contact_events``                count  supporting
``jerk_integral``                 —      trajectory smoothness (∫|jerk|dt)
================================  =====  =================================

``seed`` and ``config_label`` are the **pairing keys**: the ablation runs each
seed once per configuration, and a (seed, config) pair identifies the matched
human-only vs. residual trials whose per-seed delta carries the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class TrialOutcome(StrEnum):
    """How a trial ended.

    The string values intentionally match ``data.trajectory.TerminalReason`` so
    eval records and data-gen metadata speak the same vocabulary — but the enum
    is re-declared here rather than imported, to keep ``eval/`` decoupled from
    the data-generation layer (the harness depends only on ``common/``).
    """

    SUCCESS = "success"  # peg seated past the depth threshold, sustained
    FORCE_ABORT = "force_abort"  # contact force exceeded the cap
    TIMEOUT = "timeout"  # step budget reached without seating


@dataclass(frozen=True)
class TrialKPIs:
    """One trial's outcome + KPI record (the harness's per-trial output).

    Frozen value object; serialize with :meth:`to_dict` and rebuild with
    :meth:`from_dict` for the on-disk results table the reporting step reads.
    """

    outcome: TrialOutcome
    time_to_insert_s: float | None
    peak_contact_force: float
    contact_events: int
    jerk_integral: float
    n_steps: int
    duration_s: float
    # Pairing keys — set by the ablation runner, absent on a standalone observe.
    seed: int | None = None
    config_label: str | None = None

    @property
    def success(self) -> bool:
        """The headline metric — derived from ``outcome`` (single source of truth)."""
        return self.outcome is TrialOutcome.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping (``outcome`` flattened to its string value)."""
        record = asdict(self)
        record["outcome"] = self.outcome.value
        record["success"] = self.success
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> TrialKPIs:
        """Rebuild from a :meth:`to_dict` mapping (ignores the derived ``success``)."""
        return cls(
            outcome=TrialOutcome(record["outcome"]),
            time_to_insert_s=record["time_to_insert_s"],
            peak_contact_force=record["peak_contact_force"],
            contact_events=record["contact_events"],
            jerk_integral=record["jerk_integral"],
            n_steps=record["n_steps"],
            duration_s=record["duration_s"],
            seed=record.get("seed"),
            config_label=record.get("config_label"),
        )
