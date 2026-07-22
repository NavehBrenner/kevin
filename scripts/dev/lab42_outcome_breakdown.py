"""LAB-42 H-7: per-config outcome mix across every committed trials.csv.

The 100-seed LAB-53 run (`runs/eval/trials.csv`, 2026-06-28) has **zero** force-aborts;
every later eval set is ~70% force-abort. This prints the per-config split so the
operating-point ledger can say whether the abort mechanism lands symmetrically on the
two arms — if it fires more on the residual, a pre-abort measurement flatters it.

Read-only. Run: `uv run python scripts/dev/lab42_outcome_breakdown.py`
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ai_teleop.eval.report import group_by_config, load_trials

TRIALS = sorted(Path("runs").rglob("trials.csv")) + sorted(
    Path("docs/results").rglob("*trials.csv")
)


def main() -> None:
    for csv_path in TRIALS:
        trials = load_trials(csv_path)
        print(f"\n=== {csv_path}")
        for label, rows in group_by_config(trials).items():
            counts = Counter(t.outcome.value for t in rows)
            steps = sorted({t.n_steps for t in rows})
            peak = sorted(t.peak_contact_force for t in rows)
            print(
                f"  {label:<12} "
                + " ".join(
                    f"{name}={counts.get(name, 0):<3}"
                    for name in ("success", "force_abort", "timeout")
                )
                + f" | max n_steps={max(steps)} | median peak F={peak[len(peak) // 2]:.1f} N"
            )


if __name__ == "__main__":
    main()
