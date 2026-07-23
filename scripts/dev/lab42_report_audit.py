"""LAB-42 1A round-2: audit `eval/report.py` against the committed trials.csv files.

Checks the two things that would silently corrupt the Phase-3 KPI dashboard:

1. **Duplicate seeds** — `pair_by_seed` builds a dict keyed by seed, so a repeated
   seed in the treatment side is silently overwritten (last wins) and a repeated
   seed on the baseline side silently duplicates the pair.
2. **Two denominators for "success rate"** — the marginal table reports
   `n_success / n_trials` over *all* of a config's trials; the paired table reports
   `(both + baseline_only) / n_pairs` over *matched* seeds only. Same label, same
   report, different number whenever the seed sets differ.

Read-only. Run: `uv run python scripts/dev/lab42_report_audit.py`
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ai_teleop.eval.report import compare_paired, group_by_config, load_trials

TRIALS = sorted(Path("runs").rglob("trials.csv")) + sorted(
    Path("docs/results").rglob("*trials.csv")
)


def main() -> None:
    for csv_path in TRIALS:
        trials = load_trials(csv_path)
        grouped = group_by_config(trials)
        print(f"\n=== {csv_path} — {len(trials)} rows, configs: {list(grouped)}")

        for label, rows in grouped.items():
            seeds = [t.seed for t in rows]
            dupes = {s: n for s, n in Counter(seeds).items() if n > 1}
            marginal = sum(1 for t in rows if t.success) / len(rows)
            print(
                f"  {label:<12} n={len(rows):<4} seeds={len(set(seeds)):<4} "
                f"marginal_success={100 * marginal:5.1f}%"
                + (f"  DUPLICATE SEEDS: {dupes}" if dupes else "")
            )

        labels = list(grouped)
        for base in labels:
            for treat in labels:
                if base == treat:
                    continue
                comparison = compare_paired(
                    grouped[base], grouped[treat], baseline_label=base, treatment_label=treat
                )
                success = comparison.success
                marginal_base = sum(1 for t in grouped[base] if t.success) / len(grouped[base])
                marginal_treat = sum(1 for t in grouped[treat] if t.success) / len(grouped[treat])
                mismatch = (
                    abs(marginal_base - success.baseline_rate) > 1e-9
                    or abs(marginal_treat - success.treatment_rate) > 1e-9
                )
                print(
                    f"  paired {base} vs {treat}: n_pairs={success.n_pairs} "
                    f"baseline {100 * success.baseline_rate:.1f}% (marginal {100 * marginal_base:.1f}%) "
                    f"treatment {100 * success.treatment_rate:.1f}% (marginal {100 * marginal_treat:.1f}%)"
                    + ("   <-- DENOMINATOR MISMATCH" if mismatch else "")
                )
                # The per-KPI pair count is computed and never rendered in the table.
                for stat in comparison.kpi_stats:
                    if stat.p_value is not None and stat.n_pairs < 6:
                        print(
                            f"      UNDERPOWERED: {stat.label} p={stat.p_value:.3f} "
                            f"over n_pairs={stat.n_pairs} (Wilcoxon cannot reach p<0.05 below n=6)"
                        )


if __name__ == "__main__":
    main()
