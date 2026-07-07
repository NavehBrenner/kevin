"""Phase-1 KPI reporting tests (LAB-38).

Pure aggregation over synthetic :class:`TrialKPIs` records — no sim, no controller,
no trained checkpoint — so these are fast and deterministic. They pin the contract
the reporting layer owes the D1 artifact: the CSV round-trips, the marginal and
paired aggregates are correct (McNemar success split + Wilcoxon KPI deltas), and the
one-command regeneration writes the tables + plots (and degrades to marginal-only
when a config is missing).
"""

from __future__ import annotations

import csv
from pathlib import Path

from ai_teleop.eval.report import (
    build_report,
    compare_paired,
    format_marginal_table,
    format_paired_table,
    group_by_config,
    load_trials,
    pair_by_seed,
    summarize_config,
)
from ai_teleop.eval.schema import TrialKPIs, TrialOutcome


def make_kpi(
    *,
    outcome: TrialOutcome = TrialOutcome.SUCCESS,
    seed: int = 0,
    config_label: str = "human_only",
    time_to_insert_s: float | None = 1.0,
    peak_contact_force: float = 10.0,
    contact_events: int = 2,
    jerk_integral: float = 5.0,
    n_steps: int = 100,
    duration_s: float = 2.0,
) -> TrialKPIs:
    """A synthetic trial record; time-to-insert is cleared on a non-success outcome."""
    return TrialKPIs(
        outcome=outcome,
        time_to_insert_s=time_to_insert_s if outcome is TrialOutcome.SUCCESS else None,
        peak_contact_force=peak_contact_force,
        contact_events=contact_events,
        jerk_integral=jerk_integral,
        n_steps=n_steps,
        duration_s=duration_s,
        seed=seed,
        config_label=config_label,
    )


def _write_csv(path: Path, trials: list[TrialKPIs]) -> None:
    """Persist records exactly as `evaluate.py pair` does (DictWriter over to_dict)."""
    rows = [t.to_dict() for t in trials]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_trials_roundtrips_records(tmp_path: Path) -> None:
    trials = [
        make_kpi(seed=0, outcome=TrialOutcome.SUCCESS, time_to_insert_s=1.5),
        make_kpi(seed=1, outcome=TrialOutcome.TIMEOUT),  # time_to_insert None
        make_kpi(seed=2, outcome=TrialOutcome.FORCE_ABORT),
    ]
    csv_path = tmp_path / "trials.csv"
    _write_csv(csv_path, trials)

    loaded = load_trials(csv_path)

    assert loaded == trials
    # The miss/abort rows restore their undefined time-to-insert to None, not "".
    assert loaded[1].time_to_insert_s is None
    assert loaded[2].time_to_insert_s is None


def test_summarize_config_success_rate_and_means() -> None:
    trials = [
        make_kpi(
            seed=0, outcome=TrialOutcome.SUCCESS, time_to_insert_s=2.0, peak_contact_force=8.0
        ),
        make_kpi(
            seed=1, outcome=TrialOutcome.SUCCESS, time_to_insert_s=4.0, peak_contact_force=12.0
        ),
        make_kpi(seed=2, outcome=TrialOutcome.TIMEOUT, peak_contact_force=20.0),
    ]
    summary = summarize_config("human_only", trials)

    assert summary.n_trials == 3
    assert summary.n_success == 2
    assert summary.success_rate == 2 / 3
    # time-to-insert averages only the two successes; peak force averages all three.
    time_stat = next(s for s in summary.kpi_stats if s.label == "Time to insert")
    force_stat = next(s for s in summary.kpi_stats if s.label == "Peak contact force")
    assert time_stat.n == 2
    assert time_stat.mean == 3.0
    assert force_stat.n == 3
    assert force_stat.mean == (8.0 + 12.0 + 20.0) / 3


def test_pair_by_seed_matches_and_drops_unmatched() -> None:
    baseline = [make_kpi(seed=s, config_label="human_only") for s in (0, 1, 2)]
    treatment = [make_kpi(seed=s, config_label="residual") for s in (1, 2, 3)]

    pairs = pair_by_seed(baseline, treatment)

    assert [base.seed for base, _ in pairs] == [1, 2]  # seed 0 and 3 dropped, sorted
    assert all(base.config_label == "human_only" for base, _ in pairs)
    assert all(treat.config_label == "residual" for _, treat in pairs)


def test_paired_success_mcnemar_split_and_delta() -> None:
    # Seeds 0-2: baseline misses, residual seats → treatment_only (wins).
    # Seed 3: both seat. Seed 4: baseline seats, residual misses → baseline_only.
    baseline = (
        [
            make_kpi(seed=s, config_label="human_only", outcome=TrialOutcome.TIMEOUT)
            for s in (0, 1, 2)
        ]
        + [make_kpi(seed=3, config_label="human_only", outcome=TrialOutcome.SUCCESS)]
        + [make_kpi(seed=4, config_label="human_only", outcome=TrialOutcome.SUCCESS)]
    )
    treatment = [
        make_kpi(seed=s, config_label="residual", outcome=TrialOutcome.SUCCESS)
        for s in (0, 1, 2, 3)
    ] + [make_kpi(seed=4, config_label="residual", outcome=TrialOutcome.TIMEOUT)]

    comparison = compare_paired(
        baseline, treatment, baseline_label="human_only", treatment_label="residual"
    )
    success = comparison.success

    assert success.n_pairs == 5
    assert success.treatment_only == 3
    assert success.baseline_only == 1
    assert success.both == 1
    assert success.neither == 0
    assert success.baseline_rate == 2 / 5
    assert success.treatment_rate == 4 / 5
    assert success.rate_delta == 2 / 5
    assert success.p_value is not None  # discordant pairs present → a test ran


def test_paired_kpi_delta_direction_and_significance() -> None:
    # Residual consistently lands lower peak force on every matched seed.
    baseline = [
        make_kpi(
            seed=s, config_label="human_only", outcome=TrialOutcome.SUCCESS, peak_contact_force=20.0
        )
        for s in range(8)
    ]
    treatment = [
        make_kpi(
            seed=s, config_label="residual", outcome=TrialOutcome.SUCCESS, peak_contact_force=12.0
        )
        for s in range(8)
    ]

    comparison = compare_paired(
        baseline, treatment, baseline_label="human_only", treatment_label="residual"
    )
    force = next(s for s in comparison.kpi_stats if s.label == "Peak contact force")

    assert force.n_pairs == 8
    assert force.mean_delta == -8.0  # treatment − baseline
    assert force.treatment_better is True  # lower force is better
    assert force.p_value is not None and force.p_value < 0.05


def test_build_report_writes_tables_and_plots_when_paired(tmp_path: Path) -> None:
    trials = [
        make_kpi(seed=s, config_label="human_only", outcome=TrialOutcome.TIMEOUT) for s in range(6)
    ] + [make_kpi(seed=s, config_label="residual", outcome=TrialOutcome.SUCCESS) for s in range(6)]

    artifacts = build_report(trials, tmp_path)

    assert (tmp_path / "kpi_tables.md").exists()
    assert artifacts.success_plot.exists()
    assert artifacts.distributions_plot.exists()
    assert artifacts.deltas_plot is not None and artifacts.deltas_plot.exists()
    assert artifacts.paired_table is not None
    # The paired table names the headline improvement.
    assert "Success rate" in artifacts.paired_table
    assert "+100.0pp" in artifacts.paired_table


def test_build_report_degrades_to_marginal_only_for_single_config(tmp_path: Path) -> None:
    trials = [make_kpi(seed=s, config_label="human_only") for s in range(4)]

    artifacts = build_report(trials, tmp_path)

    assert artifacts.paired_table is None  # no residual config → no paired result
    assert artifacts.deltas_plot is None
    assert artifacts.success_plot.exists()
    assert artifacts.distributions_plot.exists()


def test_marginal_table_lists_every_config_and_kpi() -> None:
    trials = [
        make_kpi(seed=0, config_label="human_only"),
        make_kpi(seed=0, config_label="residual"),
    ]
    grouped = group_by_config(trials)
    summaries = [summarize_config(label, group) for label, group in grouped.items()]

    table = format_marginal_table(summaries)

    assert "human_only" in table and "residual" in table
    assert "Success rate" in table
    for label in ("Time to insert", "Peak contact force", "Trajectory jerk"):
        assert label in table


def test_paired_table_reports_discordant_footer() -> None:
    baseline = [
        make_kpi(seed=s, config_label="human_only", outcome=TrialOutcome.TIMEOUT) for s in range(3)
    ]
    treatment = [
        make_kpi(seed=s, config_label="residual", outcome=TrialOutcome.SUCCESS) for s in range(3)
    ]
    comparison = compare_paired(
        baseline, treatment, baseline_label="human_only", treatment_label="residual"
    )

    table = format_paired_table(comparison)

    assert "matched seeds" in table
    assert "won by residual" in table
