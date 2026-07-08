"""Phase-1 KPI aggregation, tables, and plots (LAB-38).

The reporting end of the M6 harness: it consumes the flat per-trial records the
ablation runner writes (``trials.csv``, one row per ``(seed, config)`` via
:meth:`TrialKPIs.to_dict`) and turns them into the publishable Phase-1 comparison —
the KPI tables, the paired-design summary statistics, and the plots that become the
D1 result.

Two kinds of aggregate, matching the evaluation protocol:

* **Marginal** (:class:`ConfigSummary`) — one config's success rate and per-KPI
  central tendency over its trials. Always available (even for a single config, e.g.
  a human-only calibration run).
* **Paired** (:class:`PairedComparison`) — the primary result. Trials are matched by
  seed across two configs (same seed ⇒ identical scripted-operator stream, only the
  assist differs), so the per-seed *delta* carries the signal with zero operator
  variance. Success uses the McNemar discordant-pair split; the continuous KPIs use
  the Wilcoxon signed-rank test over matched pairs.

Pure aggregation + rendering — no sim, no controller, no re-run. Every number is a
function of the stored records, so the tables/plots regenerate from ``trials.csv``
with one command (``scripts/report_results.py``), which is the LAB-38 acceptance.

The peak-force KPI is reported as any other column, but note it is **bounded by
construction** (the residual is hard-clamped and the impedance backbone caps contact
force mechanically) — the table states a guarantee, not a hopeful statistic.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from ai_teleop.eval.schema import TrialKPIs  # noqa: E402


@dataclass(frozen=True)
class KpiSpec:
    """One continuous KPI's reporting metadata — drives tables and plots uniformly.

    ``attribute`` is the :class:`TrialKPIs` field; ``success_only`` restricts the
    aggregate to successful trials (time-to-insert is undefined on a miss).
    ``lower_is_better`` orients the "which config wins" reading in the tables.
    """

    attribute: str
    label: str
    unit: str
    success_only: bool
    lower_is_better: bool


# The four continuous KPIs (success rate is the headline, handled separately).
CONTINUOUS_KPIS: tuple[KpiSpec, ...] = (
    KpiSpec("time_to_insert_s", "Time to insert", "s", success_only=True, lower_is_better=True),
    KpiSpec(
        "peak_contact_force", "Peak contact force", "N", success_only=False, lower_is_better=True
    ),
    KpiSpec("contact_events", "Contact events", "", success_only=False, lower_is_better=True),
    KpiSpec(
        "jerk_integral", "Trajectory jerk (∫|jerk|)", "", success_only=False, lower_is_better=True
    ),
)


def load_trials(csv_path: str | Path) -> list[TrialKPIs]:
    """Load the ablation runner's ``trials.csv`` back into :class:`TrialKPIs` records.

    Inverts :meth:`TrialKPIs.to_dict` — the CSV stores every field as text, so the
    numeric columns are coerced back and an empty ``time_to_insert_s`` (a miss) is
    restored to ``None``.
    """
    records: list[TrialKPIs] = []
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            time_to_insert = row["time_to_insert_s"]
            records.append(
                TrialKPIs.from_dict({
                    "outcome": row["outcome"],
                    "time_to_insert_s": (
                        float(time_to_insert) if time_to_insert not in ("", "None") else None
                    ),
                    "peak_contact_force": float(row["peak_contact_force"]),
                    "contact_events": int(row["contact_events"]),
                    "jerk_integral": float(row["jerk_integral"]),
                    "n_steps": int(row["n_steps"]),
                    "duration_s": float(row["duration_s"]),
                    "seed": int(row["seed"]) if row.get("seed") not in (None, "", "None") else None,
                    "config_label": row.get("config_label") or None,
                })
            )
    return records


def group_by_config(trials: Sequence[TrialKPIs]) -> dict[str, list[TrialKPIs]]:
    """Partition trials by ``config_label``, preserving first-seen config order."""
    grouped: dict[str, list[TrialKPIs]] = {}
    for trial in trials:
        label = trial.config_label or "unlabeled"
        grouped.setdefault(label, []).append(trial)
    return grouped


def _kpi_values(trials: Sequence[TrialKPIs], spec: KpiSpec) -> np.ndarray:
    """The KPI's values over the relevant trials (successes only when specified)."""
    subset = [t for t in trials if t.success] if spec.success_only else list(trials)
    values = [getattr(t, spec.attribute) for t in subset]
    return np.asarray([v for v in values if v is not None], dtype=float)


@dataclass(frozen=True)
class KpiStat:
    """Central tendency of one KPI over one config's trials."""

    label: str
    unit: str
    n: int  # trials contributing (successes only for success-only KPIs)
    mean: float | None
    median: float | None
    std: float | None


@dataclass(frozen=True)
class ConfigSummary:
    """Marginal summary of one configuration."""

    config_label: str
    n_trials: int
    n_success: int
    kpi_stats: tuple[KpiStat, ...]

    @property
    def success_rate(self) -> float:
        """Fraction of trials that seated the peg (the headline metric)."""
        return self.n_success / self.n_trials if self.n_trials else 0.0


def summarize_config(config_label: str, trials: Sequence[TrialKPIs]) -> ConfigSummary:
    """Aggregate one config's trials into its success rate + per-KPI central tendency."""
    stats_out: list[KpiStat] = []
    for spec in CONTINUOUS_KPIS:
        values = _kpi_values(trials, spec)
        has = values.size > 0
        stats_out.append(
            KpiStat(
                label=spec.label,
                unit=spec.unit,
                n=int(values.size),
                mean=float(values.mean()) if has else None,
                median=float(np.median(values)) if has else None,
                std=float(values.std(ddof=1)) if values.size > 1 else (0.0 if has else None),
            )
        )
    return ConfigSummary(
        config_label=config_label,
        n_trials=len(trials),
        n_success=sum(1 for t in trials if t.success),
        kpi_stats=tuple(stats_out),
    )


@dataclass(frozen=True)
class SuccessMcNemar:
    """The paired success contingency (McNemar) between baseline and treatment.

    ``both`` / ``neither`` are the concordant pairs; ``treatment_only`` (b) and
    ``baseline_only`` (c) are the discordant pairs that carry the paired signal. The
    exact binomial p-value is over the discordant split (``b`` vs ``b + c``).
    """

    n_pairs: int
    both: int
    neither: int
    treatment_only: int  # treatment seats, baseline misses — the wins
    baseline_only: int  # baseline seats, treatment misses — the regressions
    baseline_rate: float
    treatment_rate: float
    p_value: float | None

    @property
    def rate_delta(self) -> float:
        """Treatment − baseline success rate (the headline improvement)."""
        return self.treatment_rate - self.baseline_rate


@dataclass(frozen=True)
class KpiPairedStat:
    """Paired per-seed delta for one continuous KPI (Wilcoxon signed-rank)."""

    label: str
    unit: str
    n_pairs: int  # matched seeds contributing (both-success for success-only KPIs)
    baseline_mean: float | None
    treatment_mean: float | None
    mean_delta: float | None  # treatment − baseline, averaged over pairs
    median_delta: float | None
    p_value: float | None
    lower_is_better: bool

    @property
    def treatment_better(self) -> bool | None:
        """Whether the treatment moved the KPI in the good direction (by mean delta)."""
        if self.mean_delta is None or self.mean_delta == 0.0:
            return None
        improved = self.mean_delta < 0.0
        return improved if self.lower_is_better else not improved


@dataclass(frozen=True)
class PairedComparison:
    """The primary paired result — baseline vs treatment matched by seed."""

    baseline_label: str
    treatment_label: str
    success: SuccessMcNemar
    kpi_stats: tuple[KpiPairedStat, ...]


def pair_by_seed(
    baseline: Sequence[TrialKPIs], treatment: Sequence[TrialKPIs]
) -> list[tuple[TrialKPIs, TrialKPIs]]:
    """Match trials sharing a seed across the two configs, in ascending seed order.

    Seeds present in only one config are dropped (an unmatched trial cannot
    contribute a paired delta). Requires seeds to be set on both sides.
    """
    by_seed_treatment = {t.seed: t for t in treatment if t.seed is not None}
    pairs: list[tuple[TrialKPIs, TrialKPIs]] = []
    for base in baseline:
        if base.seed is None:
            continue
        match = by_seed_treatment.get(base.seed)
        if match is not None:
            pairs.append((base, match))
    pairs.sort(key=lambda pair: pair[0].seed or 0)
    return pairs


def _mcnemar(pairs: Sequence[tuple[TrialKPIs, TrialKPIs]]) -> SuccessMcNemar:
    """The paired success contingency + exact-binomial p over the discordant split."""
    both = neither = treatment_only = baseline_only = 0
    for base, treat in pairs:
        if treat.success and base.success:
            both += 1
        elif treat.success and not base.success:
            treatment_only += 1
        elif base.success and not treat.success:
            baseline_only += 1
        else:
            neither += 1
    n = len(pairs)
    discordant = treatment_only + baseline_only
    # Exact McNemar: the discordant pairs split Binomial(n=discordant, p=0.5).
    p_value = float(stats.binomtest(treatment_only, discordant, 0.5).pvalue) if discordant else None
    return SuccessMcNemar(
        n_pairs=n,
        both=both,
        neither=neither,
        treatment_only=treatment_only,
        baseline_only=baseline_only,
        baseline_rate=(both + baseline_only) / n if n else 0.0,
        treatment_rate=(both + treatment_only) / n if n else 0.0,
        p_value=p_value,
    )


def _paired_kpi(pairs: Sequence[tuple[TrialKPIs, TrialKPIs]], spec: KpiSpec) -> KpiPairedStat:
    """Paired delta stats for one KPI over the matched pairs.

    For a success-only KPI (time-to-insert) a pair contributes only when *both* trials
    succeeded — a delta is otherwise undefined. The Wilcoxon signed-rank test needs at
    least one non-zero difference and is skipped (``p_value=None``) below that.
    """
    baseline_values: list[float] = []
    treatment_values: list[float] = []
    for base, treat in pairs:
        if spec.success_only and not (base.success and treat.success):
            continue
        base_v = getattr(base, spec.attribute)
        treat_v = getattr(treat, spec.attribute)
        if base_v is None or treat_v is None:
            continue
        baseline_values.append(float(base_v))
        treatment_values.append(float(treat_v))

    if not baseline_values:
        return KpiPairedStat(
            label=spec.label,
            unit=spec.unit,
            n_pairs=0,
            baseline_mean=None,
            treatment_mean=None,
            mean_delta=None,
            median_delta=None,
            p_value=None,
            lower_is_better=spec.lower_is_better,
        )

    base_array = np.asarray(baseline_values)
    treat_array = np.asarray(treatment_values)
    deltas = treat_array - base_array
    p_value: float | None = None
    if deltas.size > 0 and np.any(deltas != 0.0):
        p_value = float(stats.wilcoxon(treat_array, base_array).pvalue)
    return KpiPairedStat(
        label=spec.label,
        unit=spec.unit,
        n_pairs=int(base_array.size),
        baseline_mean=float(base_array.mean()),
        treatment_mean=float(treat_array.mean()),
        mean_delta=float(deltas.mean()),
        median_delta=float(np.median(deltas)),
        p_value=p_value,
        lower_is_better=spec.lower_is_better,
    )


def compare_paired(
    baseline: Sequence[TrialKPIs],
    treatment: Sequence[TrialKPIs],
    *,
    baseline_label: str,
    treatment_label: str,
) -> PairedComparison:
    """Build the primary paired comparison (matched by seed) between two configs."""
    pairs = pair_by_seed(baseline, treatment)
    return PairedComparison(
        baseline_label=baseline_label,
        treatment_label=treatment_label,
        success=_mcnemar(pairs),
        kpi_stats=tuple(_paired_kpi(pairs, spec) for spec in CONTINUOUS_KPIS),
    )


# ---------------------------------------------------------------------------
# Markdown table rendering
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 2) -> str:
    """Fixed-point cell, or an em dash for a missing value."""
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_p(value: float | None) -> str:
    """A p-value cell — tiny values collapse to ``<0.001``."""
    if value is None:
        return "—"
    return "<0.001" if value < 1e-3 else f"{value:.3f}"


def format_marginal_table(summaries: Sequence[ConfigSummary]) -> str:
    """Per-config marginal KPI table (means) as GitHub-flavored markdown."""
    header = ["KPI"] + [s.config_label for s in summaries]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    success_cells = [f"{100 * s.success_rate:.1f}% ({s.n_success}/{s.n_trials})" for s in summaries]
    lines.append("| " + " | ".join(["**Success rate**", *success_cells]) + " |")

    for index, spec in enumerate(CONTINUOUS_KPIS):
        label = f"{spec.label} ({spec.unit})" if spec.unit else spec.label
        cells = [_fmt(s.kpi_stats[index].mean) for s in summaries]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def format_paired_table(comparison: PairedComparison) -> str:
    """The paired comparison (per-seed deltas + significance) as markdown."""
    baseline = comparison.baseline_label
    treatment = comparison.treatment_label
    success = comparison.success
    header = ["KPI", baseline, treatment, "Δ (paired)", "p"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    lines.append(
        "| "
        + " | ".join([
            "**Success rate**",
            f"{100 * success.baseline_rate:.1f}%",
            f"{100 * success.treatment_rate:.1f}%",
            f"{100 * success.rate_delta:+.1f}pp",
            _fmt_p(success.p_value),
        ])
        + " |"
    )

    for stat in comparison.kpi_stats:
        delta = (
            "—"
            if stat.mean_delta is None
            else f"{stat.mean_delta:+.2f}" + (f" {stat.unit}" if stat.unit else "")
        )
        lines.append(
            "| "
            + " | ".join([
                f"{stat.label} ({stat.unit})" if stat.unit else stat.label,
                _fmt(stat.baseline_mean),
                _fmt(stat.treatment_mean),
                delta,
                _fmt_p(stat.p_value),
            ])
            + " |"
        )
    footer = (
        f"\nPaired over {success.n_pairs} matched seeds — discordant success pairs: "
        f"{success.treatment_only} won by {treatment}, {success.baseline_only} by {baseline} "
        f"(both {success.both}, neither {success.neither})."
    )
    return "\n".join(lines) + "\n" + footer


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_success_rates(summaries: Sequence[ConfigSummary], path: str | Path) -> None:
    """Bar chart of success rate per config (the headline plot)."""
    figure, axes = plt.subplots(figsize=(5.0, 4.0))
    labels = [s.config_label for s in summaries]
    rates = [100 * s.success_rate for s in summaries]
    # baseline grey, then treatments; enough colors for the M7 3-way (human/ftonly/vision).
    palette = ["#8c8c8c", "#3b7dd8", "#3aa657", "#d1495b"]
    bars = axes.bar(labels, rates, color=[palette[i % len(palette)] for i in range(len(labels))])
    axes.set_ylabel("Insertion success rate (%)")
    axes.set_ylim(0, 100)
    axes.set_title("Phase-1 insertion success — assist off vs on")
    axes.grid(True, axis="y", alpha=0.3)
    for bar, summary in zip(bars, summaries, strict=True):
        axes.annotate(
            f"{100 * summary.success_rate:.0f}%\n({summary.n_success}/{summary.n_trials})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_kpi_distributions(grouped: dict[str, Sequence[TrialKPIs]], path: str | Path) -> None:
    """Small-multiples box plots of each continuous KPI, one box per config."""
    labels = list(grouped)
    figure, axes_row = plt.subplots(
        1, len(CONTINUOUS_KPIS), figsize=(4.0 * len(CONTINUOUS_KPIS), 4.0)
    )
    axes_list = np.atleast_1d(axes_row).ravel()
    for axis, spec in zip(axes_list, CONTINUOUS_KPIS, strict=True):
        data = [_kpi_values(grouped[label], spec) for label in labels]
        # boxplot rejects empty groups — substitute an empty list it tolerates.
        axis.boxplot([d if d.size else [] for d in data], tick_labels=labels, showfliers=False)
        unit = f" ({spec.unit})" if spec.unit else ""
        axis.set_title(f"{spec.label}{unit}")
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("Per-KPI distributions by config")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_paired_deltas(comparison: PairedComparison, path: str | Path) -> None:
    """Bar chart of the mean paired per-seed delta for each continuous KPI.

    Bars are colored by whether the treatment moved the KPI in its good direction
    (green) or the wrong way (red); a KPI with no matched pairs is left blank.
    """
    stats_with_data = [s for s in comparison.kpi_stats if s.mean_delta is not None]
    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    if not stats_with_data:
        axes.text(0.5, 0.5, "no matched pairs", ha="center", va="center")
        axes.set_axis_off()
        figure.savefig(path, dpi=120)
        plt.close(figure)
        return
    labels = [f"{s.label}\n({s.unit})" if s.unit else s.label for s in stats_with_data]
    deltas = [s.mean_delta or 0.0 for s in stats_with_data]
    colors = ["#3aa657" if s.treatment_better else "#d1495b" for s in stats_with_data]
    axes.bar(labels, deltas, color=colors)
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_ylabel(f"mean Δ ({comparison.treatment_label} − {comparison.baseline_label})")
    axes.set_title("Paired per-seed KPI deltas (green = improvement)")
    axes.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Top-level report assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths written by :func:`build_report`, plus the rendered markdown body."""

    markdown: str
    marginal_table: str
    paired_table: str | None  # the primary (baseline vs treatment) comparison
    extra_paired_tables: tuple[str, ...]  # any further pairings (the M7 3-way)
    success_plot: Path
    distributions_plot: Path
    deltas_plot: Path | None


def build_report(
    trials: Sequence[TrialKPIs],
    out_dir: str | Path,
    *,
    baseline_label: str = "human_only",
    treatment_label: str = "residual",
    extra_comparisons: Sequence[tuple[str, str]] = (),
) -> ReportArtifacts:
    """Aggregate trials into tables + plots under ``out_dir``; return the artifacts.

    The marginal table/plots cover **all** configs present, so a 3-way run (M7:
    human-only / F/T-only / vision) shows all three side by side. The primary paired
    comparison + delta plot are ``baseline_label`` vs ``treatment_label``; pass
    ``extra_comparisons`` (e.g. ``[("human_only","ftonly"), ("ftonly","vision")]``) to
    append further paired tables — the M7 headline is one 3-way report, not three
    reports. A single-config run (human-only calibration) reports the marginal side only.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    grouped = group_by_config(trials)

    # Order configs baseline-first so tables/plots read "off then on".
    ordered_labels = [label for label in (baseline_label, treatment_label) if label in grouped]
    ordered_labels += [label for label in grouped if label not in ordered_labels]
    summaries = [summarize_config(label, grouped[label]) for label in ordered_labels]

    marginal_table = format_marginal_table(summaries)
    success_plot = out_path / "success_rates.png"
    distributions_plot = out_path / "kpi_distributions.png"
    plot_success_rates(summaries, success_plot)
    plot_kpi_distributions({label: grouped[label] for label in ordered_labels}, distributions_plot)

    def _paired(base: str, treat: str) -> str | None:
        if base not in grouped or treat not in grouped:
            return None
        return format_paired_table(
            compare_paired(
                grouped[base], grouped[treat], baseline_label=base, treatment_label=treat
            )
        )

    paired_table = _paired(baseline_label, treatment_label)
    deltas_plot: Path | None = None
    if paired_table is not None:
        deltas_plot = out_path / "paired_deltas.png"
        plot_paired_deltas(
            compare_paired(
                grouped[baseline_label],
                grouped[treatment_label],
                baseline_label=baseline_label,
                treatment_label=treatment_label,
            ),
            deltas_plot,
        )

    extra_tables = tuple(
        table for base, treat in extra_comparisons if (table := _paired(base, treat)) is not None
    )

    markdown = _assemble_markdown(marginal_table, paired_table, extra_tables)
    (out_path / "kpi_tables.md").write_text(markdown + "\n", encoding="utf-8")
    return ReportArtifacts(
        markdown=markdown,
        marginal_table=marginal_table,
        paired_table=paired_table,
        extra_paired_tables=extra_tables,
        success_plot=success_plot,
        distributions_plot=distributions_plot,
        deltas_plot=deltas_plot,
    )


def _assemble_markdown(
    marginal_table: str, paired_table: str | None, extra_paired_tables: Sequence[str] = ()
) -> str:
    """Compose the KPI-tables markdown fragment written next to the plots."""
    parts = ["## KPI summary (marginal)", "", marginal_table]
    if paired_table is not None:
        parts += ["", "## Paired comparison (matched seeds)", "", paired_table]
    for table in extra_paired_tables:
        parts += ["", "## Paired comparison (matched seeds)", "", table]
    return "\n".join(parts)
