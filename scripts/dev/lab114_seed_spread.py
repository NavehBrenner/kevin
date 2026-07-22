"""LAB-114 G2: how much does the *recipe* vary across training seeds?

Every M5–M7 conclusion in this project rests on one checkpoint per condition. Until
LAB-114 seeded training, "one checkpoint" also meant "one unrepeatable draw" — which is
how the Phase-1 headline (+33.3 pp) came to be unreproducible. This aggregates the N
seeded retrains of one fixed recipe (same corpus, same hyperparameters, seeds 0..N-1),
each evaluated against `human_only` on the *same* 100 paired eval seeds, and prints the
spread: the number that decides whether any single-checkpoint result here is meaningful.

Also answers the free secondary question: is `best_val_loss` predictive of closed-loop
success across seeds? (LAB-106 found offline metrics anti-predictive *across
interventions*; across seeds of one recipe it was unmeasured.)

Read-only. Run: `uv run python scripts/dev/lab114_seed_spread.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from ai_teleop.eval.report import compare_paired, group_by_config, load_trials  # noqa: E402

SEEDS = range(5)
RUN_DIR = Path("outputs/policy/runs")
EVAL_DIR = Path("runs")
RESULTS = Path("docs/results/phase-1")
PLOT_PATH = RESULTS / "lab114_val_loss_vs_success.png"
HEADLINE_TRIALS = RESULTS / "band_scale0.4_trials.csv"  # the 30 seeds behind the 70.0% claim
HEADLINE_RATE = 70.0  # what the unreproducible 2026-07-07 checkpoint scored on them


def _paired(grouped: dict, keep: set[int] | None = None):
    """The paired success contingency, optionally restricted to a seed subset."""

    def subset(trials):
        return [t for t in trials if keep is None or t.seed in keep]

    return compare_paired(
        subset(grouped["human_only"]),
        subset(grouped["residual"]),
        baseline_label="human_only",
        treatment_label="residual",
    ).success


def main() -> None:
    # The 2026-07-07 headline (36.7% → 70.0%) was measured on *these* 30 seeds, which are a
    # subset of the 100 used here — so every checkpoint below can be re-scored on exactly
    # the seed set that produced 70.0%, with no extra compute and no seed-set confound.
    headline_seeds = {
        t.seed
        for t in group_by_config(load_trials(HEADLINE_TRIALS))["human_only"]
        if t.seed is not None
    }

    rows = []
    for seed in SEEDS:
        metadata = json.loads((RUN_DIR / f"lab114_seed{seed}" / "metadata.json").read_text())
        grouped = group_by_config(load_trials(EVAL_DIR / f"eval_lab114_seed{seed}" / "trials.csv"))
        success = _paired(grouped)
        headline_subset = _paired(grouped, headline_seeds)
        rows.append({
            "residual_30": 100 * headline_subset.treatment_rate,
            "baseline_30": 100 * headline_subset.baseline_rate,
            "seed": seed,
            "best_val_loss": metadata["results"]["best_val_loss"],
            "epochs_run": metadata["results"]["epochs_run"],
            "baseline": 100 * success.baseline_rate,
            "residual": 100 * success.treatment_rate,
            "delta": 100 * success.rate_delta,
            "b": success.treatment_only,
            "c": success.baseline_only,
            "p": success.p_value,
            "n": success.n_pairs,
        })

    print(
        "\n| train seed | best_val_loss | epochs | human_only | residual | Δ pp | b/c | "
        "McNemar p | n | residual on the 30 headline seeds |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['seed']} | {r['best_val_loss']:.5f} | {r['epochs_run']} | {r['baseline']:.1f}% "
            f"| {r['residual']:.1f}% | {r['delta']:+.1f} | {r['b']}/{r['c']} "
            f"| {r['p']:.4f} | {r['n']} | {r['residual_30']:.1f}% (n=30) |"
        )

    deltas = [r["delta"] for r in rows]
    residuals = [r["residual"] for r in rows]
    losses = [r["best_val_loss"] for r in rows]
    # The baseline arm uses no checkpoint, so it must be identical across runs — if it
    # is not, the eval harness moved and the spread below is not a training-seed spread.
    baselines = {round(r["baseline"], 6) for r in rows}
    print(f"\nhuman_only across all runs: {sorted(baselines)} (must be a single value)")
    print(
        f"paired Δ: mean {sum(deltas) / len(deltas):+.1f} pp, "
        f"range [{min(deltas):+.1f}, {max(deltas):+.1f}] pp, spread {max(deltas) - min(deltas):.1f} pp"
    )
    print(
        f"residual success: mean {sum(residuals) / len(residuals):.1f}%, "
        f"range [{min(residuals):.1f}, {max(residuals):.1f}]%"
    )

    # The decisive comparison for H-A: on the *same* 30 seeds that produced 70.0%, what does
    # the recipe actually produce? A seed spread can only explain the headline if 70.0% is
    # inside this range.
    on_headline = [r["residual_30"] for r in rows]
    print(
        f"\non the 30 headline seeds: human_only {rows[0]['baseline_30']:.1f}%, "
        f"residual range [{min(on_headline):.1f}, {max(on_headline):.1f}]% over {len(rows)} "
        f"training seeds — the 2026-07-07 checkpoint scored {HEADLINE_RATE:.1f}% "
        f"({'inside' if min(on_headline) <= HEADLINE_RATE <= max(on_headline) else 'OUTSIDE'} "
        f"the range)"
    )

    # n=5 — a rank correlation here is a direction, not a measurement. Reported with its n.
    rho = stats.spearmanr(losses, residuals)
    print(
        f"best_val_loss vs residual success: Spearman ρ={rho.statistic:+.2f} (p={rho.pvalue:.3f}, n={len(rows)})"
    )

    figure, axes = plt.subplots(figsize=(5.5, 4.0))
    axes.scatter(losses, residuals, s=60)
    for r in rows:
        axes.annotate(
            f"seed {r['seed']}",
            (r["best_val_loss"], r["residual"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    axes.set_xlabel("best_val_loss (offline)")
    axes.set_ylabel("closed-loop success @ es0.4 (%)")
    axes.set_title(f"One recipe, {len(rows)} training seeds × 100 paired eval seeds")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=120)
    print(f"\nplot → {PLOT_PATH}")


if __name__ == "__main__":
    main()
