"""LAB-42 H-7: how tightly is the Phase-1 headline actually pinned?

The +33.3 pp headline rests on **12 discordant pairs** out of 30 seeds (11 won by the
residual, 1 by human-only). McNemar's p answers "is the sign real"; it says nothing about
the magnitude. This prints the exact (Clopper-Pearson, conditional on the discordant
count) interval for the paired rate difference, which is what a 100-seed re-run would
tighten — and what D-6 needs in order to score "scale to 100 seeds" honestly.

Read-only. Run: `uv run python scripts/dev/lab42_headline_interval.py`
"""

from __future__ import annotations

from pathlib import Path

from scipy import stats

from ai_teleop.eval.report import compare_paired, group_by_config, load_trials

SETS = [
    ("band es0.4 (headline)", Path("docs/results/phase-1/band_scale0.4_trials.csv"), "residual"),
    ("flat wall es1.0", Path("docs/results/phase-1/flatwall_scale1.0_trials.csv"), "residual"),
    ("100-seed LAB-53", Path("runs/eval/trials.csv"), "residual"),
]


def main() -> None:
    for name, path, treatment in SETS:
        grouped = group_by_config(load_trials(path))
        success = compare_paired(
            grouped["human_only"],
            grouped[treatment],
            baseline_label="human_only",
            treatment_label=treatment,
        ).success
        b, c, n = success.treatment_only, success.baseline_only, success.n_pairs
        discordant = b + c
        # Conditional on the discordant count, b ~ Binomial(discordant, p). The paired
        # rate difference is (2p - 1) * discordant / n, so a CP interval on p maps onto it.
        # Clopper-Pearson (exact) on p, straight from scipy's binomial test.
        cp = stats.binomtest(b, discordant, 0.5).proportion_ci(0.95, method="exact")
        scale = discordant / n
        print(
            f"\n{name}  ({path})\n"
            f"  n={n} pairs, discordant={discordant} (treatment {b} / baseline {c})\n"
            f"  paired difference = {100 * (b - c) / n:+.1f} pp   McNemar p = {success.p_value:.4f}\n"
            f"  95% CI on the difference ≈ {100 * (2 * cp.low - 1) * scale:+.1f} pp "
            f"to {100 * (2 * cp.high - 1) * scale:+.1f} pp"
        )


if __name__ == "__main__":
    main()
