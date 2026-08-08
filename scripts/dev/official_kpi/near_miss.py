"""The near-miss result (LAB-121): did the assist get the peg *closer*, seated or not?

Every other reported KPI is either conditioned on seating or outcome-agnostic, so an assist
that improves approach accuracy without crossing the success threshold is invisible. This
scores that: ``min_tip_hole_distance_mm``, the closest the peg tip ever came to the hole centre,
paired by eval seed and aggregated over each recipe's **training** seeds.

Four tables, in the order the argument runs:

1. **Headline** — paired Δ per recipe, mean over training seeds with the observed range.
   Sign convention: negative Δ = the assist got closer = better.
2. **Against the noise floor** — the same Δ next to the spread the *same recipe* produces
   across training seeds. Post-LAB-114 that spread, not the p-value, decides whether a
   number is a finding. `noise_floor.py` prints this for every metric; it is repeated here
   so the near-miss verdict is readable without cross-referencing.
3. **By outcome** — closest approach split by how the trial ended. `force_abort` is the
   largest stratum and a real measurement in its own right ("was the peg driven into the
   plate near the hole, or far from it?"), not contamination to be filtered out. Median
   step count rides along so a shift in *when* aborts happen cannot masquerade as a shift
   in *where*.
4. **Accuracy vs outcome-mix** — direct standardization, because table 3 read on its own is
   actively misleading: the assist moves trials *between* strata, and that migration alone
   moves every stratum's mean. Table 4 holds the mix fixed so the two effects separate.

Read-only. Run from kevin/ (after `backfill_near_miss.py --write`)::

    uv run python scripts/dev/official_kpi/near_miss.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kpi_data import DEFAULT_RUNS_ROOT, EvalPoint, Recipe, load_recipes  # noqa: E402
from plot_kpis import RECIPE_ORDER  # noqa: E402

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.eval.schema import TrialKPIs  # noqa: E402

log = get_logger("near-miss")

METRIC = "min_tip_hole_distance_mm"
OUTCOMES = ("success", "timeout", "force_abort")


def _values(trials: tuple[TrialKPIs, ...]) -> list[float]:
    return [t.min_tip_hole_distance_mm for t in trials if t.min_tip_hole_distance_mm is not None]


def matched_pairs(point: EvalPoint) -> list[tuple[TrialKPIs, TrialKPIs]]:
    """``(baseline, treatment)`` trials sharing an eval seed — the same wall, the same operator."""
    baseline = {trial.seed: trial for trial in point.baseline_trials}
    return [
        (baseline[trial.seed], trial)
        for trial in point.treatment_trials
        if trial.seed in baseline
        and trial.min_tip_hole_distance_mm is not None
        and baseline[trial.seed].min_tip_hole_distance_mm is not None
    ]


def paired_deltas(recipe: Recipe, *, outcome_stable_only: bool = False) -> list[float]:
    """One mean paired Δ (treatment − baseline) per training seed, in millimetres.

    Pairing is by eval seed within a training seed: the same wall, the same operator
    stream, only the assist differs — so the Δ carries no operator or scene variance.

    ``outcome_stable_only`` additionally drops every pair whose two arms ended *differently*
    — see :func:`_report_outcome_stable` for why that is a principled restriction here and
    not just another way of conditioning on the outcome.
    """
    deltas = []
    for point in recipe.points:
        pairs = matched_pairs(point)
        if outcome_stable_only:
            pairs = [(b, t) for b, t in pairs if b.outcome is t.outcome]
        if pairs:
            deltas.append(
                statistics.mean(
                    (t.min_tip_hole_distance_mm or 0.0) - (b.min_tip_hole_distance_mm or 0.0)
                    for b, t in pairs
                )
            )
    return deltas


def _report_headline(recipes: list[Recipe]) -> None:
    log.info("")
    log.info("Closest approach to hole — paired Δ (treatment − human_only), mm")
    log.info("Negative = the assist got the peg closer. Lower is better.")
    log.info("%-22s %6s %10s %10s %22s", "recipe", "seeds", "human", "assist", "paired Δ (range)")
    for recipe in recipes:
        deltas = paired_deltas(recipe)
        if not deltas:
            continue
        human = statistics.mean(
            value for point in recipe.points for value in _values(point.baseline_trials)
        )
        assist = statistics.mean(
            value for point in recipe.points for value in _values(point.treatment_trials)
        )
        log.info(
            "%-22s %6d %9.2f %9.2f %+9.2f  [%+.2f, %+.2f]",
            recipe.label,
            len(deltas),
            human,
            assist,
            statistics.mean(deltas),
            min(deltas),
            max(deltas),
        )


def _report_noise_floor(recipes: list[Recipe]) -> None:
    """The Δ against the spread the same recipe produces across training seeds."""
    log.info("")
    log.info("Against the training-seed noise floor (LAB-114 discipline)")
    log.info("A Δ smaller than its own recipe's spread is not established, whatever its sign.")
    log.info("%-22s %12s %12s %10s", "recipe", "mean Δ (mm)", "floor (mm)", "verdict")
    for recipe in recipes:
        deltas = paired_deltas(recipe)
        if len(deltas) < 2:
            log.info("%-22s %12s %12s %10s", recipe.label, "—", "(needs ≥2 seeds)", "—")
            continue
        mean_delta = statistics.mean(deltas)
        floor = max(deltas) - min(deltas)
        verdict = "clears" if abs(mean_delta) > floor else "within floor"
        log.info("%-22s %+12.2f %12.2f %10s", recipe.label, mean_delta, floor, verdict)


def _report_by_outcome(recipes: list[Recipe]) -> None:
    """Closest approach split by how the trial ended — every stratum is a result."""
    log.info("")
    log.info("Closest approach by outcome — mean mm (n trials, median steps)")
    log.info("force_abort is not contamination: it measures where the peg hit the plate.")
    header = "%-22s %-10s " + " ".join(["%26s"] * 2)
    log.info(header, "recipe", "outcome", "human_only", "assist")
    for recipe in recipes:
        for outcome in OUTCOMES:
            cells = []
            for attribute in ("baseline_trials", "treatment_trials"):
                trials = [
                    trial
                    for point in recipe.points
                    for trial in getattr(point, attribute)
                    if trial.outcome.value == outcome and trial.min_tip_hole_distance_mm is not None
                ]
                if not trials:
                    cells.append("—")
                    continue
                distance = statistics.mean(t.min_tip_hole_distance_mm or 0.0 for t in trials)
                steps = statistics.median(t.n_steps for t in trials)
                cells.append(f"{distance:8.2f}  (n={len(trials):4d}, {steps:5.0f} steps)")
            log.info("%-22s %-10s %26s %26s", recipe.label, outcome, *cells)


def _report_standardized(recipes: list[Recipe]) -> None:
    """Split the raw Δ into "the assist aimed better" vs "the assist changed the outcome mix".

    The per-outcome table is dangerously readable: the assist's timeout trials sit far closer
    to the hole than the baseline's, which looks like a large accuracy win. It is mostly
    arithmetic. The assist also *moves trials between strata*, and the timeout stratum is by
    far the most distant one — so pushing trials into it raises the overall mean even when
    every stratum's own mean falls (Simpson's paradox, live).

    Direct standardization separates the two: re-weight the assist's per-stratum means by the
    **baseline's** outcome mix. The result answers "how close would the assist have got, had
    it produced the same mix of successes/timeouts/aborts as the human?" — accuracy with the
    composition shift held still.

    This is a **diagnostic, not the headline**. Standardizing away the outcome mix also
    standardizes away a real behavioural difference: whether a trial force-aborts or times out
    is an outcome the assist genuinely changes, not a nuisance covariate.
    """
    log.info("")
    log.info("Accuracy vs outcome-mix, by direct standardization (diagnostic)")
    log.info("Standardized = assist's per-outcome means re-weighted to the human's outcome mix.")
    log.info("%-22s %12s %14s %14s", "recipe", "human (mm)", "assist raw", "assist std.")
    for recipe in recipes:
        baseline_by_outcome: dict[str, list[float]] = {outcome: [] for outcome in OUTCOMES}
        assist_by_outcome: dict[str, list[float]] = {outcome: [] for outcome in OUTCOMES}
        for point in recipe.points:
            for trials, bucket in (
                (point.baseline_trials, baseline_by_outcome),
                (point.treatment_trials, assist_by_outcome),
            ):
                for trial in trials:
                    if trial.min_tip_hole_distance_mm is not None:
                        bucket[trial.outcome.value].append(trial.min_tip_hole_distance_mm)

        total = sum(len(values) for values in baseline_by_outcome.values())
        if not total or any(not assist_by_outcome[outcome] for outcome in OUTCOMES):
            continue
        human = statistics.mean(v for values in baseline_by_outcome.values() for v in values)
        assist = statistics.mean(v for values in assist_by_outcome.values() for v in values)
        standardized = sum(
            (len(baseline_by_outcome[outcome]) / total)
            * statistics.mean(assist_by_outcome[outcome])
            for outcome in OUTCOMES
        )
        log.info("%-22s %12.2f %14.2f %14.2f", recipe.label, human, assist, standardized)


def _report_outcome_stable(recipes: list[Recipe]) -> None:
    """The Δ restricted to pairs where **both** arms ended the same way.

    This is the cleanest read of "did the assist aim better", and it is a stronger design
    than either the raw Δ or the standardization above.

    *Why it is not just more conditioning-on-the-outcome.* Selecting on one arm's outcome is
    a collider (Rule 5 in the wiki page). Selecting on the **joint** outcome of both arms is
    different: it defines a *principal stratum* — the set of walls whose result the assist
    never flipped — and a stratum defined by both potential outcomes is a fixed property of
    the wall, not something the treatment moved. Comparing inside it is a legitimate causal
    contrast.

    *Why it is computable at all.* Principal stratification is normally crippled by never
    observing both potential outcomes for the same unit. Here the simulation is deterministic
    and the design is paired on the eval seed, so **both arms of every wall are observed** and
    the stratum membership is read off directly rather than modelled. That is a luxury a real
    randomized trial does not have.

    *What it costs.* The stratum excludes exactly the walls where the assist did the most —
    every conversion, in both directions. So this answers "on the walls it did not flip, did it
    aim better?", not "is it a better policy". Both directions of migration are printed beside
    it so the excluded set is never invisible.
    """
    log.info("")
    log.info("Outcome-stable pairs only — both arms ended the same way (principal stratum)")
    log.info("Excludes every wall the assist flipped, in either direction; those are counted here.")
    log.info(
        "%-22s %8s %10s %10s %22s",
        "recipe",
        "stable",
        "→ better",
        "→ worse",
        "stable Δ (range)",
    )
    for recipe in recipes:
        deltas = paired_deltas(recipe, outcome_stable_only=True)
        if not deltas:
            continue
        stable = total = gained = lost = 0
        for point in recipe.points:
            for baseline_trial, treatment_trial in matched_pairs(point):
                total += 1
                if baseline_trial.outcome is treatment_trial.outcome:
                    stable += 1
                elif treatment_trial.success:
                    gained += 1
                elif baseline_trial.success:
                    lost += 1
        log.info(
            "%-22s %5d/%-4d %10d %10d %+9.2f  [%+.2f, %+.2f]",
            recipe.label,
            stable,
            total,
            gained,
            lost,
            statistics.mean(deltas),
            min(deltas),
            max(deltas),
        )

    log.info("")
    log.info("%-22s %12s %12s %10s", "recipe", "stable Δ (mm)", "floor (mm)", "verdict")
    for recipe in recipes:
        deltas = paired_deltas(recipe, outcome_stable_only=True)
        if len(deltas) < 2:
            continue
        mean_delta = statistics.mean(deltas)
        floor = max(deltas) - min(deltas)
        verdict = "clears" if abs(mean_delta) > floor else "within floor"
        log.info("%-22s %+12.2f %12.2f %10s", recipe.label, mean_delta, floor, verdict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    loaded = load_recipes(RECIPE_ORDER, arguments.runs_root)
    recipes = [loaded[label] for label in RECIPE_ORDER if label in loaded]
    if not recipes:
        log.error("no finished official evals under %s", arguments.runs_root)
        return 1
    if not any(_values(point.treatment_trials) for recipe in recipes for point in recipe.points):
        log.error("no %s column found — run backfill_near_miss.py --write first", METRIC)
        return 1

    _report_headline(recipes)
    _report_noise_floor(recipes)
    _report_by_outcome(recipes)
    _report_standardized(recipes)
    _report_outcome_stable(recipes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
