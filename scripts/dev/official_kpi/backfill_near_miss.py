"""Backfill the LAB-121 near-miss columns into existing ``trials.csv`` files.

The near-miss trio (``min_tip_hole_distance`` + its axial/lateral split at the same step)
is a pure function of the realized state, and every official eval already logged that
state per tick under ``runs/<run>/traces/``. So the metric is recoverable for the whole
back-catalogue **without re-running a single episode** — replay each trace through the
canonical :class:`~ai_teleop.eval.observer.TrialObserver` via
:func:`~ai_teleop.eval.ablation.replay_kpis` and rewrite the CSV.

**The fidelity gate is the point.** Before touching a run's CSV, every replayed trial must
reproduce that trial's *already-stored* KPI columns (outcome, peak force, jerk, step count).
If it does, the replay path is provably the same calculator that scored the run live and the
new columns are trustworthy; if it does not, the run is skipped and reported. This costs
nothing — the replay had to happen anyway — and it is the only thing standing between a
backfill and silently corrupting the project's evidence base.

``runs/`` is gitignored, so these CSVs have **no git safety net**: the default is a dry run
that reports the gate and changes nothing. Pass ``--write`` to actually rewrite (temp file +
atomic rename, so an interrupted run cannot leave a truncated CSV).

Run from kevin/::

    uv run python scripts/dev/official_kpi/backfill_near_miss.py            # dry run
    uv run python scripts/dev/official_kpi/backfill_near_miss.py --write
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.eval.ablation import replay_kpis  # noqa: E402
from ai_teleop.eval.schema import TrialKPIs  # noqa: E402

log = get_logger("backfill-near-miss")

# Columns the replay must reproduce for the run to be considered sound. `outcome` is
# compared as a string; the rest are floats/ints compared with a tight relative tolerance
# (the arithmetic is deterministic and in the same order, so this is a formality — but an
# exact-equality gate would trip on CSV text round-tripping rather than on real drift).
GATE_COLUMNS = ("outcome", "peak_contact_force", "jerk_integral", "n_steps")
GATE_TOLERANCE = 1e-9

NEW_COLUMNS = (
    "min_tip_hole_distance_mm",
    "penetration_at_closest_mm",
    "lateral_error_at_closest_mm",
)


def _matches(stored: str, replayed: object) -> bool:
    """Is the stored CSV text equal to the replayed value (numerically, where numeric)?"""
    if isinstance(replayed, str):
        return stored == replayed
    if replayed is None:
        return stored in ("", "None")
    try:
        return math.isclose(float(stored), float(replayed), rel_tol=GATE_TOLERANCE, abs_tol=1e-12)
    except ValueError:
        return False


def _gate(row: dict[str, str], kpis: TrialKPIs) -> list[str]:
    """Names of the gate columns where the replay disagrees with the stored row."""
    replayed = kpis.to_dict()
    return [column for column in GATE_COLUMNS if not _matches(row[column], replayed[column])]


def _trace_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "traces").glob("*/episode_*/trace.npz"))


def _write_atomic(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Rewrite the CSV via a sibling temp file + rename, so a crash cannot truncate it."""
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)


def backfill_run(run_dir: Path, *, write: bool) -> tuple[int, int]:
    """Backfill one eval run. Returns ``(trials_updated, gate_mismatches)``.

    A run with any mismatch is left untouched — a partial rewrite of a run whose replay
    path is suspect is worse than no rewrite at all.
    """
    csv_path = run_dir / "trials.csv"
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # (config_label, seed) is the pairing key the whole eval protocol is built on.
    by_key = {(row["config_label"], row["seed"]): row for row in rows}

    mismatches = 0
    updated = 0
    for trace_path in _trace_paths(run_dir):
        kpis = replay_kpis(trace_path)
        key = (str(kpis.config_label), str(kpis.seed))
        row = by_key.get(key)
        if row is None:
            log.warning("%s: trace %s has no CSV row for %s", run_dir.name, trace_path.name, key)
            mismatches += 1
            continue
        differing = _gate(row, kpis)
        if differing:
            if mismatches == 0:  # log the first offender in full, then just count
                log.error(
                    "%s: replay disagrees with stored KPIs for %s on %s "
                    "(stored=%s replayed=%s) — run will be skipped",
                    run_dir.name,
                    key,
                    differing,
                    {column: row[column] for column in differing},
                    {column: kpis.to_dict()[column] for column in differing},
                )
            mismatches += 1
            continue
        replayed = kpis.to_dict()
        for column in NEW_COLUMNS:
            row[column] = replayed[column]
        updated += 1

    if mismatches:
        log.error(
            "%s: %d/%d trials failed the fidelity gate — NOT written",
            run_dir.name,
            mismatches,
            len(rows),
        )
        return 0, mismatches

    for column in NEW_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    if write:
        _write_atomic(csv_path, rows, fieldnames)
    log.info("%s: %d trials %s", run_dir.name, updated, "written" if write else "ready (dry run)")
    return updated, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--glob", default="*", help="Which run directories under --runs-root.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually rewrite trials.csv. Without it this only reports the fidelity gate.",
    )
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    run_dirs = [
        directory
        for directory in sorted(arguments.runs_root.glob(arguments.glob))
        if (directory / "trials.csv").is_file() and (directory / "traces").is_dir()
    ]
    if not run_dirs:
        log.error("no runs with both trials.csv and traces/ under %s", arguments.runs_root)
        return 1

    if not arguments.write:
        log.info("DRY RUN — no file will be modified (pass --write to rewrite)")

    total_updated = 0
    failed_runs = []
    for run_dir in run_dirs:
        updated, mismatches = backfill_run(run_dir, write=arguments.write)
        total_updated += updated
        if mismatches:
            failed_runs.append(run_dir.name)

    log.info(
        "%d/%d runs passed the fidelity gate, %d trials backfilled",
        len(run_dirs) - len(failed_runs),
        len(run_dirs),
        total_updated,
    )
    if failed_runs:
        log.error("runs skipped: %s", ", ".join(failed_runs))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
