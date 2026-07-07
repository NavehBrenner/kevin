"""Regenerate the Phase-1 KPI tables + plots from a stored ablation run (LAB-38).

The reporting front door: it reads the per-trial ``trials.csv`` that
``scripts/evaluate.py pair`` writes and regenerates the publishable comparison — the
KPI tables (markdown), the success-rate / distribution / paired-delta plots, and the
paired summary statistics — with **one command**, which is the LAB-38 acceptance::

    uv run python scripts/report_results.py --trials runs/eval/trials.csv --out-dir runs/eval/report

Every number is a pure function of the stored records (no sim, no re-run), so the
result is reproducible from the committed CSV. With only a human-only config present
(e.g. a calibration run) it reports the marginal side and skips the paired table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.eval.report import build_report, load_trials  # noqa: E402

log = get_logger("report")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        default="runs/eval/trials.csv",
        help="Per-trial CSV written by `evaluate.py pair` (one row per seed × config).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write the tables + plots (default: a `report/` beside the CSV).",
    )
    parser.add_argument(
        "--baseline",
        default="human_only",
        help="Config label for the assist-off baseline.",
    )
    parser.add_argument(
        "--treatment",
        default="residual",
        help="Config label for the assist-on treatment.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    trials_path = Path(args.trials)
    if not trials_path.exists():
        log.error("no trials CSV at %s — run `evaluate.py pair` first", trials_path)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else trials_path.parent / "report"
    trials = load_trials(trials_path)
    log.info("loaded %d trial records from %s", len(trials), trials_path)

    artifacts = build_report(
        trials,
        out_dir,
        baseline_label=args.baseline,
        treatment_label=args.treatment,
    )
    plots = [artifacts.success_plot, artifacts.distributions_plot, artifacts.deltas_plot]
    log.info("wrote KPI tables → %s", out_dir / "kpi_tables.md")
    log.info("wrote plots → %s", ", ".join(p.name for p in plots if p is not None))
    if artifacts.paired_table is None:
        log.warning(
            "only one of {%s, %s} present — reported marginal KPIs only (no paired result)",
            args.baseline,
            args.treatment,
        )
    _echo(artifacts.markdown)
    return 0


def _echo(markdown: str) -> None:
    """Print the tables to stdout, UTF-8-safe on a legacy Windows console.

    The KPI labels carry non-Latin-1 glyphs (e.g. ``∫``); a cp1252 console would
    otherwise raise on ``print``. The ``kpi_tables.md`` file is already UTF-8, so a
    console that still can't encode a glyph degrades to a placeholder rather than
    failing the run.
    """
    text = "\n" + markdown + "\n"
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
        print(text)
    else:  # pragma: no cover - stdout without reconfigure (redirected/wrapped)
        sys.stdout.write(text.encode("utf-8", "replace").decode("utf-8", "replace"))


if __name__ == "__main__":
    raise SystemExit(main())
