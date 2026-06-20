"""M6 evaluation driver — paired ablation + human-only difficulty sweep (LAB-37).

The CLI front door to the paired-seed ablation infrastructure (``ai_teleop.eval``).
It does **not** produce the publishable tables/plots — that is LAB-38; this writes the
flat per-trial CSV (one row per seed × config) those consume, and runs the human-only
difficulty sweep used to find an operating point with headroom.

Two subcommands::

    # paired ablation over a seed range → per-trial CSV (+ traces)
    uv run python scripts/evaluate.py pair --seeds 20 --out-dir runs/eval \\
        --residual-checkpoint runs/train/<run>/checkpoint.pt

    # human-only difficulty sweep over the command-clamp knob → success rate per setting
    uv run python scripts/evaluate.py sweep --seeds 20 --max-dpos 0.02,0.025,0.03

Without ``--residual-checkpoint`` the ablation runs human-only only — useful for
calibration, where no policy is needed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.eval.ablation import (  # noqa: E402
    DEFAULT_MAX_DPOS,
    HUMAN_ONLY,
    Config,
    run_paired,
)
from ai_teleop.sim.runner import DEFAULT_MAX_STEPS  # noqa: E402

log = get_logger("evaluate")


def _residual_config(checkpoint: str) -> Config:
    """Build the F/T-residual config from a checkpoint (lazy import — needs torch)."""
    from ai_teleop.policy import LearnedResidual

    return Config(
        label="residual",
        assist_factory=lambda: LearnedResidual.from_checkpoint(checkpoint),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_pair(args: argparse.Namespace) -> int:
    configs = [HUMAN_ONLY]
    if args.residual_checkpoint:
        configs.append(_residual_config(args.residual_checkpoint))

    out_dir = Path(args.out_dir)
    rows: list[dict[str, object]] = []
    for episode_index in range(args.seeds):
        results = run_paired(
            episode_index,
            configs,
            master_seed=args.master_seed,
            out_dir=out_dir / "traces",
            max_steps=args.max_steps,
            max_dpos=args.max_dpos,
        )
        for kpis in results.values():
            rows.append(kpis.to_dict())
        seated = {label: r.success for label, r in results.items()}
        log.info("seed %4d │ %s", episode_index, seated)

    csv_path = out_dir / "trials.csv"
    _write_csv(csv_path, rows)
    for config in configs:
        rate = sum(r["config_label"] == config.label and r["success"] for r in rows) / args.seeds
        log.info("%-12s success rate: %.1f%%", config.label, 100 * rate)
    log.info("wrote %d trial records → %s", len(rows), csv_path)
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    clamp_values = [float(v) for v in args.max_dpos.split(",")]
    log.info("human-only sweep over max_dpos=%s, %d seeds each", clamp_values, args.seeds)
    for max_dpos in clamp_values:
        successes = 0
        for episode_index in range(args.seeds):
            results = run_paired(
                episode_index,
                [HUMAN_ONLY],
                master_seed=args.master_seed,
                max_steps=args.max_steps,
                max_dpos=max_dpos,
            )
            successes += int(results["human_only"].success)
        rate = successes / args.seeds
        log.info(
            "max_dpos %.3f m │ human-only success %3d/%d (%.0f%%)",
            max_dpos,
            successes,
            args.seeds,
            100 * rate,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seeds", type=int, default=20, help="Number of paired seeds (episodes).")
    common.add_argument("--master-seed", type=int, default=0, help="Master seed for the SimEnv.")
    common.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Per-episode step budget."
    )
    add_logging_arguments(common)

    pair = sub.add_parser("pair", parents=[common], help="Paired ablation → per-trial CSV.")
    pair.add_argument("--out-dir", default="runs/eval", help="Where to write trials.csv + traces.")
    pair.add_argument(
        "--max-dpos",
        type=float,
        default=DEFAULT_MAX_DPOS,
        help="Controller command clamp (m/step).",
    )
    pair.add_argument(
        "--residual-checkpoint",
        default=None,
        help="Add the F/T-residual config from this checkpoint (else human-only only).",
    )
    pair.set_defaults(func=_run_pair)

    sweep = sub.add_parser("sweep", parents=[common], help="Human-only difficulty sweep.")
    sweep.add_argument(
        "--max-dpos",
        default="0.02,0.025,0.03",
        help="Comma-separated command-clamp values to sweep (m/step).",
    )
    sweep.set_defaults(func=_run_sweep)

    args = parser.parse_args()
    configure_from_args(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
