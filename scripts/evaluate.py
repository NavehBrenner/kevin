"""M6 evaluation driver — paired ablation + human-only difficulty sweep (LAB-37).

The CLI front door to the paired-seed ablation infrastructure (``ai_teleop.eval``).
It does **not** produce the publishable tables/plots — that is LAB-38; this writes the
flat per-trial CSV (one row per seed × config) those consume, and runs the human-only
difficulty sweep used to find an operating point with headroom.

Two subcommands::

    # paired ablation over a seed range → per-trial CSV (+ traces)
    uv run python scripts/evaluate.py pair --seeds 20 --out-dir runs/eval \\
        --residual-checkpoint runs/train/<run>/checkpoint.pt

    # human-only difficulty sweep over the operator-error knob → success rate per setting
    uv run python scripts/evaluate.py sweep --seeds 20 --error-scale 0.1,0.2,0.3,0.5,1.0

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
from ai_teleop.data.generate import DEFAULT_JOINT_DAMPING  # noqa: E402
from ai_teleop.eval.ablation import (  # noqa: E402
    DEFAULT_MAX_DPOS,
    DEFAULT_OPERATOR_ERROR_SCALE,
    DEFAULT_WRIST_RENDER_EVERY,
    HUMAN_ONLY,
    INSERTION_MAX_STEPS,
    Config,
    run_paired,
)

log = get_logger("evaluate")


def _policy_config(checkpoint: str, *, label: str, device: str) -> Config:
    """Build a learned-residual config from a checkpoint (lazy import — needs torch).

    Works for both the F/T-only and the vision checkpoint: whether the policy
    conditions on the wrist image is read from the checkpoint's own config
    (``use_vision``), so the ablation harness turns on the env's wrist capture for
    the vision one automatically — no flag needed here.
    """
    from ai_teleop.policy import LearnedResidual

    return Config(
        label=label,
        assist_factory=lambda: LearnedResidual.from_checkpoint(checkpoint, device=device),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_pair(args: argparse.Namespace) -> int:
    # The M7 3-way ablation (LAB-83): human-only vs F/T-only vs vision, matched
    # seeds. Any subset is fine — supply only the checkpoints you have. `--residual-
    # checkpoint` stays as the single-treatment "residual" label (back-compat).
    configs = [HUMAN_ONLY]
    if args.ftonly_checkpoint:
        configs.append(_policy_config(args.ftonly_checkpoint, label="ftonly", device=args.device))
    if args.vision_checkpoint:
        configs.append(_policy_config(args.vision_checkpoint, label="vision", device=args.device))
    if args.residual_checkpoint:
        configs.append(
            _policy_config(args.residual_checkpoint, label="residual", device=args.device)
        )

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
            joint_damping=args.joint_damping,
            operator_error_scale=args.error_scale,
            wrist_render_every=args.wrist_render_every,
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
    scale_values = [float(v) for v in args.error_scale.split(",")]
    log.info(
        "human-only sweep over operator_error_scale=%s, %d seeds each",
        scale_values,
        args.seeds,
    )
    for error_scale in scale_values:
        successes = 0
        for episode_index in range(args.seeds):
            results = run_paired(
                episode_index,
                [HUMAN_ONLY],
                master_seed=args.master_seed,
                max_steps=args.max_steps,
                operator_error_scale=error_scale,
            )
            successes += int(results["human_only"].success)
        rate = successes / args.seeds
        log.info(
            "error_scale %.3f │ human-only success %3d/%d (%.0f%%)",
            error_scale,
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
        "--max-steps",
        type=int,
        default=INSERTION_MAX_STEPS,
        help="Per-episode step budget (default matches the data-gen corpus; insertion needs ~12 s).",
    )
    add_logging_arguments(common)

    pair = sub.add_parser("pair", parents=[common], help="Paired ablation → per-trial CSV.")
    pair.add_argument("--out-dir", default="runs/eval", help="Where to write trials.csv + traces.")
    pair.add_argument(
        "--max-dpos",
        type=float,
        default=DEFAULT_MAX_DPOS,
        help="Controller command clamp (m/step). Default is the deployment (teleop) "
        "config the corpus is generated under (LAB-98 re-anchor).",
    )
    pair.add_argument(
        "--joint-damping",
        type=float,
        default=DEFAULT_JOINT_DAMPING,
        help="Controller joint-space velocity damping kd. Default is the deployment "
        "(teleop) config (LAB-98 re-anchor), not the Controller's careful-insertion 4.0.",
    )
    pair.add_argument(
        "--residual-checkpoint",
        default=None,
        help="Add a single learned-residual config (label 'residual') from this checkpoint. "
        "For the M7 3-way, prefer --ftonly-checkpoint / --vision-checkpoint instead.",
    )
    pair.add_argument(
        "--ftonly-checkpoint",
        default=None,
        help="F/T-only residual checkpoint → the 'ftonly' config (LAB-83 3-way ablation).",
    )
    pair.add_argument(
        "--vision-checkpoint",
        default=None,
        help="Vision (image+F/T) residual checkpoint → the 'vision' config. The env's "
        "wrist-camera capture is enabled automatically for it (LAB-83).",
    )
    pair.add_argument(
        "--device",
        default="cuda",
        help="Torch device for policy inference (cuda by default — vision needs it for "
        "real-time; falls to the checkpoint on CPU only if you pass --device cpu).",
    )
    pair.add_argument(
        "--wrist-render-every",
        type=int,
        default=DEFAULT_WRIST_RENDER_EVERY,
        help="Vision only: render a new wrist frame every N ticks, hold between "
        "(the env is the frame-rate limiter). Default matches the M7 corpus cadence.",
    )
    pair.add_argument(
        "--error-scale",
        type=float,
        default=DEFAULT_OPERATOR_ERROR_SCALE,
        help="Operator lateral-error scale (the difficulty pin). 1.0 == training σ's "
        "(contact on the flat wall, outside the capture band); <1.0 shrinks the error "
        "toward the chamfer-contact band where the F/T residual has a lever. Locate the "
        "band with the `sweep` subcommand first.",
    )
    pair.set_defaults(func=_run_pair)

    sweep = sub.add_parser("sweep", parents=[common], help="Human-only difficulty sweep.")
    sweep.add_argument(
        "--error-scale",
        default="0.1,0.2,0.3,0.5,1.0",
        help="Comma-separated operator-error scales to sweep (1.0 == training σ's).",
    )
    sweep.set_defaults(func=_run_sweep)

    args = parser.parse_args()
    configure_from_args(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
