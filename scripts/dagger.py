"""DAgger CLI — on-policy expert relabel to close the BC imitation gap (LAB-105).

Thin front door over ``ai_teleop.dagger``. One invocation runs the batched loop:
roll out the current vision policy on a fresh wall family, relabel every visited
state with the corpus expert, aggregate onto the seed corpus, retrain the
frozen-encoder policy, and re-ablate on the held-out eval walls — repeated
``--rounds`` times.

Run from ``kevin/`` (always render on the WSL Mesa-d3d12 GPU path — see
``project-wiki`` / the LAB-105 issue)::

    GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \\
    LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
    uv run python scripts/dagger.py \\
        --base data/dataset_vision \\
        --checkpoint outputs/policy/runs/vision_frozen_ar100/checkpoint.pt \\
        --aggregate data/dagger_agg --rounds 1 --n-rollout 40

The dominant cost is rendering (every on-policy step needs a wrist frame), so a
round is roughly ``n_rollout`` corpus-episodes of render — budget accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.dagger import DEFAULT_ROLLOUT_MASTER_SEED, run_dagger  # noqa: E402

log = get_logger("dagger")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Seed corpus dir (holds metadata.json).")
    parser.add_argument("--checkpoint", required=True, help="Round-0 vision policy checkpoint.")
    parser.add_argument(
        "--aggregate",
        required=True,
        help="Aggregate corpus dir (created; seed corpus symlinked in).",
    )
    parser.add_argument("--rounds", type=int, default=1, help="DAgger rounds.")
    parser.add_argument(
        "--n-rollout", type=int, default=40, help="On-policy rollout episodes per round."
    )
    parser.add_argument(
        "--rollout-master-seed",
        type=int,
        default=DEFAULT_ROLLOUT_MASTER_SEED,
        help="Wall family for rollouts (distinct from corpus seed 82 and eval seed 0).",
    )
    parser.add_argument("--render-every", type=int, default=20, help="Wrist-capture cadence.")
    parser.add_argument("--device", default="cuda", help="Torch device for policy + training.")
    parser.add_argument("--epochs", type=int, default=40, help="Retrain epochs per round.")
    parser.add_argument("--batch-size", type=int, default=2, help="Retrain batch size (8 GB box).")
    parser.add_argument(
        "--action-rate-weight", type=float, default=100.0, help="Smoothness penalty."
    )
    parser.add_argument("--eval-seeds", type=int, default=20, help="Held-out eval walls.")
    parser.add_argument("--error-scale", type=float, default=1.0, help="Eval operator-error scale.")
    parser.add_argument(
        "--runs-root", default="outputs/policy/runs", help="Where retrained runs are written."
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    if not (Path(args.base) / "metadata.json").exists():
        log.error("no metadata.json under %s — not a dataset dir", args.base)
        return 2
    if not Path(args.checkpoint).exists():
        log.error("checkpoint not found: %s", args.checkpoint)
        return 2

    start = time.time()
    results = run_dagger(
        base_dir=args.base,
        checkpoint=args.checkpoint,
        aggregate_dir=args.aggregate,
        runs_root=args.runs_root,
        rounds=args.rounds,
        n_rollout=args.n_rollout,
        rollout_master_seed=args.rollout_master_seed,
        render_every=args.render_every,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        action_rate_weight=args.action_rate_weight,
        eval_seeds=args.eval_seeds,
        error_scale=args.error_scale,
    )
    log.info("DAgger done in %.0fs", time.time() - start)
    log.info("results:\n%s", json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
