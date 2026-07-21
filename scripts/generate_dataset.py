"""M4 data-generation CLI — thin entry point over `ai_teleop.data.generate`.

The generation pipeline is core functionality and lives in the package
(`ai_teleop.data.generate`); this script is just its command-line front door
(also reachable as `kvn gen`). See that module for the algorithm, the on-disk
layout, and the paired human-only baseline.

Run from the `kevin/` directory:

    uv run python scripts/generate_dataset.py --episodes 200            # → data/dataset_0
    uv run python scripts/generate_dataset.py --episodes 200 --seed 7   # → data/dataset_7
    uv run python scripts/generate_dataset.py --episodes 5 --out /tmp/smoke --max-steps 800
"""

from __future__ import annotations

import argparse
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
from ai_teleop.data.generate import (  # noqa: E402
    SCENE_PATH,
    GenerationConfig,
    generate_dataset,
    regenerate_from_metadata,
)

log = get_logger("datagen")

# The CLI's flag defaults ARE the corpus defaults — read off the config object so
# the two cannot drift.
DEFAULTS = GenerationConfig()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to run.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Dataset directory (default: data/dataset_<seed>). Holds runs/ + metadata.json.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the paired human-only (NoAssist) baseline rollout (~halves wall-clock).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULTS.max_steps, help="Per-episode cap."
    )
    parser.add_argument(
        "--max-dpos",
        type=float,
        default=DEFAULTS.max_dpos,
        help="Controller command clamp in m/step. Default is the deployment (teleop) "
        "config the recorded reference corpus ran under (LAB-96), not the Controller's "
        "careful-insertion 0.025.",
    )
    parser.add_argument(
        "--joint-damping",
        type=float,
        default=DEFAULTS.joint_damping,
        help="Controller joint-space velocity damping kd. Default is the deployment "
        "(teleop) config (LAB-96), not the Controller's careful-insertion 4.0.",
    )
    parser.add_argument(
        "--expert-d-far",
        type=float,
        default=DEFAULTS.expert_d_far,
        help="Distance (m) at which the expert starts engaging.",
    )
    parser.add_argument(
        "--expert-brake-gain",
        type=float,
        default=DEFAULTS.expert_brake_gain,
        help="Expert approach-speed brake gain (LAB-98): allowed command lead is "
        "gain * distance + floor; 0 disables the brake (pre-LAB-98 aim-only expert).",
    )
    parser.add_argument(
        "--expert-brake-lead-floor",
        type=float,
        default=DEFAULTS.expert_brake_lead_floor,
        help="Expert brake lead floor (m) — the minimum allowed command lead.",
    )
    parser.add_argument(
        "--speed-lognormal-median",
        type=float,
        default=DEFAULTS.speed_lognormal_median,
        help="Median (m/s) of the operator's per-episode lognormal max_approach_speed "
        "draw (LAB-96); 0 disables the draw (fixed max_approach_speed).",
    )
    parser.add_argument(
        "--speed-lognormal-sigma",
        type=float,
        default=DEFAULTS.speed_lognormal_sigma,
        help="Log-space sigma of that draw (0.76 fits the recorded corpus' p90/median).",
    )
    parser.add_argument(
        "--delta-clamp",
        type=float,
        default=DEFAULTS.delta_clamp,
        help="Shared expert/policy per-step Δ-position bound (m) — the label bound BC "
        "clones and the brake's authority ceiling (LAB-100). Pre-LAB-100 corpora used 0.02.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if a cached episode with a matching fingerprint exists.",
    )
    parser.add_argument(
        "--record",
        choices=["commands", "all"],
        default="commands",
        help="What to record per episode: 'commands' saves the trajectory episode.npz only "
        "(F/T-only M5 corpus, default); 'all' also renders the wrist camera into each "
        "episode's imgs/ folder (opt-in M7/vision plumbing).",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="With --record all, save a frame every N recorded steps (cadence knob).",
    )
    parser.add_argument(
        "--from-metadata",
        type=str,
        default=None,
        help="Reproduce the dataset described by a metadata.json (rebuilds runs/ from the "
        "committed config); --out overrides where it lands. Ignores generation flags.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    if not SCENE_PATH.exists():
        log.error("scene file not found at %s", SCENE_PATH)
        return 2

    if args.from_metadata is not None:
        log.info("regenerating dataset from %s", args.from_metadata)
        start = time.time()
        written = regenerate_from_metadata(
            args.from_metadata, out_dir=args.out, force=args.force, progress=True
        )
        target = Path(args.out) if args.out is not None else Path(args.from_metadata).parent
        elapsed = time.time() - start
        log.info(
            "regenerated %d episode files in %.1fs → %s", len(written), elapsed, target / "runs"
        )
        return 0

    if args.record == "all" and args.render_every < 2:
        # ponytail: warn, don't block — dense frames are legitimate for a small dev corpus.
        log.warning(
            "--record all with --render-every %d renders ~1 frame/step (hundreds/episode). "
            "For a large corpus this is terabytes on disk; pass --render-every 10-20.",
            args.render_every,
        )

    out_dir = Path(args.out) if args.out is not None else Path("data") / f"dataset_{args.seed}"
    log.info("generating %d episodes → %s  (seed=%d)", args.episodes, out_dir, args.seed)
    start = time.time()
    written = generate_dataset(
        out_dir,
        args.episodes,
        GenerationConfig(
            seed=args.seed,
            max_steps=args.max_steps,
            max_dpos=args.max_dpos,
            joint_damping=args.joint_damping,
            expert_d_far=args.expert_d_far,
            expert_brake_gain=args.expert_brake_gain,
            expert_brake_lead_floor=args.expert_brake_lead_floor,
            speed_lognormal_median=args.speed_lognormal_median,
            speed_lognormal_sigma=args.speed_lognormal_sigma,
            delta_clamp=args.delta_clamp,
        ),
        cache=not args.force,
        baseline=not args.no_baseline,
        render_images=args.record == "all",
        render_every=args.render_every,
        progress=True,
    )
    elapsed = time.time() - start
    log.info("wrote %d episode files in %.1fs → %s", len(written), elapsed, out_dir / "runs")
    log.info("dataset summary → %s", out_dir / "metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
