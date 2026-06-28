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
    DEFAULT_EXPERT_D_FAR,
    DEFAULT_MAX_APPROACH_SPEED,
    DEFAULT_MAX_DPOS,
    DEFAULT_MAX_STEPS,
    SCENE_PATH,
    generate_dataset,
    regenerate_from_metadata,
)

log = get_logger("datagen")


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
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Per-episode cap.")
    parser.add_argument(
        "--max-dpos",
        type=float,
        default=DEFAULT_MAX_DPOS,
        help="Controller command clamp in m/step (approach-speed / strictness knob).",
    )
    parser.add_argument(
        "--expert-d-far",
        type=float,
        default=DEFAULT_EXPERT_D_FAR,
        help="Distance (m) at which the expert starts engaging.",
    )
    parser.add_argument(
        "--max-approach-speed",
        type=float,
        default=DEFAULT_MAX_APPROACH_SPEED,
        help="Operator command sweep cap in m/s (realism knob; LAB-78/77 fit target).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if a cached episode with a matching fingerprint exists.",
    )
    parser.add_argument(
        "--render-images",
        action="store_true",
        help="Render the wrist camera and save PNG frames into each episode's imgs/ "
        "folder (opt-in M7/vision plumbing; off by default — M5 is F/T-only).",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="With --render-images, save a frame every N recorded steps (cadence knob).",
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

    out_dir = Path(args.out) if args.out is not None else Path("data") / f"dataset_{args.seed}"
    log.info("generating %d episodes → %s  (seed=%d)", args.episodes, out_dir, args.seed)
    start = time.time()
    written = generate_dataset(
        out_dir,
        args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        max_dpos=args.max_dpos,
        expert_d_far=args.expert_d_far,
        max_approach_speed=args.max_approach_speed,
        cache=not args.force,
        baseline=not args.no_baseline,
        render_images=args.render_images,
        render_every=args.render_every,
        progress=True,
    )
    elapsed = time.time() - start
    log.info("wrote %d episode files in %.1fs → %s", len(written), elapsed, out_dir / "runs")
    log.info("dataset summary → %s", out_dir / "metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
