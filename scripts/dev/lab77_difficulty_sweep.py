"""LAB-77 difficulty-calibration sweep: chamfer width x expert authority.

Uses the LAB-37 ablation mechanism (`ai_teleop.eval.ablation.run_trial`) directly
(not the `evaluate.py sweep` CLI, which only varies `operator_error_scale` -- a
knob that pushes baseline and expert difficulty down together, the opposite of
what LAB-77 wants). The two knobs swept here *decouple* baseline from expert
capability:

  * chamfer width (`sim/scenegen/config.py`'s SamplingRanges.chamfer) -- shared
    physical geometry, but a wider funnel disproportionately helps the *expert*
    (which servos precisely once gated) over the sloppy scripted human-only
    baseline.
  * expert `epsilon_lateral` (`expert/expert.py`) -- the alignment tolerance that
    gates axial advance. Expert-only: never touches the human-only baseline at
    all, the cleanest decoupling lever.

Baseline is computed once per chamfer value (independent of epsilon_lateral) and
reused across the epsilon sweep for that chamfer, rather than recomputed for each
(chamfer, epsilon) pair.

`d_near`/`advance_per_step`/`d_far` are held at their defaults for this pass --
a secondary refinement if the chamfer x epsilon_lateral grid doesn't find a good
operating point.

Chamfer is a scenegen-level default with no CLI knob in `generate_dataset.py`, so
this monkeypatches `ai_teleop.sim.scenegen.config.DEFAULT_RANGES` before scenegen
is first (lazily) imported inside `make_env` -- the same technique used for the
LAB-84 wall-diversity isolation probe. Not a permanent code change.

**One chamfer value per process, by design**: `sample_wall_spec`'s `ranges`
default is bound to the `DEFAULT_RANGES` object at `generate.py`'s *own* import
time (a Python default-argument gotcha -- re-patching `config.DEFAULT_RANGES`
after that first import is silently inert, verified empirically). So this script
takes a single `--chamfer-mm` value and sweeps only `--epsilon-lateral-mm`
in-process; sweep multiple chamfer values by invoking the script once per value
(each a fresh process, so the patch lands before scenegen's first import).

Run: uv run python scripts/dev/lab77_difficulty_sweep.py --chamfer-mm 3 --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import ai_teleop.sim.scenegen.config as scenegen_config  # noqa: E402
from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.eval.ablation import HUMAN_ONLY, Config, run_trial  # noqa: E402
from ai_teleop.expert import Expert  # noqa: E402

log = get_logger("lab77_sweep")


def _set_chamfer(chamfer_fixed: float) -> None:
    """Pin the sampled chamfer to a single fixed value for this sweep point."""
    scenegen_config.DEFAULT_RANGES = replace(
        scenegen_config.DEFAULT_RANGES, chamfer=(chamfer_fixed, chamfer_fixed)
    )


def _expert_config(epsilon_lateral: float, d_far: float) -> Config:
    return Config(
        label=f"expert_eps{epsilon_lateral * 1000:.1f}mm_dfar{d_far * 1000:.0f}mm",
        assist_factory=lambda: Expert(epsilon_lateral=epsilon_lateral, d_far=d_far),
    )


def _success_rate(config: Config, *, seeds: int, master_seed: int) -> float:
    successes = sum(
        run_trial(episode_index, config, master_seed=master_seed).success
        for episode_index in range(seeds)
    )
    return successes / seeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument(
        "--chamfer-mm", type=float, required=True, help="Single fixed chamfer width (mm)."
    )
    ap.add_argument(
        "--epsilon-lateral-mm", default="3", help="Comma-separated expert epsilon_lateral (mm)."
    )
    ap.add_argument(
        "--d-far-mm",
        default="100",
        help="Comma-separated expert d_far gate-engagement distance (mm).",
    )
    add_logging_arguments(ap)
    args = ap.parse_args()
    configure_from_args(args)

    chamfer = args.chamfer_mm / 1000
    epsilons = [float(v) / 1000 for v in args.epsilon_lateral_mm.split(",")]
    d_fars = [float(v) / 1000 for v in args.d_far_mm.split(",")]

    _set_chamfer(chamfer)
    rows: list[dict[str, float]] = []
    baseline = _success_rate(HUMAN_ONLY, seeds=args.seeds, master_seed=args.master_seed)
    log.info("chamfer=%.1fmm  baseline (n=%d) = %.0f%%", chamfer * 1000, args.seeds, 100 * baseline)
    for eps in epsilons:
        for d_far in d_fars:
            expert_rate = _success_rate(
                _expert_config(eps, d_far), seeds=args.seeds, master_seed=args.master_seed
            )
            log.info(
                "  epsilon_lateral=%.1fmm  d_far=%.0fmm  expert (n=%d) = %.0f%%",
                eps * 1000,
                d_far * 1000,
                args.seeds,
                100 * expert_rate,
            )
            rows.append({
                "chamfer_mm": chamfer * 1000,
                "epsilon_lateral_mm": eps * 1000,
                "d_far_mm": d_far * 1000,
                "baseline": baseline,
                "expert": expert_rate,
            })

    print(
        f"\n{'chamfer_mm':<12}{'eps_lat_mm':<12}{'d_far_mm':<10}{'baseline':<10}{'expert':<10}{'lift':<10}"
    )
    print("-" * 64)
    for row in rows:
        print(
            f"{row['chamfer_mm']:<12.1f}{row['epsilon_lateral_mm']:<12.1f}{row['d_far_mm']:<10.0f}"
            f"{row['baseline']:<10.0%}{row['expert']:<10.0%}{row['expert'] - row['baseline']:<10.0%}"
        )


if __name__ == "__main__":
    main()
