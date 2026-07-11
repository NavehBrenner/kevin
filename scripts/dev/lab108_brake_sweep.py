"""LAB-108 approach-brake sweep: raise the expert's seating ceiling by *preventing*
the force-abort slam, measured on the deployment-config corpus.

Context (LAB-106): the dominant insertion failure is a force-abort — the
contact-unaware scripted operator drives the peg into the flat wall outside the
chamfer at speed, jamming it at the force cap. The bounded analytical expert can't
recover that, so its own seating ceiling (~65-72%, LAB-77/LAB-100) caps any BC
clone. The one imitation-compatible lever (LAB-108) is a *better expert* that
prevents the slam. The mechanism already exists — the LAB-98 approach brake
(`expert.py`: retract the command's axial lead beyond `brake_gain*distance +
lead_floor`) — but at the deployment default (gain 1.0, d_far 0.15) the operator's
per-episode lognormal speed draw still has a fast tail it can't arrest.

This sweeps `expert_brake_gain` x `expert_d_far` and reads each corpus's
`metadata.json` expert `success_rate` + terminal-reason counts, so we can find the
setting that drops force-abort without trading it for timeouts (over-braking stalls
the approach → the peg never seats). No code change to the expert — pure knob tuning,
the first rung. Measured only by the closed-loop seating the corpus records (the
teacher-forced offline BC metric is anti-predictive here — LAB-106).

F/T-only (no image render) and `--no-baseline` keep each run fast; the expert ceiling
doesn't need images or the human-only replay. Baseline lift is re-measured once on the
winner via a full `generate_dataset.py` run.

Run: uv run python scripts/dev/lab108_brake_sweep.py --episodes 60 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_teleop.common.log import add_logging_arguments, configure_from_args, get_logger
from ai_teleop.data.generate import (
    DEFAULT_DELTA_CLAMP,
    DEFAULT_EXPERT_BRAKE_LEAD_FLOOR,
    generate_dataset,
)

log = get_logger("lab108-brake")

# Two levers, measured together against the expert's seating ceiling.
#
# BRAKE (LAB-98): retracts the command's axial lead beyond `allowed_lead =
# brake_gain*distance + brake_lead_floor` (expert.py). Braking gets STRONGER as
# brake_gain / brake_lead_floor DECREASE. Sweep-1 (raising gain) weakened the brake
# and worsened seating (73→63%); sweep-2 (lowering gain) halved force-aborts
# (17→8%) but seating stayed ~73-75% because the averted slams became TIMEOUTS —
# braking slows a lateral miss, it doesn't relocate the peg onto the chamfer.
#
# Δ-CLAMP (delta_clamp → the expert's max_delta_position): the per-step position
# authority. The distance gate zeroes the correction in free space, so this bound
# only binds NEAR CONTACT — and the expert is clamp-SATURATED on exactly the abort
# episodes (the wiki's LAB-98/LAB-100 note). So raising it hands lateral authority
# precisely where the slam happens: the peg can align onto the chamfer before axial
# contact instead of being capped at 3 cm/step. This is the one lever with headroom
# (re-opened 2026-07-11) — braking provably can't fix a lateral miss. A larger clamp
# widens the assist envelope (the label bound the BC clone reproduces), so it must
# stay bounded; the sweep finds the smallest clamp that lifts the ceiling.
_F = DEFAULT_EXPERT_BRAKE_LEAD_FLOOR  # 0.008
_C = DEFAULT_DELTA_CLAMP  # 0.03

# DIAGNOSTIC (sweep-4): the Δ-clamp proved inert (sweep-3) — the correction is
# `gate*lateral_error`, and the distance gate is ~0 until the peg is almost at the
# wall, so a large lateral error yields a SMALL gated correction and the clamp never
# binds. The only way to hand the expert real lateral authority is to engage EARLIER
# — widen d_far into free space. This sweeps d_far (with a strong brake + large
# clamp so authority isn't the limiter) to answer one question: is the privileged
# ceiling liftable AT ALL by early engagement? (A yes is not a clone fix — early
# free-space correction needs lateral hole info the deployed policy lacks, LAB-105 —
# it only establishes whether the expert ceiling and the clone are decoupled.)
#
# (brake_gain, brake_lead_floor, delta_clamp, d_far). Row 0 = deployment baseline.
SETTINGS: list[tuple[float, float, float, float]] = [
    (1.0, _F, _C, 0.15),  # deployment default — baseline (~73%)
    (0.5, _F, 0.08, 0.20),
    (0.5, _F, 0.08, 0.30),
    (0.5, _F, 0.08, 0.40),
    (0.25, _F, 0.12, 0.40),
    (0.5, _F, 0.12, 0.60),  # engage very early — upper bound on early-engagement gain
]


def _pct(counts: dict[str, int], reason: str, n: int) -> float:
    return 100.0 * counts.get(reason, 0) / n if n else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=60, help="Episodes per setting.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed (shared across settings).")
    parser.add_argument(
        "--out-root",
        default="data/lab108_brake_sweep",
        help="Root dir; each setting writes a subdir with its own metadata.json.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    rows: list[tuple[float, float, float, float, float]] = []
    for brake_gain, brake_lead_floor, delta_clamp, d_far in SETTINGS:
        out_dir = Path(args.out_root) / f"g{brake_gain:g}_c{delta_clamp:g}_d{d_far:g}"
        log.info(
            "generating %d eps · brake_gain=%.2f floor=%.3f clamp=%.3f d_far=%.2f → %s",
            args.episodes,
            brake_gain,
            brake_lead_floor,
            delta_clamp,
            d_far,
            out_dir,
        )
        generate_dataset(
            out_dir,
            args.episodes,
            seed=args.seed,
            expert_brake_gain=brake_gain,
            expert_brake_lead_floor=brake_lead_floor,
            expert_d_far=d_far,
            delta_clamp=delta_clamp,
            baseline=False,
            render_images=False,
            cache=False,
            progress=False,
        )
        metadata = json.loads((out_dir / "metadata.json").read_text())
        n = metadata["n_episodes"]
        counts = metadata["expert"]["counts"]
        seating = 100.0 * (metadata["expert"]["success_rate"] or 0.0)
        rows.append((
            d_far,
            delta_clamp,
            seating,
            _pct(counts, "force_abort", n),
            _pct(counts, "timeout", n),
        ))

    log.info("=== LAB-108 d_far diagnostic sweep (n=%d, seed=%d) ===", args.episodes, args.seed)
    log.info("%-7s %-7s %8s %8s %8s", "d_far", "clamp", "seat%", "abort%", "timeout%")
    for d_far, delta_clamp, seating, abort, timeout in rows:
        log.info("%-7.2f %-7.3f %8.1f %8.1f %8.1f", d_far, delta_clamp, seating, abort, timeout)
    best = max(rows, key=lambda r: r[2])
    log.info("best seating: d_far=%.2f clamp=%.3f → %.1f%%", best[0], best[1], best[2])


if __name__ == "__main__":
    sys.exit(main())
