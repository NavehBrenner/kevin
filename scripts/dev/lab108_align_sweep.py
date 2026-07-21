"""LAB-108 alignment-tolerance sweep: does a tighter advance gate lift the expert
ceiling that brake/clamp/d_far couldn't?

Context: the brake/clamp/d_far sweeps (scripts/dev/lab108_brake_sweep.py) showed the
expert's ~73-75% seating ceiling is inert to approach *speed* and per-step
*authority* — prevention only converts force-aborts into timeouts. One untested
lever remains, targeting a different failure than the operator slam: **jam-on-
advance**. The expert advances axially the moment `norm(lateral_error) <
epsilon_lateral` (3 mm) and `angular_error < epsilon_angular` (8°). If those gates
are too loose, it commits into the bore while still slightly cocked and catches the
rim → a force-abort that is the *expert's* doing, not the operator's. Tightening the
gate (advance only when better aligned) should remove those — but risks never
committing → timeouts, so both failure modes are reported. Paired with the sweep-2
brake (gain 0.5) so the peg is held back while it aligns rather than timing out.

Measured through the eval ablation harness (`run_trial`, deployment config, the
LAB-107 `max_steps=9000` fix) with an `Expert(...)` factory — the same path
`scripts/evaluate.py pair` uses, and the LAB-77 pattern. Expert-only closed-loop
seating on the generated eval walls is the ceiling any BC clone inherits.

Run: uv run python scripts/dev/lab108_align_sweep.py --seeds 60 --master-seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from ai_teleop.common.log import add_logging_arguments, configure_from_args, get_logger
from ai_teleop.eval.ablation import Config, run_trial
from ai_teleop.eval.schema import TrialOutcome
from ai_teleop.expert import Expert

log = get_logger("lab108-align")

# (epsilon_lateral_mm, epsilon_angular_deg, brake_gain). Row 0 is the deployment
# default (eps 3mm/8°, brake gain 1.0); the rest tighten the advance gate under the
# sweep-2 brake (gain 0.5). Last row is brake-only (loose gate) for reference.
SETTINGS: list[tuple[float, float, float]] = [
    (3.0, 8.0, 1.0),  # deployment default — baseline
    (1.5, 4.0, 0.5),
    (1.0, 4.0, 0.5),
    (1.0, 2.0, 0.5),
    (1.5, 8.0, 0.5),
    (3.0, 8.0, 0.5),  # brake-only, loose gate — isolates the gate effect
]


def _expert_factory(epsilon_lateral: float, epsilon_angular_deg: float, brake_gain: float):
    return lambda: Expert(
        epsilon_lateral=epsilon_lateral,
        epsilon_angular=float(np.deg2rad(epsilon_angular_deg)),
        brake_gain=brake_gain,
        brake_lead_floor=0.008,
        d_far=0.15,
    )


def _rates(config: Config, *, seeds: int, master_seed: int) -> tuple[float, float, float]:
    """Return (seating%, force_abort%, timeout%) over `seeds` paired eval walls."""
    outcomes = [
        run_trial(episode_index, config, master_seed=master_seed).outcome
        for episode_index in range(seeds)
    ]
    n = len(outcomes)
    pct = lambda o: 100.0 * sum(x is o for x in outcomes) / n  # noqa: E731
    return pct(TrialOutcome.SUCCESS), pct(TrialOutcome.FORCE_ABORT), pct(TrialOutcome.TIMEOUT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=60, help="Eval walls (episodes) per setting.")
    ap.add_argument("--master-seed", type=int, default=0)
    add_logging_arguments(ap)
    args = ap.parse_args()
    configure_from_args(args)

    rows: list[tuple[float, float, float, float, float]] = []
    for eps_lat_mm, eps_ang_deg, brake_gain in SETTINGS:
        config = Config(
            label=f"eps{eps_lat_mm:g}mm_{eps_ang_deg:g}deg_g{brake_gain:g}",
            assist_factory=_expert_factory(eps_lat_mm / 1000, eps_ang_deg, brake_gain),
        )
        log.info(
            "eval %d walls · eps_lat=%.1fmm eps_ang=%.0f° brake_gain=%.2f",
            args.seeds,
            eps_lat_mm,
            eps_ang_deg,
            brake_gain,
        )
        seating, abort, timeout = _rates(config, seeds=args.seeds, master_seed=args.master_seed)
        rows.append((eps_lat_mm, eps_ang_deg, seating, abort, timeout))

    log.info(
        "=== LAB-108 alignment-tolerance sweep (n=%d, seed=%d) ===", args.seeds, args.master_seed
    )
    log.info("%-8s %-8s %8s %8s %8s", "eps_lat", "eps_ang", "seat%", "abort%", "timeout%")
    for eps_lat_mm, eps_ang_deg, seating, abort, timeout in rows:
        log.info(
            "%-8.1f %-8.0f %8.1f %8.1f %8.1f", eps_lat_mm, eps_ang_deg, seating, abort, timeout
        )
    best = max(rows, key=lambda r: r[2])
    log.info("best seating: eps_lat=%.1fmm eps_ang=%.0f° → %.1f%%", best[0], best[1], best[2])


if __name__ == "__main__":
    main()
