"""LAB-92 probe: raw (pre-assist) realized motion vs recorded, by outcome.

`motion_profile_analysis.py` reads the M4 corpus (`data/dataset_N/runs/`), which
only persists the EXPERT-ASSISTED trajectory -- the expert engages at
`expert_d_far` (0.10 m), well outside `decel_radius` (0.016 m), so its
correction can dominate the realized near-goal motion before a raw operator's
carelessness (LAB-92) ever shows up. `data/recorded` is raw/unassisted. So the
apples-to-apples comparison for calibrating `careless_probability` is raw
(`NoAssist`) vs recorded, not the M4 corpus.

**Do not use `eval.ablation.run_trial` for this** -- its `TrialObserver` (LAB-36)
classifies FORCE_ABORT purely from raw wrist force vs its own `force_cap`
(default 50N), never consulting the controller's own watchdog lock state. Since
`Controller`'s watchdog trips (and freezes the arm) at 30N by default -- lower
than the observer's 50N -- force essentially never climbs to 50N, so
`TrialObserver` can't observe FORCE_ABORT at all under default wiring (verified:
0/40, 0/40, 0/30 force_abort at careless_probability 0.0/0.3/1.0 respectively,
including the 1.0 upper bound). `data.step_callbacks.episode_terminal_reason`
(what data-gen's `TerminationProbe` uses) is the one that's right: it ORs the raw
force check with `locked` (the controller's own HOLD state) -- this is flagged
as a real LAB-36 gap, filed separately, not fixed here (out of LAB-92 scope).

This probe runs a raw NoAssist rollout directly (mirroring
`eval.ablation.run_trial`'s scene/operator setup, but using `TerminationProbe`
for correct outcome classification) at a given `careless_probability`, across N
seeds, recording ee_pose per step, and prints the same recorded-vs-scripted-by-
outcome table `motion_profile_analysis.py` does.

Run: uv run python scripts/dev/lab92_raw_motion_probe.py --seeds 30 --careless-probability 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.data.step_callbacks import TerminationProbe  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.input import ScriptedNoisyHuman  # noqa: E402
from ai_teleop.sim.config import EnvConfig, episode_wall_seed  # noqa: E402
from ai_teleop.sim.env_setup import make_env  # noqa: E402
from ai_teleop.sim.runner import run_episode  # noqa: E402

_TARGET_HOLE_INDEX = 0
_MAX_STEPS = 6000  # matches data-gen's DEFAULT_MAX_STEPS (~12s @ 500Hz)
_SUCCESS_DEPTH = 0.015
_LATERAL_TOLERANCE = 0.010  # post-LAB-77 (data.generate.DEFAULT_LATERAL_TOLERANCE)
_FORCE_CAP = 50.0


def _human_seed(master_seed: int, episode_index: int) -> int:
    return int(np.random.SeedSequence([master_seed, episode_index]).generate_state(1)[0])


def run_one(master_seed: int, episode_index: int, careless_probability: float) -> dict:
    wall_seed = episode_wall_seed(master_seed, episode_index)
    environment = make_env(EnvConfig(wall_seed=wall_seed), render_mode="headless")
    try:
        controller = Controller(environment)
        observation = environment.reset()
        target_position = observation.hole_poses[_TARGET_HOLE_INDEX][:3].copy()
        home_quaternion = controller.home_pose[3:]
        target_pose = np.concatenate([target_position, home_quaternion])
        human = ScriptedNoisyHuman(
            target_pose,
            careless_probability=careless_probability,
            seed=_human_seed(master_seed, episode_index),
        )
        probe = TerminationProbe(
            controller,
            target_hole_index=_TARGET_HOLE_INDEX,
            success_depth=_SUCCESS_DEPTH,
            lateral_tolerance=_LATERAL_TOLERANCE,
            force_cap=_FORCE_CAP,
        )
        ee_positions: list[np.ndarray] = []

        def step_callback(step, obs, base_command, delta, command) -> bool:
            ee_positions.append(obs.ee_pose[:3].copy())
            return probe(step, obs, base_command, delta, command)

        run_episode(
            environment,
            controller,
            human,
            NoAssist(),
            max_steps=_MAX_STEPS,
            step_callback=step_callback,
        )
        ee = np.array(ee_positions)
        motion_mm = np.linalg.norm(np.diff(ee, axis=0), axis=1) * 1e3
        return {
            "outcome": probe.terminal_reason.value,
            "motion_med_mm": float(np.median(motion_mm)) if len(motion_mm) else float("nan"),
            "n_steps": len(ee_positions),
        }
    finally:
        environment.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--master-seed", type=int, default=920)
    ap.add_argument("--careless-probability", type=float, default=0.0)
    args = ap.parse_args()

    rows = [
        run_one(args.master_seed, episode_index, args.careless_probability)
        for episode_index in range(args.seeds)
    ]

    print(f"\ncareless_probability={args.careless_probability}  n_seeds={args.seeds}")
    all_motion_meds = np.array([r["motion_med_mm"] for r in rows])
    print(
        f"pooled per-episode motion_med: median={np.median(all_motion_meds):.4f}mm "
        f"p90={np.percentile(all_motion_meds, 90):.4f}mm "
        f"ratio={np.percentile(all_motion_meds, 90) / np.median(all_motion_meds):.3g}x"
    )

    print(f"\n{'outcome':<14}{'n':<6}{'motion_med mm/step [IQR]':<30}")
    print("-" * 50)
    by_outcome: dict[str, list[float]] = {}
    for r in rows:
        by_outcome.setdefault(r["outcome"], []).append(r["motion_med_mm"])
    for outcome, meds in sorted(by_outcome.items()):
        arr = np.array(meds)
        print(
            f"{outcome:<14}{len(arr):<6}"
            f"{np.median(arr):.3g} [{np.percentile(arr, 25):.2g},{np.percentile(arr, 75):.2g}]"
        )


if __name__ == "__main__":
    main()
