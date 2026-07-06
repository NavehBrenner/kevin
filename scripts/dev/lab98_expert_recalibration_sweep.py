"""LAB-98 sweep: expert recalibration under the deployment controller config.

LAB-96 moved the corpus to the deployment (teleop) controller config
(`joint_damping=1.5, max_dpos=0.3`) + the per-episode lognormal approach-speed
draw, and the kd=4-tuned expert stopped reaching its ceiling: dataset_7 measures
expert 56% success / 28% force-abort (was 77.5% / 5% under kd=4). The expert
corrects *aim* but not *approach speed* — under the responsive controller the
arm tracks the operator's command tightly, so a hasty episode slams the wall at
its drawn sweep speed and trips the controller's 30 N watchdog.

This probe is the LAB-77-style calibration harness for that regime. Wiring
mirrors `lab95_scripted_contact_probe.py` (same wall/operator pairing, same
TerminationProbe outcome policy as data generation), but the assist layer is
configurable: a human-only baseline plus a grid of `Expert` variants
(`d_far` x the LAB-98 braking knobs `brake_gain`/`brake_lead_floor`). Chamfer
sweeps stay one-value-per-process (`--chamfer-mm`, same monkeypatch + default-arg
gotcha as `lab77_difficulty_sweep.py`).

Per-config output: outcome counts, the lab95 contact-forensics table, plus
expert-specific diagnostics — effective (post-delta) vs base command lead along
the bore at episode end, and how often the expert's Δ saturates its 2 cm clamp
(the structural bound on braking authority: the expert can never retract the
command more than 2 cm from where the operator put it).

Run (diagnosis, current expert):
    uv run python scripts/dev/lab98_expert_recalibration_sweep.py --seeds 40
Run (brake sweep):
    uv run python scripts/dev/lab98_expert_recalibration_sweep.py --seeds 40 \
        --brake-gain 0.25,0.5 --brake-lead-floor-mm 5,10
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from lab95_recorded_forensics import contact_forensics, print_forensics_table  # noqa: E402

import ai_teleop.sim.scenegen.config as scenegen_config  # noqa: E402
from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.common.utils.rotations import axis_from_quat  # noqa: E402
from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.data.step_callbacks import TerminationProbe  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.domain.interfaces import AssistProvider  # noqa: E402
from ai_teleop.expert import Expert  # noqa: E402
from ai_teleop.input import ScriptedNoisyHuman  # noqa: E402
from ai_teleop.sim.config import EnvConfig, episode_wall_seed  # noqa: E402
from ai_teleop.sim.env_setup import make_env  # noqa: E402
from ai_teleop.sim.runner import run_episode  # noqa: E402

log = get_logger("lab98_sweep")

_TARGET_HOLE_INDEX = 0
_MAX_STEPS = 6000  # matches data-gen (~12 s @ 500 Hz)
_SUCCESS_DEPTH = 0.015
_LATERAL_TOLERANCE = 0.010  # data.generate.DEFAULT_LATERAL_TOLERANCE (LAB-77)
_FORCE_CAP = 50.0  # probe-level raw-force abort; the controller's own 30 N watchdog trips first
_DELTA_CLAMP = 0.02  # domain.delta._MAX_DELTA_POSITION — the expert's per-tick authority
_LEAD_WINDOW_TICKS = 25  # ~50 ms tail over which end-of-episode command lead is averaged


def _human_seed(master_seed: int, episode_index: int) -> int:
    return int(np.random.SeedSequence([master_seed, episode_index]).generate_state(1)[0])


def _set_chamfer(chamfer_fixed: float) -> None:
    """Pin the sampled chamfer for this process (see lab77_difficulty_sweep.py:
    re-patching after scenegen's first import is silently inert, so one chamfer
    value per process)."""
    scenegen_config.DEFAULT_RANGES = replace(
        scenegen_config.DEFAULT_RANGES, chamfer=(chamfer_fixed, chamfer_fixed)
    )


def run_one(
    master_seed: int,
    episode_index: int,
    assist_factory: Callable[[], AssistProvider],
    *,
    joint_damping: float,
    max_dpos: float,
    speed_lognorm_median: float,
    speed_lognorm_sigma: float,
) -> dict:
    """One episode; returns a contact-forensics row + expert diagnostics."""
    wall_seed = episode_wall_seed(master_seed, episode_index)
    environment = make_env(EnvConfig(wall_seed=wall_seed), render_mode="headless")
    try:
        controller = Controller(
            environment, max_dpos_per_step=max_dpos, joint_damping=joint_damping
        )
        observation = environment.reset()
        hole_pose = observation.hole_poses[_TARGET_HOLE_INDEX]
        insertion_axis = axis_from_quat(hole_pose[3:], 0)
        target_pose = np.concatenate([hole_pose[:3], controller.home_pose[3:]])

        human = ScriptedNoisyHuman(
            target_pose,
            seed=_human_seed(master_seed, episode_index),
            speed_lognormal_median=speed_lognorm_median,
            speed_lognormal_sigma=speed_lognorm_sigma,
        )
        probe = TerminationProbe(
            controller,
            target_hole_index=_TARGET_HOLE_INDEX,
            success_depth=_SUCCESS_DEPTH,
            lateral_tolerance=_LATERAL_TOLERANCE,
            force_cap=_FORCE_CAP,
        )
        ee_track: list[np.ndarray] = []
        base_cmd_track: list[np.ndarray] = []
        effective_cmd_track: list[np.ndarray] = []
        force_track: list[float] = []
        delta_norm_track: list[float] = []

        def step_callback(step, obs, base_command, delta, command) -> bool:
            ee_track.append(obs.ee_pose[:3].copy())
            base_cmd_track.append(base_command.target_position.copy())
            effective_cmd_track.append(command.target_position.copy())
            force_track.append(float(np.linalg.norm(obs.wrist_ft[:3])))
            delta_norm_track.append(float(np.linalg.norm(delta.delta_position)))
            return probe(step, obs, base_command, delta, command)

        run_episode(
            environment,
            controller,
            human,
            assist_factory(),
            max_steps=_MAX_STEPS,
            step_callback=step_callback,
        )

        ee = np.array(ee_track)
        base_cmd = np.array(base_cmd_track)
        effective_cmd = np.array(effective_cmd_track)
        row = contact_forensics(
            np.array(force_track),
            ee,
            base_cmd,
            insertion_axis,
            probe.terminal_reason.value,
        )
        motion_mm = np.linalg.norm(np.diff(ee, axis=0), axis=1) * 1e3
        row["motion_med_mm"] = float(np.median(motion_mm)) if len(motion_mm) else float("nan")
        row["n_steps"] = len(ee)
        row["drawn_speed"] = human.max_approach_speed
        row["peak_force"] = float(np.max(force_track)) if force_track else float("nan")

        # Command lead along the bore over the episode tail: what the impedance
        # law was actually pulling toward (effective = post-delta) vs what the
        # operator commanded (base). Their difference is the expert's realized
        # axial authority.
        tail = slice(max(0, len(ee) - _LEAD_WINDOW_TICKS), len(ee))
        row["base_lead_end_mm"] = float(
            np.median((base_cmd[tail] - ee[tail]) @ insertion_axis) * 1e3
        )
        row["effective_lead_end_mm"] = float(
            np.median((effective_cmd[tail] - ee[tail]) @ insertion_axis) * 1e3
        )
        delta_norms = np.array(delta_norm_track)
        engaged = delta_norms > 1e-6
        row["delta_engaged_frac"] = float(np.mean(engaged)) if len(delta_norms) else 0.0
        row["delta_saturated_frac"] = (
            float(np.mean(delta_norms[engaged] >= 0.995 * _DELTA_CLAMP)) if engaged.any() else 0.0
        )
        return row
    finally:
        environment.close()


def _summarize(label: str, rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    n = len(rows)
    counts_str = "  ".join(
        f"{outcome}={count}/{n} ({count / n:.0%})" for outcome, count in sorted(counts.items())
    )
    print(f"\n=== {label} ===")
    print(f"outcomes: {counts_str}")

    drawn = np.array([row["drawn_speed"] for row in rows])
    print(
        f"drawn approach speed (m/s): median={np.median(drawn):.3f} "
        f"p90={np.percentile(drawn, 90):.3f} max={np.max(drawn):.3f}"
    )
    for key, fmt in (
        ("peak_force", "{:.1f}N"),
        ("base_lead_end_mm", "{:.1f}mm"),
        ("effective_lead_end_mm", "{:.1f}mm"),
        ("delta_engaged_frac", "{:.2f}"),
        ("delta_saturated_frac", "{:.2f}"),
    ):
        by_outcome = {}
        for row in rows:
            by_outcome.setdefault(row["outcome"], []).append(row[key])
        parts = [
            f"{outcome}: {fmt.format(float(np.median(values)))}"
            for outcome, values in sorted(by_outcome.items())
        ]
        print(f"  {key:<24} " + "   ".join(parts))

    print()
    print_forensics_table(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--master-seed", type=int, default=950)
    # Deployment (teleop / data-gen) config by default — the LAB-98 regime.
    ap.add_argument("--joint-damping", type=float, default=1.5)
    ap.add_argument("--max-dpos", type=float, default=0.3)
    ap.add_argument("--speed-lognorm-median", type=float, default=0.09)
    ap.add_argument("--speed-lognorm-sigma", type=float, default=0.76)
    # Optional chamfer pin (mm). One value per process — see module docstring.
    ap.add_argument("--chamfer-mm", type=float, default=None)
    # Expert grid (comma-separated lists; the grid is their product).
    ap.add_argument("--d-far-mm", default="100")
    ap.add_argument("--epsilon-lateral-mm", default="3")
    ap.add_argument("--brake-gain", default="0", help="0 disables the LAB-98 brake.")
    ap.add_argument("--brake-lead-floor-mm", default="8")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-expert", action="store_true")
    add_logging_arguments(ap)
    args = ap.parse_args()
    configure_from_args(args)

    if args.chamfer_mm is not None:
        _set_chamfer(args.chamfer_mm / 1000)

    rollout_config = dict(
        joint_damping=args.joint_damping,
        max_dpos=args.max_dpos,
        speed_lognorm_median=args.speed_lognorm_median,
        speed_lognorm_sigma=args.speed_lognorm_sigma,
    )
    print(
        f"n_seeds={args.seeds} master_seed={args.master_seed} "
        + " ".join(f"{key}={value}" for key, value in rollout_config.items())
        + (f" chamfer_mm={args.chamfer_mm}" if args.chamfer_mm is not None else "")
    )

    def run_config(label: str, assist_factory: Callable[[], AssistProvider]) -> None:
        rows = []
        for episode_index in range(args.seeds):
            rows.append(
                run_one(
                    args.master_seed,
                    episode_index,
                    assist_factory,
                    **rollout_config,
                )
            )
        _summarize(label, rows)

    if not args.skip_baseline:
        run_config("baseline (NoAssist)", NoAssist)

    if not args.skip_expert:
        d_fars = [float(v) / 1000 for v in args.d_far_mm.split(",")]
        epsilons = [float(v) / 1000 for v in args.epsilon_lateral_mm.split(",")]
        brake_gains = [float(v) for v in args.brake_gain.split(",")]
        brake_floors = [float(v) / 1000 for v in args.brake_lead_floor_mm.split(",")]
        for d_far, epsilon, gain, floor in product(d_fars, epsilons, brake_gains, brake_floors):
            expert_kwargs: dict[str, float | int] = dict(
                target_hole_index=_TARGET_HOLE_INDEX, d_far=d_far, epsilon_lateral=epsilon
            )
            label = f"expert d_far={d_far * 1000:.0f}mm eps={epsilon * 1000:.1f}mm"
            if gain > 0.0:
                expert_kwargs.update(brake_gain=gain, brake_lead_floor=floor)
                label += f" brake_gain={gain} floor={floor * 1000:.0f}mm"
            run_config(label, lambda kw=expert_kwargs: Expert(**kw))

    print("\ndone.")


if __name__ == "__main__":
    main()
