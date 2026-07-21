"""LAB-95 probe: scripted raw rollouts, measured with the recorded-corpus forensics.

Companion to `lab95_recorded_forensics.py` (which established that recorded
force-aborts are contact-time transients: the real operator's command is still
sweeping at ~120-260 mm/s when the peg physically meets the wall, because real
operators aim *through* the wall — bore-axial cmd-vs-ee error ~45 mm — so the
near-goal deceleration zone sits deep inside the wall and never fires before
contact). This probe measures the scripted operator with the identical
forensics, and exposes the candidate lever directly:

    --aim-depth-mean / --aim-depth-std   (metres)

shift the operator's target deeper along the hole's insertion axis by a
per-episode draw `N(mean, std)` clipped at >= 0 — simulating "aim through the
hole, not at its entry plane" WITHOUT touching `ScriptedNoisyHuman` (the goal
the actor chases is just `target + depth * insertion_axis`). If the recorded
contact signature (fast contact-time command, 30N+ transient, force_abort rate
~54%, force-abort-vs-success motion differential ~+45%) emerges under a deep
aim, the lever is validated and worth productizing in the operator.

Rollout wiring mirrors `lab92_raw_motion_probe.py` (raw NoAssist +
TerminationProbe — see there for why `eval.ablation.run_trial` is not used).

RESULT (2026-07-06): the aim-depth lever turned out to be moot — the scripted
bore-axial command depth already matches recorded (~45 mm). What closed the gap
was (a) the controller config (`--joint-damping 1.5 --max-dpos 0.3`, the config
the recorded corpus was actually captured under — data-gen uses 4.0/0.025,
which pins realized contact speed at ~55 mm/s and structurally suppresses the
force-abort mechanism) plus (b) a per-episode lognormal `max_approach_speed`
draw (`--speed-lognorm-median 0.09..0.12 --speed-lognorm-sigma 0.76`). See
`project-wiki/synthesis/scripted-vs-real-operator.md` (LAB-95 section).

Run: uv run python scripts/dev/lab95_scripted_contact_probe.py --seeds 40 \
    --joint-damping 1.5 --max-dpos 0.3 --speed-lognorm-median 0.09
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from lab95_recorded_forensics import contact_forensics, print_forensics_table  # noqa: E402

from ai_teleop.common.geometry import axis_from_quat  # noqa: E402
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


def run_one(
    master_seed: int,
    episode_index: int,
    aim_depth_mean: float,
    aim_depth_std: float,
    max_dpos: float,
    joint_damping: float,
    speed_lognorm_median: float,
    speed_lognorm_sigma: float,
) -> dict:
    wall_seed = episode_wall_seed(master_seed, episode_index)
    environment = make_env(EnvConfig(wall_seed=wall_seed), render_mode="headless")
    try:
        controller = Controller(
            environment, max_dpos_per_step=max_dpos, joint_damping=joint_damping
        )
        observation = environment.reset()
        hole_pose = observation.hole_poses[_TARGET_HOLE_INDEX]
        target_position = hole_pose[:3].copy()
        insertion_axis = axis_from_quat(hole_pose[3:], 0)

        # The candidate lever: aim a per-episode depth PAST the hole entry plane,
        # along the bore. Drawn from its own RNG stream (seeded off the human
        # seed) so the operator's bias/drift streams stay untouched.
        if aim_depth_mean > 0.0 or aim_depth_std > 0.0:
            depth_rng = np.random.default_rng(_human_seed(master_seed, episode_index) ^ 0x5EED)
            aim_depth = max(0.0, float(depth_rng.normal(aim_depth_mean, aim_depth_std)))
            target_position = target_position + aim_depth * insertion_axis

        home_quaternion = controller.home_pose[3:]
        target_pose = np.concatenate([target_position, home_quaternion])

        # Per-episode approach-speed draw: the recorded operator's realized
        # contact speed varies ~4x across episodes (26-105 mm/s p10-p90), and
        # that variance IS the force-abort signature (fast episodes abort).
        # Post-LAB-96 the lognormal draw is productized inside the operator
        # itself (drawn from its seeded RNG, after the bias/careless draws), so
        # the probe routes through the shipped code path — n>=40 runs verify the
        # corpus recipe, not a simulation of it. (The original LAB-95 probe drew
        # externally from a separate RNG stream, so per-seed numbers shift
        # slightly vs the RESULT above; the statistics are the same recipe.)
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
        cmd_track: list[np.ndarray] = []
        force_track: list[float] = []

        def step_callback(step, obs, base_command, delta, command) -> bool:
            ee_track.append(obs.ee_pose[:3].copy())
            cmd_track.append(base_command.target_position.copy())
            force_track.append(float(np.linalg.norm(obs.wrist_ft[:3])))
            return probe(step, obs, base_command, delta, command)

        run_episode(
            environment,
            controller,
            human,
            NoAssist(),
            max_steps=_MAX_STEPS,
            step_callback=step_callback,
        )
        ee = np.array(ee_track)
        motion_mm = np.linalg.norm(np.diff(ee, axis=0), axis=1) * 1e3
        row = contact_forensics(
            np.array(force_track),
            ee,
            np.array(cmd_track),
            insertion_axis,
            probe.terminal_reason.value,
        )
        row["motion_med_mm"] = float(np.median(motion_mm)) if len(motion_mm) else float("nan")
        return row
    finally:
        environment.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--master-seed", type=int, default=950)
    ap.add_argument("--aim-depth-mean", type=float, default=0.0)
    ap.add_argument("--aim-depth-std", type=float, default=0.0)
    # Controller config. Defaults match data-gen (`data.generate`); the recorded
    # live-teleop sessions ran `run_episode.py`'s live-input config instead:
    # --max-dpos 0.3 --joint-damping 1.5 (see scripts/run_episode.py) — the
    # LAB-95 contact-time-dynamics candidate confound.
    ap.add_argument("--max-dpos", type=float, default=0.025)
    ap.add_argument("--joint-damping", type=float, default=4.0)
    # Per-episode lognormal draw on max_approach_speed (m/s); 0 disables (keep
    # the operator's fixed 0.35 default). Recorded near-field cmd speed:
    # median ~0.121 m/s, p90/median ~2.7 => sigma ~0.76.
    ap.add_argument("--speed-lognorm-median", type=float, default=0.0)
    ap.add_argument("--speed-lognorm-sigma", type=float, default=0.76)
    args = ap.parse_args()

    rows = [
        run_one(
            args.master_seed,
            episode_index,
            args.aim_depth_mean,
            args.aim_depth_std,
            args.max_dpos,
            args.joint_damping,
            args.speed_lognorm_median,
            args.speed_lognorm_sigma,
        )
        for episode_index in range(args.seeds)
    ]

    print(
        f"\naim_depth_mean={args.aim_depth_mean} aim_depth_std={args.aim_depth_std} "
        f"max_dpos={args.max_dpos} joint_damping={args.joint_damping} "
        f"speed_lognorm_median={args.speed_lognorm_median} "
        f"speed_lognorm_sigma={args.speed_lognorm_sigma} "
        f"n_seeds={args.seeds} master_seed={args.master_seed}"
    )
    n_abort = sum(r["outcome"] == "force_abort" for r in rows)
    print(f"force_abort rate: {n_abort}/{len(rows)} ({n_abort / len(rows):.1%})")

    motion_meds = np.array([r["motion_med_mm"] for r in rows])
    print(
        f"pooled per-episode motion_med: median={np.median(motion_meds):.4f}mm "
        f"p90={np.percentile(motion_meds, 90):.4f}mm "
        f"ratio={np.percentile(motion_meds, 90) / np.median(motion_meds):.3g}x"
    )
    by_outcome: dict[str, list[float]] = {}
    for r in rows:
        by_outcome.setdefault(r["outcome"], []).append(r["motion_med_mm"])
    for outcome, meds in sorted(by_outcome.items()):
        arr = np.array(meds)
        print(f"  motion_med[{outcome}] (n={len(arr)}): {np.median(arr):.4f} mm/step")

    print()
    print_forensics_table(rows)


if __name__ == "__main__":
    main()
