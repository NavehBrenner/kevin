"""M4 data-generation driver — produce the behavioral-cloning corpus.

Runs N unattended episodes (coverage-randomized scene → realistic noisy human →
analytical expert → controller → sim) and writes **one NPZ trajectory file per
episode** under ``--out``. This is the BC training corpus M5 trains against.

The per-tick loop itself stays in `run_episode` (logging-free); this driver
bolts logging on through its ``step_callback`` hook, detects the episode's
terminal condition (insertion depth → success, force-cap → abort, timeout →
failure), and keeps **all** episodes (failures included — diverse state coverage
helps BC). Every episode is reproducible from ``(seed, episode_index)``.

Run from the `code/` directory:

    uv run python scripts/generate_dataset.py --episodes 200 --out data/runs/dev
    uv run python scripts/generate_dataset.py --episodes 5 --out /tmp/smoke --max-steps 800

The on-disk schema is the stable contract M5 reads — see
`src/ai_teleop/data/trajectory.py` and `docs/data-schema.md`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from run_episode import run_episode  # noqa: E402

from ai_teleop.common.observation import Observation  # noqa: E402
from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.data import EpisodeRecorder, TerminalReason  # noqa: E402
from ai_teleop.domain import Delta  # noqa: E402
from ai_teleop.expert import Expert  # noqa: E402
from ai_teleop.input import ScriptedNoisyHuman  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = REPO_ROOT / "assets" / "mjcf" / "full_scene.xml"

_PEG_HALF_LENGTH = 0.030
DEFAULT_MAX_STEPS = 6000  # ~12 s — enough to approach and seat the peg.
DEFAULT_SUCCESS_DEPTH = 0.015  # insertion past the hole entry → success (m)
DEFAULT_LATERAL_TOLERANCE = 0.006  # max lateral error for a "seated" peg (m)
DEFAULT_FORCE_CAP = 50.0  # wrist force magnitude that aborts the episode (N)
DEFAULT_MAX_DPOS = 0.025  # controller command clamp (m/step); approach-speed knob
DEFAULT_EXPERT_D_FAR = 0.10  # distance (m) at which the expert starts engaging


def _episode_fingerprint(
    *, seed: int, max_steps: int, max_dpos: float, expert_d_far: float, scene_path: str
) -> str:
    """Stable hash of every input that determines an episode's trajectory.

    Two runs with the same fingerprint produce byte-identical episodes, so a
    cached file carrying this fingerprint can be reused instead of re-simulated.
    """
    import hashlib

    payload = f"{seed}|{max_steps}|{max_dpos:.6f}|{expert_d_far:.6f}|{Path(scene_path).name}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cached_matches(path: Path, fingerprint: str) -> bool:
    """True if ``path`` is a readable episode whose stored fingerprint matches."""
    import json

    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
    except (OSError, ValueError, KeyError):
        return False  # unreadable / truncated → regenerate
    return bool(metadata.get("fingerprint") == fingerprint)


def _peg_tip_and_axis(peg_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, peg_pose[3:])
    axis = rotation.reshape(3, 3)[:, 2]
    return peg_pose[:3] + _PEG_HALF_LENGTH * axis, axis


class _EpisodeLogger:
    """`run_episode` step_callback that records rows and detects termination."""

    def __init__(
        self,
        ft_bias: np.ndarray,
        *,
        success_depth: float,
        lateral_tolerance: float,
        force_cap: float,
    ) -> None:
        self.recorder = EpisodeRecorder()
        self.terminal_reason = TerminalReason.TIMEOUT
        self._ft_bias = ft_bias
        self._success_depth = success_depth
        self._lateral_tolerance = lateral_tolerance
        self._force_cap = force_cap

    def __call__(
        self,
        step: int,
        observation: Observation,
        base_command,
        delta: Delta,
        command,
    ) -> bool:
        tip, _ = _peg_tip_and_axis(observation.peg_pose)
        hole_pose = observation.hole_poses[observation.target_hole_index]
        insertion_axis = np.zeros(9)
        mujoco.mju_quat2Mat(insertion_axis, hole_pose[3:])
        insertion_axis = insertion_axis.reshape(3, 3)[:, 0]

        error = hole_pose[:3] - tip
        distance = float(np.linalg.norm(error))
        axial_error = float(error @ insertion_axis)
        lateral_error = float(np.linalg.norm(error - axial_error * insertion_axis))
        penetration = -axial_error
        force_magnitude = float(np.linalg.norm(observation.wrist_ft[:3]))

        seated = penetration >= self._success_depth and lateral_error < self._lateral_tolerance

        self.recorder.add(
            step=step,
            sim_time=observation.sim_time,
            wrist_ft=observation.wrist_ft - self._ft_bias,  # bias-subtracted
            joint_positions=observation.joint_positions,
            joint_velocities=observation.joint_velocities,
            ee_pose=observation.ee_pose,
            gripper_width=observation.gripper_width,
            cmd_position=base_command.target_position,
            cmd_quaternion=base_command.target_quaternion,
            cmd_grip=base_command.delta_grip_force,
            delta_position=delta.delta_position,
            delta_orientation=delta.delta_orientation,
            delta_grip=delta.delta_grip_force,
            peg_pose=observation.peg_pose,
            target_hole_pose=hole_pose,
            distance=distance,
            step_success=seated,
        )

        if seated:
            self.terminal_reason = TerminalReason.SUCCESS
            return True
        if force_magnitude > self._force_cap:
            self.terminal_reason = TerminalReason.FORCE_ABORT
            return True
        return False


def generate_dataset(
    out_dir: str | Path,
    n_episodes: int,
    *,
    seed: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
    success_depth: float = DEFAULT_SUCCESS_DEPTH,
    lateral_tolerance: float = DEFAULT_LATERAL_TOLERANCE,
    force_cap: float = DEFAULT_FORCE_CAP,
    max_dpos: float = DEFAULT_MAX_DPOS,
    expert_d_far: float = DEFAULT_EXPERT_D_FAR,
    scene_path: str | Path = SCENE_PATH,
    cache: bool = True,
    progress: bool = False,
) -> list[Path]:
    """Generate ``n_episodes`` trajectory files; return the written paths.

    Keeps every episode (success or failure). Each is reproducible from
    ``(seed, episode_index)`` plus the controller/expert config: the scene
    randomization and the noisy human both derive from the seed. When
    ``cache`` is set, an existing episode file whose stored ``fingerprint``
    matches the current config is reused instead of being re-simulated.
    """
    out_dir = Path(out_dir)
    environment = SimEnv(str(scene_path), render_mode="headless", seed=seed, randomize=True)
    controller = Controller(environment, max_dpos_per_step=max_dpos)
    expert = Expert(d_far=expert_d_far)
    home_quaternion = controller.home_pose[3:]
    fingerprint = _episode_fingerprint(
        seed=seed,
        max_steps=max_steps,
        max_dpos=max_dpos,
        expert_d_far=expert_d_far,
        scene_path=str(scene_path),
    )

    written: list[Path] = []
    for episode_index in range(n_episodes):
        path = out_dir / f"episode_{episode_index:05d}.npz"
        if cache and _cached_matches(path, fingerprint):
            written.append(path)
            if progress:
                print(f"  episode {episode_index:5d} │ ✓ loaded from cache")
            continue
        # Reset once to read the randomized target + tare the F/T bias, then let
        # run_episode reset to the identical state (deterministic per index).
        # Clear the controller's lock too: it persists across episodes, so a
        # prior force-cap → HOLD trip would otherwise freeze every later episode.
        controller.reset()
        observation = environment.reset(episode_index)
        target_position = observation.hole_poses[observation.target_hole_index][:3].copy()
        ft_bias = observation.wrist_ft.copy()

        human_seed = int(np.random.SeedSequence([seed, episode_index]).generate_state(1)[0])
        human = ScriptedNoisyHuman(
            np.concatenate([target_position, home_quaternion]), seed=human_seed
        )
        logger = _EpisodeLogger(
            ft_bias,
            success_depth=success_depth,
            lateral_tolerance=lateral_tolerance,
            force_cap=force_cap,
        )

        run_episode(
            environment,
            controller,
            human,
            expert,
            max_steps=max_steps,
            reset_episode_index=episode_index,
            step_callback=logger,
        )

        logger.recorder.save(
            path,
            metadata={
                "master_seed": seed,
                "episode_index": episode_index,
                "fingerprint": fingerprint,
                "max_dpos": max_dpos,
                "expert_d_far": expert_d_far,
                "target_hole_index": int(observation.target_hole_index),
                "terminal_reason": logger.terminal_reason.value,
                "episode_success": logger.terminal_reason is TerminalReason.SUCCESS,
                "success_depth": success_depth,
                "lateral_tolerance": lateral_tolerance,
                "force_cap": force_cap,
            },
        )
        written.append(path)
        if progress:
            print(
                f"  episode {episode_index:5d} │ generated · {len(logger.recorder):5d} steps "
                f"· {logger.terminal_reason.value}"
            )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to run.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/runs/dev",
        help="Output directory for NPZ files (default: data/runs/dev).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
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
        "--force",
        action="store_true",
        help="Regenerate even if a cached episode with a matching fingerprint exists.",
    )
    args = parser.parse_args()

    if not SCENE_PATH.exists():
        print(f"FATAL: scene file not found at {SCENE_PATH}", file=sys.stderr)
        return 2

    print(f"Generating {args.episodes} episodes → {args.out}  (seed={args.seed})")
    start = time.time()
    written = generate_dataset(
        args.out,
        args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        max_dpos=args.max_dpos,
        expert_d_far=args.expert_d_far,
        cache=not args.force,
        progress=True,
    )
    elapsed = time.time() - start
    print(f"Wrote {len(written)} episode files in {elapsed:.1f}s → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
