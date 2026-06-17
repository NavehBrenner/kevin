"""M3 runner CLI — thin entry point over `ai_teleop.sim.runner`.

The reusable per-episode loop is core functionality and lives in the package
(`ai_teleop.sim.runner.run_episode`); this script is just its command-line front
door (also reachable as `kvn episode`). It builds the concrete no-assist stack
(scene + controller + scripted human aimed at the trial's target hole +
NoAssist) and reports a one-line summary.

Run from the `code/` directory:

    uv run python scripts/run_episode.py                 # interactive viewer
    uv run python scripts/run_episode.py --headless      # CI / batch
    uv run python scripts/run_episode.py --headless --seed 7 --max-steps 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.input import ScriptedNoisyHuman  # noqa: E402
from ai_teleop.sim.runner import DEFAULT_MAX_STEPS, run_episode  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402
from ai_teleop.sim.scene_source import resolve_scene_path  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--headless", action="store_true", help="Skip the viewer; run the loop and print a summary."
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Seed for the scripted human's noise and the SimEnv."
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Episode step budget (one step == one 2 ms sim tick).",
    )
    p.add_argument(
        "--generated-wall",
        action="store_true",
        help="Run on a freshly generated procedural wall instead of the static scene.",
    )
    p.add_argument("--wall-seed", type=int, default=7, help="Seed for --generated-wall.")
    p.add_argument(
        "--distractors", type=int, default=None, help="Distractor-hole count for --generated-wall."
    )
    p.add_argument(
        "--max-dpos",
        type=float,
        default=0.025,
        help="Controller command clamp in m/step (approach-speed / strictness knob).",
    )
    args = p.parse_args()

    scene_path = resolve_scene_path(
        generated=args.generated_wall,
        wall_seed=args.wall_seed,
        distractors=args.distractors,
    )
    if not scene_path.exists():
        print(f"FATAL: scene file not found at {scene_path}", file=sys.stderr)
        return 2

    render_mode = "headless" if args.headless else "viewer"
    print(f"Loading scene ({render_mode}): {scene_path}")
    env = SimEnv(str(scene_path), render_mode=render_mode, seed=args.seed)
    obs = env.reset()
    if not args.headless:
        env.launch_viewer()

    controller = Controller(env, max_dpos_per_step=args.max_dpos)

    # Aim the scripted human at the active trial's hole *position*, but keep the
    # home grasp orientation rather than the hole-site frame: M3 is plumbing, and
    # commanding an arbitrary wrist reorientation would make the crude scripted
    # approach fight the impedance law. Real orientation corrections are the
    # expert's job (M4). The controller's 2 cm/step command clamp turns the
    # full-target command into a smooth bounded approach.
    target_position = obs.hole_poses[obs.target_hole_index][:3].copy()
    home_quat = controller.home_pose[3:]
    target_pose = np.concatenate([target_position, home_quat])
    human = ScriptedNoisyHuman(target_pose, seed=args.seed)
    assist = NoAssist()

    start_dist = float(np.linalg.norm(obs.ee_pose[:3] - target_position))
    print(
        f"Target hole {obs.target_hole_index} at "
        f"{np.array2string(target_position, precision=3)} "
        f"({start_dist * 1000:.0f} mm from home EE)"
    )
    print(f"Running {args.max_steps} steps with ScriptedNoisyHuman + NoAssist...")

    result = run_episode(
        env,
        controller,
        human,
        assist,
        max_steps=args.max_steps,
        render=not args.headless,
    )

    final_dist = float(np.linalg.norm(result.final_observation.ee_pose[:3] - target_position))
    print(
        f"\nEpisode done: {result.n_steps} steps  "
        f"final lock state = {result.lock_status.state.value}  "
        f"EE-to-hole {start_dist * 1000:.0f} mm -> {final_dist * 1000:.0f} mm"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
