"""M3 end-to-end runner — compose the full no-human loop through the seam.

Wires the M3 stack together for one episode:

    ScriptedNoisyHuman → base Command → AssistProvider → apply_delta
        → Controller.compute → SimEnv.step

`run_episode(...)` is the reusable loop function; M4's data-generation rollout
imports it directly (which is why it is side-effect free beyond its return
value — no printing, no logging). The `__main__` block builds the concrete
stack (scene + controller + scripted human aimed at the trial's target hole +
NoAssist) and reports a one-line summary.

Run from the `code/` directory:

    uv run python scripts/run_episode.py                 # interactive viewer
    uv run python scripts/run_episode.py --headless      # CI / batch
    uv run python scripts/run_episode.py --headless --seed 7 --max-steps 1500

The seam composes *around* the controller, not inside it: `Controller.compute`
still receives a single `Command` and knows nothing about a Δ source. Swapping
NoAssist for the expert (M4) or the learned residual (M5) is a one-argument
change here with no edit to the input strategy or the controller — the
dependency-inversion property M3 exists to establish.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common.observation import Observation  # noqa: E402
from ai_teleop.control import Controller, LockStatus  # noqa: E402
from ai_teleop.domain import NoAssist, apply_delta  # noqa: E402
from ai_teleop.input import ScriptedNoisyHuman  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = REPO_ROOT / "assets" / "mjcf" / "full_scene.xml"

# Sim runs at 500 Hz (dt=2 ms in the MJCF). One control tick == one sim step.
SIM_DT = 0.002
DEFAULT_MAX_STEPS = 2000  # ~4 s of sim time — a full M3 episode budget.


@dataclass(frozen=True)
class EpisodeResult:
    """What one episode leaves behind for the caller to inspect."""

    final_observation: Observation
    lock_status: LockStatus
    n_steps: int


def run_episode(
    environment,
    controller,
    input_strategy,
    assist,
    *,
    max_steps: int,
    render: bool = False,
) -> EpisodeResult:
    """Run one episode of the composed M3 loop; return the terminal state.

    The single per-tick composition — base command, correction Δ, combine,
    control, step — lives here and nowhere else. Anything that wants a
    different Δ source (NoAssist now, expert/residual later) passes a different
    `assist`; nothing else changes. Deliberately free of console/logging side
    effects so M4 can wrap it for data generation.

    Args:
        environment: a `SimEnv` (anything with reset/step/get_observation).
        controller: the M2 `Controller` (or any object exposing
            `compute(obs, command)` and a `status` property).
        input_strategy: an `InputStrategy` producing the base `Command`.
        assist: an `AssistProvider` producing the correction `Delta`.
        max_steps: episode step budget (one step == one sim/control tick).
        render: when True, sleep one timestep per tick so a `viewer`-mode env
            runs at roughly real time. Leave False for headless/batch.
    """
    observation = environment.reset()
    steps = 0
    for _ in range(max_steps):
        base_command = input_strategy.get_command(observation)
        delta = assist.get_delta(observation, base_command)
        command = apply_delta(base_command, delta)
        controller.compute(observation, command)
        environment.step()
        observation = environment.get_observation()
        steps += 1
        if render:
            time.sleep(SIM_DT)

    return EpisodeResult(
        final_observation=observation,
        lock_status=controller.status,
        n_steps=steps,
    )


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
    args = p.parse_args()

    if not SCENE_PATH.exists():
        print(f"FATAL: scene file not found at {SCENE_PATH}", file=sys.stderr)
        return 2

    render_mode = "headless" if args.headless else "viewer"
    print(f"Loading scene ({render_mode}): {SCENE_PATH}")
    env = SimEnv(str(SCENE_PATH), render_mode=render_mode, seed=args.seed)
    obs = env.reset()
    if not args.headless:
        env.launch_viewer()

    controller = Controller(env)

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
