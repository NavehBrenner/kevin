"""Measure the real-time factor of the render=True episode loop (no camera, no GUI).

The teleop gesture timings in VisionInput (recenter hold, clutch/dropout grace,
lock delay) are all in *sim-time* (`observation.sim_time`). If the loop can't keep
sim-time pinned to wall-time, those timings stretch in the real world: at 0.25x
real-time a "3 s" recenter hold becomes 12 wall-seconds of holding still inside a
2 cm tolerance — which feels like "calibration suddenly got impossible".

This runs the real loop with the scripted human (no webcams) and render=True (so it
takes the same `time.sleep(SIM_DT)` path the live teleop does), without launching a
viewer, and reports achieved loop Hz + sim/wall real-time factor.

    cd kevin && .venv/bin/python scripts/dev/loop_rate_probe.py --steps 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.input import ScriptedNoisyHuman  # noqa: E402
from ai_teleop.sim.runner import SIM_DT, run_episode  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402
from ai_teleop.sim.scene_source import resolve_scene_path  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument(
        "--render", action="store_true", default=True, help="take the sleep(SIM_DT) path"
    )
    p.add_argument("--no-render", dest="render", action="store_false")
    args = p.parse_args()

    scene_path = resolve_scene_path(generated=False, wall_seed=7, distractors=None)
    env = SimEnv(str(scene_path), render_mode="headless", seed=0)
    obs = env.reset()
    controller = Controller(env)
    target_pose = np.concatenate([obs.hole_poses[0][:3], controller.home_pose[3:]])  # goal: hole_0
    human = ScriptedNoisyHuman(target_pose, seed=0)

    t0 = time.monotonic()
    result = run_episode(
        env, controller, human, NoAssist(), max_steps=args.steps, render=args.render
    )
    wall = time.monotonic() - t0

    sim_seconds = result.n_steps * SIM_DT
    hz = result.n_steps / wall
    rt = sim_seconds / wall
    print(f"render={args.render}  steps={result.n_steps}")
    print(f"wall={wall:.3f}s  sim={sim_seconds:.3f}s")
    print(f"loop rate    = {hz:7.1f} Hz   (sleep target is {1 / SIM_DT:.0f} Hz)")
    print(f"real-time    = {rt:7.3f}x  ({'slow motion' if rt < 0.95 else 'ok'})")
    print(f"=> a {3.0:.0f}s sim-time recenter hold = {3.0 / rt:.1f} wall-seconds")
    env.close()


if __name__ == "__main__":
    main()
