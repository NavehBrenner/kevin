"""Probe: how deep does the peg actually get in an eval trial, per operator scale?

Why 0% human-only even at near-zero operator error? Track the max penetration /
min lateral error reached so we can see whether the peg approaches the 15 mm seat
threshold at all, or stalls far short.

    uv run python scripts/dev/probe_seating.py
"""

from __future__ import annotations

import numpy as np

from ai_teleop.common.seating import SeatingGeometry
from ai_teleop.control import Controller
from ai_teleop.eval.ablation import _human_seed
from ai_teleop.input.scripted_noisy_human import (
    DEFAULT_DRIFT_POSITION_STD,
    DEFAULT_POSITION_BIAS_STD,
    ScriptedNoisyHuman,
)
from ai_teleop.sim.config import EnvConfig, episode_wall_seed
from ai_teleop.sim.env_setup import make_env
from ai_teleop.sim.runner import run_episode

_TARGET_HOLE_INDEX = 0  # data-gen places the goal at hole_0


def probe(
    scale: float, *, use_expert: bool = False, episode_index: int = 0, master_seed: int = 0
) -> None:
    environment = make_env(
        EnvConfig(wall_seed=episode_wall_seed(master_seed, episode_index)), render_mode="headless"
    )
    try:
        controller = Controller(environment)
        observation = environment.reset()
        target_position = observation.hole_poses[_TARGET_HOLE_INDEX][:3].copy()
        home_quaternion = controller.home_pose[3:]
        target_pose = np.concatenate([target_position, home_quaternion])
        human = ScriptedNoisyHuman(
            target_pose,
            position_bias_std=DEFAULT_POSITION_BIAS_STD * scale,
            drift_position_std=DEFAULT_DRIFT_POSITION_STD * scale,
            seed=_human_seed(master_seed, episode_index),
        )

        best = {"pen": -1e9, "lat_at_best": None, "min_lat": 1e9}

        def step_callback(step, obs, base_command, delta, command) -> bool:
            geom = SeatingGeometry.from_observation(obs, _TARGET_HOLE_INDEX)
            if geom.penetration > best["pen"]:
                best["pen"] = geom.penetration
                best["lat_at_best"] = geom.lateral_error
            best["min_lat"] = min(best["min_lat"], geom.lateral_error)
            return False

        from ai_teleop.domain import NoAssist
        from ai_teleop.expert import Expert

        assist = Expert(target_hole_index=_TARGET_HOLE_INDEX) if use_expert else NoAssist()
        run_episode(
            environment, controller, human, assist, max_steps=6000, step_callback=step_callback
        )
        tag = "EXPERT  " if use_expert else "NoAssist"
        print(
            f"{tag} scale={scale:>5}  max_penetration={best['pen'] * 1000:7.2f} mm  "
            f"lateral_at_max={best['lat_at_best'] * 1000:6.2f} mm  "
            f"min_lateral={best['min_lat'] * 1000:6.2f} mm"
        )
    finally:
        environment.close()


if __name__ == "__main__":
    for s in (0.0, 0.2, 1.0):
        probe(s, use_expert=False)
    for s in (0.0, 0.2, 1.0):
        probe(s, use_expert=True)
