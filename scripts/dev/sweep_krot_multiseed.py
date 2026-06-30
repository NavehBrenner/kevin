"""Multi-seed success rate for raised rotational stiffness on moderate tilts.

Compares the current K_rot=3 vs a candidate K_rot=10 (with matched damping),
over several human seeds, on upright / 8deg / 12deg walls with a bore-aware
human. Single-seed runs were too noisy to trust; this counts seated/total.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ai_teleop.common.utils.rotations import axis_from_quat
from ai_teleop.control import Controller
from ai_teleop.domain import apply_delta
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman, bore_aligned_grasp
from ai_teleop.sim.scene import SimEnv
from ai_teleop.sim.scenegen.compose import compose_scene
from ai_teleop.sim.scenegen.generate import generate_from_spec
from ai_teleop.sim.scenegen.sampler import sample_wall_spec

STATIC = Path("assets/mjcf/full_scene.xml")
SEEDS = range(1, 7)


def seated(scene_path: Path, k_rot: float, target_idx: int, seed: int) -> bool:
    d_rot = 4.0 * (k_rot / 3.0) ** 0.5
    env = SimEnv(str(scene_path), render_mode="headless")
    obs = env.reset()
    controller = Controller(
        env,
        stiffness_tcp=np.array([400.0, 400.0, 500.0, k_rot, k_rot, k_rot]),
        damping_tcp=np.array([80.0, 80.0, 89.0, d_rot, d_rot, d_rot]),
    )
    hole = obs.hole_poses[target_idx]
    bore = axis_from_quat(hole[3:], 0)
    grasp = bore_aligned_grasp(controller.home_pose[3:], bore)
    human = ScriptedNoisyHuman(
        np.concatenate([hole[:3], grasp]),
        position_bias_std=0.012,
        orientation_bias_std=np.deg2rad(4),
        seed=seed,
    )
    ever = False
    for _ in range(6000):
        base = human.get_command(obs)
        controller.compute(
            obs, apply_delta(base, Expert(target_hole_index=target_idx).get_delta(obs, base))
        )
        env.step()
        obs = env.get_observation()
        tip = obs.peg_pose[:3] + 0.030 * axis_from_quat(obs.peg_pose[3:], 2)
        err = hole[:3] - tip
        pen = -float(err @ bore) * 1000
        lat = float(np.linalg.norm(err - (err @ bore) * bore)) * 1000
        if pen >= 15 and lat < 6:
            ever = True
            break
    env.close()
    return ever


def tilted(tilt_deg: float) -> Path:
    spec = sample_wall_spec(
        seed=3,
        true_hole={"pos": (0.0, 0.0), "size": {"diameter": 0.012}, "chamfer": 0.004},
        distractors=0,
    )
    spec = replace(spec, orientation=(0.0, np.deg2rad(tilt_deg), 0.0))
    scene = generate_from_spec(spec, Path(f"outputs/walls/_ms_{int(tilt_deg)}"))
    return compose_scene(Path(scene.mjcf_path), with_robot=True)


scenes = [("upright", STATIC, 1), ("tilt-8", tilted(8), 0), ("tilt-12", tilted(12), 0)]
print(f"{'K_rot':>6}   " + "".join(f"{name:>12}" for name, _, _ in scenes))
for k_rot in (3.0, 10.0):
    cells = []
    for _name, scene, idx in scenes:
        n = sum(seated(scene, k_rot, idx, s) for s in SEEDS)
        cells.append(f"{n}/{len(SEEDS)}")
    print(f"{k_rot:6.1f}   " + "".join(f"{c:>12}" for c in cells))
