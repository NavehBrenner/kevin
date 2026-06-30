"""Tune rotational stiffness so the backbone can actively reorient moderate
bore tilts without re-triggering the position limit cycle (K_rot=25 was
unstable; K_rot=3 is the current stable-but-floppy default).

For each K_rot (with damping scaled to hold the damping ratio) we measure:
  * upright wall  — seats cleanly? (stability / no limit cycle proxy)
  * 10deg & 15deg tilted walls, bore-aware human — does it reorient and seat?
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
POS_K = [400.0, 400.0, 500.0]
POS_D = [80.0, 80.0, 89.0]


def run(
    scene_path: Path, k_rot: float, d_rot: float, target_idx: int, bore_aware: bool
) -> tuple[float, float]:
    env = SimEnv(str(scene_path), render_mode="headless")
    obs = env.reset()
    controller = Controller(
        env,
        stiffness_tcp=np.array(POS_K + [k_rot] * 3),
        damping_tcp=np.array(POS_D + [d_rot] * 3),
    )
    hole = obs.hole_poses[target_idx]
    bore = axis_from_quat(hole[3:], 0)
    grasp = (
        bore_aligned_grasp(controller.home_pose[3:], bore)
        if bore_aware
        else controller.home_pose[3:]
    )
    human = ScriptedNoisyHuman(
        np.concatenate([hole[:3], grasp]),
        position_bias_std=0.012,
        orientation_bias_std=np.deg2rad(4),
        seed=2,
    )
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
    env.close()
    return pen, lat


def tilted_scene(tilt_deg: float) -> Path:
    spec = sample_wall_spec(
        seed=3,
        true_hole={"pos": (0.0, 0.0), "size": {"diameter": 0.012}, "chamfer": 0.004},
        distractors=0,
    )
    spec = replace(spec, orientation=(0.0, np.deg2rad(tilt_deg), 0.0))
    scene = generate_from_spec(spec, Path(f"outputs/walls/_rk_{int(tilt_deg)}"))
    return compose_scene(Path(scene.mjcf_path), with_robot=True)


scenes = [
    ("upright", STATIC, 1),
    ("tilt-10", tilted_scene(10), 0),
    ("tilt-15", tilted_scene(15), 0),
]
print(f"{'K_rot':>6}{'D_rot':>7}   " + "".join(f"{name + ' pen/lat':>20}" for name, _, _ in scenes))
for k_rot in (3.0, 6.0, 10.0, 15.0):
    d_rot = 4.0 * (k_rot / 3.0) ** 0.5
    cells = []
    for _name, scene, idx in scenes:
        pen, lat = run(scene, k_rot, d_rot, idx, bore_aware=True)
        flag = "✓" if pen >= 15 and lat < 6 else " "
        cells.append(f"{pen:6.1f}/{lat:4.1f}{flag:>3}")
    print(f"{k_rot:6.1f}{d_rot:7.1f}   " + "".join(f"{c:>20}" for c in cells))
