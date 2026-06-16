"""Does a bore-aware scripted human make a tilted wall seatable?

Compares the fixed-upright human (current) vs a bore-aimed human (commands the
tilted bore + noise) on a 21deg-tilted wall, with the Expert assisting.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ai_teleop.control import Controller
from ai_teleop.domain import apply_delta
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman, bore_aligned_grasp
from ai_teleop.sim.scene import SimEnv
from ai_teleop.sim.scenegen.compose import compose_scene
from ai_teleop.sim.scenegen.generate import generate_wall

STATIC = Path("assets/mjcf/full_scene.xml")


def axis(q, c):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)[:, c]


def run(scene_path, bore_aware, seed=2, steps=6000):
    env = SimEnv(str(scene_path), render_mode="headless")
    o = env.reset()
    c = Controller(env)
    hole = o.hole_poses[o.target_hole_index]
    bore = axis(hole[3:], 0)
    grasp = bore_aligned_grasp(c.home_pose[3:], bore) if bore_aware else c.home_pose[3:]
    human = ScriptedNoisyHuman(
        np.concatenate([hole[:3], grasp]),
        position_bias_std=0.012,
        orientation_bias_std=np.deg2rad(4),
        seed=seed,
    )
    start = np.degrees(np.arccos(np.clip(axis(o.peg_pose[3:], 2) @ bore, -1, 1)))
    for _ in range(steps):
        b = human.get_command(o)
        c.compute(o, apply_delta(b, Expert().get_delta(o, b)))
        env.step()
        o = env.get_observation()
    pegN = axis(o.peg_pose[3:], 2)
    end = np.degrees(np.arccos(np.clip(pegN @ bore, -1, 1)))
    pen = -float((hole[:3] - (o.peg_pose[:3] + 0.030 * pegN)) @ bore) * 1000
    env.close()
    return start, end, pen


tilted = generate_wall(seed=19, distractors=2)
tscene = compose_scene(Path(tilted.mjcf_path), with_robot=True)
print(f"tilted wall seed 19 tilt(deg) = {np.round(np.rad2deg(tilted.spec.orientation), 1)}")
print(f"{'scene':14}{'human':16}{'misalign s->e':18}{'penetration(mm)':>16}")
for name, scene in (("upright", STATIC), ("tilted-21deg", tscene)):
    for label, ba in (("fixed-upright", False), ("bore-aware", True)):
        s, e, pen = run(scene, ba)
        seat = "  <- SEATS" if pen >= 15 else ""
        print(f"{name:14}{label:16}{s:5.1f} -> {e:4.1f} deg{'':4}{pen:12.1f}{seat}")
