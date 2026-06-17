"""Does the system actually correct orientation? Measure peg-axis vs bore-axis
misalignment over an episode, on the upright static wall AND a tilted wall.

The scripted human commands a FIXED home orientation (+ small bias/drift); the
expert's angular-alignment term is the only thing that reorients the peg. This
quantifies how much each does, and whether the expert copes when a tilted wall
puts the bore well off the home orientation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_teleop.common.utils.rotations import axis_from_quat
from ai_teleop.control import Controller
from ai_teleop.domain import NoAssist, apply_delta
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.scene import SimEnv
from ai_teleop.sim.scenegen.compose import compose_scene
from ai_teleop.sim.scenegen.generate import generate_wall

STATIC = Path("assets/mjcf/full_scene.xml")


def run(scene_path, assist, seed=2, steps=6000):
    env = SimEnv(str(scene_path), render_mode="headless")
    o = env.reset()
    c = Controller(env)
    hole = o.hole_poses[o.target_hole_index]
    bore = axis_from_quat(hole[3:], 0)  # hole local +x = bore
    human = ScriptedNoisyHuman(
        np.concatenate([hole[:3], c.home_pose[3:]]),
        position_bias_std=0.012,
        orientation_bias_std=np.deg2rad(4),
        seed=seed,
    )
    peg0 = axis_from_quat(o.peg_pose[3:], 2)
    start = np.degrees(np.arccos(np.clip(peg0 @ bore, -1, 1)))
    for _ in range(steps):
        b = human.get_command(o)
        c.compute(o, apply_delta(b, assist.get_delta(o, b)))
        env.step()
        o = env.get_observation()
    pegN = axis_from_quat(o.peg_pose[3:], 2)
    end = np.degrees(np.arccos(np.clip(pegN @ bore, -1, 1)))
    err = hole[:3] - (o.peg_pose[:3] + 0.030 * pegN)
    pen = -float(err @ bore) * 1000
    env.close()
    return start, end, pen


# tilted wall (seed 19: bore tilted ~21 deg off +x)
tilted = generate_wall(seed=19, distractors=2)
tilted_scene = compose_scene(Path(tilted.mjcf_path), with_robot=True)
print(f"tilted wall seed 19 tilt(deg) = {np.round(np.rad2deg(tilted.spec.orientation), 1)}")
print(f"{'scene':14}{'assist':10}{'misalign start->end':22}{'penetration(mm)':>16}")
for name, scene in (("upright", STATIC), ("tilted-21deg", tilted_scene)):
    for label, a in (("NoAssist", NoAssist()), ("Expert", Expert())):
        s, e, pen = run(scene, a)
        print(f"{name:14}{label:10}{s:6.1f} -> {e:5.1f} deg{'':6}{pen:12.1f}")
