"""Spot-check: does the expert improve seating vs NoAssist under a biased human?

Paired run (same seed ⇒ identical operator) of NoAssist and the Expert through
the M3 seam. Reports the final tip→hole lateral error and axial penetration.
"""

from pathlib import Path

import numpy as np

from ai_teleop.common import Observation
from ai_teleop.common.utils.rotations import quat_to_matrix
from ai_teleop.control import Controller
from ai_teleop.domain import NoAssist, apply_delta
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.scene import SimEnv

SCENE = Path(__file__).resolve().parents[2] / "assets" / "mjcf" / "full_scene.xml"
N_AXIS = np.array([1.0, 0.0, 0.0])


def peg_tip(o: Observation):
    R = quat_to_matrix(o.peg_pose[3:])
    return o.peg_pose[:3] + R @ np.array([0, 0, 0.030]), R[:, 2]


def run(assist, seed=2, steps=6000, bias=0.012):
    env = SimEnv(str(SCENE), render_mode="headless")
    obs = env.reset()
    controller = Controller(env)
    target = obs.hole_poses[obs.target_hole_index][:3].copy()
    home_quat = controller.home_pose[3:]
    human = ScriptedNoisyHuman(
        np.concatenate([target, home_quat]),
        position_bias_std=bias,
        orientation_bias_std=np.deg2rad(4),
        seed=seed,
    )
    for _ in range(steps):
        base = human.get_command(obs)
        cmd = apply_delta(base, assist.get_delta(obs, base))
        controller.compute(obs, cmd)
        env.step()
        obs = env.get_observation()
    tip, axis = peg_tip(obs)
    e = target - tip
    e_ax = float(e @ N_AXIS)
    e_lat = float(np.linalg.norm(e - e_ax * N_AXIS))
    ang = float(np.degrees(np.arccos(np.clip(axis @ N_AXIS, -1, 1))))
    penetration = -e_ax  # tip past the entry plane along +x
    return e_lat, penetration, ang


for label, assist in (("NoAssist", NoAssist()), ("Expert", Expert())):
    e_lat, pen, ang = run(assist)
    print(
        f"{label:10s}  lateral_err={e_lat * 1000:6.1f} mm  penetration={pen * 1000:6.1f} mm  "
        f"axis_misalign={ang:5.1f} deg"
    )
