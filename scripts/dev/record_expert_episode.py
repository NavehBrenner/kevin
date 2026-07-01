"""Record an expert-driven insertion episode to an animated GIF.

Runs the M4 stack (ScriptedNoisyHuman -> analytical Expert -> seam -> controller
-> sim) and renders two synchronized views per decimated frame:

  * left  - third-person free camera framing the arm + wall
  * right - the wrist camera (exactly what the M5/M7 policy will be fed)

No ffmpeg needed: frames are stitched into a GIF via Pillow.

Run: uv run python scripts/dev/record_expert_episode.py [--seed N] [--steps N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from ai_teleop.control import Controller
from ai_teleop.domain import apply_delta
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.scene import SimEnv

SCENE = Path(__file__).resolve().parents[2] / "assets" / "mjcf" / "full_scene.xml"
OUT = Path(__file__).resolve().parents[2] / "outputs" / "expert_episode.gif"

H = W = 480


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--every", type=int, default=50, help="render 1 frame per N sim steps")
    ap.add_argument("--bias", type=float, default=0.012)
    args = ap.parse_args()

    env = SimEnv(str(SCENE), render_mode="headless", camera_height=H, camera_width=W)
    obs = env.reset()
    controller = Controller(env)
    target = obs.hole_poses[0][:3].copy()  # task goal: hole_0
    human = ScriptedNoisyHuman(
        np.concatenate([target, controller.home_pose[3:]]),
        position_bias_std=args.bias,
        orientation_bias_std=np.deg2rad(4),
        seed=args.seed,
    )

    scene_cam = mujoco.MjvCamera()
    scene_cam.lookat[:] = [0.55, 0.0, 0.5]
    scene_cam.distance = 1.4
    scene_cam.azimuth = 140
    scene_cam.elevation = -18
    third = mujoco.Renderer(env.model, height=H, width=W)

    frames: list[Image.Image] = []
    for t in range(args.steps):
        base = human.get_command(obs)
        cmd = apply_delta(base, Expert().get_delta(obs, base))
        controller.compute(obs, cmd)
        env.step()
        obs = env.get_observation()
        if t % args.every == 0:
            third.update_scene(env.data, camera=scene_cam)
            left = third.render()
            right = env.render_wrist_camera()
            right = np.asarray(Image.fromarray(right).resize((H, H), Image.NEAREST))
            canvas = np.concatenate([left, right], axis=1)
            frames.append(Image.fromarray(canvas))
    third.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=40, loop=0)
    d = float(np.linalg.norm(obs.peg_pose[:3] - target))
    print(f"wrote {OUT}  ({len(frames)} frames)  final peg->hole = {d * 1000:.0f} mm")


if __name__ == "__main__":
    main()
