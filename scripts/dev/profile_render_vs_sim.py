"""Profile where wall-clock goes in a recorded episode, and how arm speed scales
with the controller command clamp.

Answers two questions:

  (A) sim vs render vs encode — for data-gen throughput planning. Phase-1
      data-gen renders nothing, but M7 (vision) will render the wrist camera per
      step, so we want the per-step render cost isolated from the physics cost.

  (B) real-time arm speed vs `max_dpos_per_step` — the controller's per-step
      command clamp is what bounds free-space approach speed; this shows the
      speedup (and whether insertion still succeeds) as it is raised.

Run: uv run python scripts/dev/profile_render_vs_sim.py
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
from PIL import Image

from ai_teleop.control import Controller
from ai_teleop.domain import apply_delta
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.scene import SimEnv
from ai_teleop.sim.scene_source import resolve_scene_path

SCENE = resolve_scene_path()
PANEL = 480
SIM_DT = 0.002


def _build(max_dpos: float = 0.02):
    env = SimEnv(str(SCENE), render_mode="headless", camera_height=PANEL, camera_width=PANEL)
    observation = env.reset()
    controller = Controller(env, max_dpos_per_step=max_dpos)
    target = observation.hole_poses[0][:3].copy()  # task goal: hole_0
    human = ScriptedNoisyHuman(
        np.concatenate([target, controller.home_pose[3:]]),
        position_bias_std=0.012,
        orientation_bias_std=np.deg2rad(4),
        seed=2,
    )
    return env, controller, human, target, observation


def profile_costs(steps: int = 2000) -> None:
    print(f"\n=== (A) cost breakdown over {steps} sim steps ===")
    env, controller, human, target, obs = _build()
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, 0, 0.5]
    cam.distance = 1.6
    cam.azimuth = -40
    cam.elevation = -18
    third = mujoco.Renderer(env.model, height=PANEL, width=PANEL)

    sim_s = third_s = wrist_s = 0.0
    third_frames: list[np.ndarray] = []
    n_render = 0
    for t in range(steps):
        base = human.get_command(obs)
        command = apply_delta(base, Expert().get_delta(obs, base))
        controller.compute(obs, command)
        t0 = time.perf_counter()
        env.step()
        sim_s += time.perf_counter() - t0
        obs = env.get_observation()
        if t % 50 == 0:
            t0 = time.perf_counter()
            third.update_scene(env.data, camera=cam)
            frame = third.render()
            third_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            env.render_wrist_camera()
            wrist_s += time.perf_counter() - t0
            third_frames.append(frame.copy())
            n_render += 1
    third.close()

    imgs = [Image.fromarray(f) for f in third_frames]
    t0 = time.perf_counter()
    imgs[0].save("/tmp/_prof.gif", save_all=True, append_images=imgs[1:], duration=40, loop=0)
    gif_s = time.perf_counter() - t0
    import imageio.v3 as iio

    t0 = time.perf_counter()
    iio.imwrite("/tmp/_prof.mp4", np.stack(third_frames), fps=25)
    mp4_s = time.perf_counter() - t0
    env.close()

    print(
        f"  pure sim        : {sim_s:7.3f} s  ({sim_s / steps * 1e3:6.3f} ms/step)  -> {steps / sim_s:8.0f} steps/s"
    )
    print(
        f"  third-person ren: {third_s:7.3f} s  ({third_s / n_render * 1e3:6.3f} ms/frame, {n_render} frames)"
    )
    print(
        f"  wrist-cam ren   : {wrist_s:7.3f} s  ({wrist_s / n_render * 1e3:6.3f} ms/frame, {n_render} frames)"
    )
    print(f"  GIF encode      : {gif_s:7.3f} s  ({n_render} frames)")
    print(f"  MP4 encode      : {mp4_s:7.3f} s  ({n_render} frames)")
    print(f"  --> a no-render data-gen episode of {steps} steps costs ~{sim_s:.2f}s of physics.")


def profile_speed(steps: int = 6000) -> None:
    print("\n=== (B) approach speed vs max_dpos_per_step (expert, seed 2) ===")
    print(f"  {'clamp(cm)':>10} {'reach<5cm':>12} {'mean v(mm/s)':>14} {'final(mm)':>11}")
    for max_dpos in (0.02, 0.04, 0.06):
        env, controller, human, target, obs = _build(max_dpos)
        reach_step = None
        for t in range(steps):
            base = human.get_command(obs)
            controller.compute(obs, apply_delta(base, Expert().get_delta(obs, base)))
            env.step()
            obs = env.get_observation()
            d = float(np.linalg.norm(obs.peg_pose[:3] - target))
            if reach_step is None and d < 0.05:
                reach_step = t
        final_mm = float(np.linalg.norm(obs.peg_pose[:3] - target)) * 1000
        if reach_step is not None:
            reach_s = reach_step * SIM_DT
            # mean approach speed from start distance to the 5cm mark
            start_d = float(np.linalg.norm(_build(max_dpos)[4].peg_pose[:3] - target))
            v = (start_d - 0.05) / reach_s * 1000
            print(f"  {max_dpos * 100:10.0f} {reach_s:11.2f}s {v:14.1f} {final_mm:11.0f}")
        else:
            print(f"  {max_dpos * 100:10.0f} {'never':>12} {'-':>14} {final_mm:11.0f}")
        env.close()


if __name__ == "__main__":
    profile_costs()
    profile_speed()
