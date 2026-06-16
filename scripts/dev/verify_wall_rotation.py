"""Verify wall tilt sampling + the generate-wall disk cache, and render a frame."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from ai_teleop.sim.scenegen.compose import compose_scene
from ai_teleop.sim.scenegen.generate import generate_wall

OUT = Path("outputs/walls/_rotation_check")

for seed in (7, 11, 19):
    Path(f"outputs/walls/wall_{seed}").exists()
    t0 = time.perf_counter()
    scene = generate_wall(seed=seed, distractors=2)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    generate_wall(seed=seed, distractors=2)  # second call: should hit the cache
    cache_s = time.perf_counter() - t0
    deg = np.rad2deg(scene.spec.orientation)
    print(
        f"seed {seed:3d}: tilt(deg)=[{deg[0]:6.1f} {deg[1]:6.1f} {deg[2]:6.1f}]  "
        f"build={build_s:5.2f}s  re-call(cached)={cache_s:5.3f}s"
    )

# Render the most-tilted of them so the rotation is visible.
scene = generate_wall(seed=11, distractors=2)
scene_path = compose_scene(Path(scene.mjcf_path), with_robot=True)
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="external_cam")
OUT.mkdir(parents=True, exist_ok=True)
Image.fromarray(renderer.render()).save(OUT / "tilted_wall.png")
renderer.close()
print(f"rendered {OUT / 'tilted_wall.png'}")
