"""Render generated walls to PNGs so the geometry (holes + chamfer funnel) can
be eyeballed. Wraps each generated wall.xml in a minimal scene (floor, light,
front + angled cameras) and renders offscreen.

Run: uv run python scripts/dev/manual_test_render.py
"""

import time
from pathlib import Path

import mujoco
from PIL import Image

from ai_teleop.sim.scenegen.generate import generate_wall

SCENE_TEMPLATE = """<mujoco model="wall_preview">
  <option gravity="0 0 -9.81"/>
  <visual><global offwidth="1024" offheight="768"/><quality shadowsize="2048"/></visual>
  <worldbody>
    <light name="key" pos="0.2 -0.6 1.4" dir="0.3 0.6 -1" diffuse="0.8 0.8 0.8"/>
    <light name="fill" pos="0 0 2" dir="0 0 -1" diffuse="0.4 0.4 0.4" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.05" rgba="0.6 0.6 0.65 1"/>
    <camera name="front" pos="-0.10 0 0.45" xyaxes="0 -1 0 0 0 1"/>
    <camera name="angled" mode="targetbody" target="wall" pos="0.10 -0.55 0.85"/>
  </worldbody>
  <include file="wall.xml"/>
</mujoco>
"""


def render_wall(out_dir: Path) -> None:
    scene_path = out_dir / "_preview_scene.xml"
    scene_path.write_text(SCENE_TEMPLATE)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=768, width=1024)
    for cam in ("front", "angled"):
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        path = out_dir / f"preview_{cam}.png"
        Image.fromarray(rgb).save(path)
        print(f"  wrote {path}")
    renderer.close()


CASES = [
    (
        "three_clean",
        dict(
            seed=11,
            distractors=2,
            true_hole={"pos": (0.0, 0.0), "size": {"diameter": 0.014}, "chamfer": 0.003},
        ),
    ),
    ("dense_random", dict(seed=7)),
    (
        "big_chamfer",
        dict(
            seed=5,
            distractors=1,
            true_hole={"pos": (0.0, 0.0), "size": {"diameter": 0.012}, "chamfer": 0.006},
        ),
    ),
]

for name, kwargs in CASES:
    out_dir = Path("outputs/walls/_preview") / name
    t0 = time.perf_counter()
    scene = generate_wall(out_dir=out_dir, **kwargs)
    dt = time.perf_counter() - t0
    print(
        f"\n[{name}] holes={len(scene.spec.holes)} parts={len(scene.collision_mesh_paths)} "
        f"gen={dt * 1000:.0f}ms"
    )
    render_wall(out_dir)
