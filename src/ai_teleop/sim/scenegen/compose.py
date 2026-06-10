"""Compose a runnable MuJoCo scene around a generated wall.

Two modes:

  * ``with_robot=True`` (default) — wires the generated wall into the full
    task scene: vendored Panda (wrist camera/light/F-T), the pre-grasped peg
    welded to the gripper, floor, ambient light, an external camera, and the
    ``home`` keyframe. This is ``full_scene.xml`` with the hand-written wall
    swapped for the generated one — the way generated walls reach the task.
  * ``with_robot=False`` — a minimal wall-only preview (floor, lights,
    front + angled cameras) for quickly eyeballing geometry.

All ``<include>`` targets are written as **absolute** paths so the composed
scene can live anywhere (e.g. beside the wall in ``outputs/``) and still
resolve the Panda/peg/wall meshes — MuJoCo resolves mesh files relative to the
(absolute) directory of each included file.
"""

from __future__ import annotations

from pathlib import Path

# Repo-rooted asset locations (this file is src/ai_teleop/sim/scenegen/compose.py).
_ASSETS = Path(__file__).resolve().parents[4] / "assets" / "mjcf"
PANDA_XML = _ASSETS / "menagerie_panda" / "panda.xml"
PEG_XML = _ASSETS / "peg.xml"

# Peg-to-gripper weld + home keyframe, copied verbatim from full_scene.xml so a
# composed task scene starts in the same valid grasped state.
_WELD_AND_KEYFRAME = """  <equality>
    <weld name="peg_grasp" body1="hand" body2="peg"
          relpose="0 0 0.115 1 0 0 0"
          solref="0.005 1" solimp="0.99 0.999 0.001"/>
  </equality>

  <keyframe>
    <key name="home"
         qpos="0 -1 0 -3 0 3.5 -0.7853 0.04 0.04 0.389841 0 0.560449 0.481968 0.517408 0.517357 0.482015"
         ctrl="0 0 0 0 0 0 0 0"/>
  </keyframe>
"""


def _scene_with_robot(wall_xml: Path) -> str:
    return f"""<mujoco model="generated_task_scene">
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
    <map fogstart="3" fogend="5"/>
  </visual>

  <include file="{PANDA_XML}"/>
  <include file="{wall_xml}"/>
  <include file="{PEG_XML}"/>

  <asset>
    <texture name="grid_tex" type="2d" builtin="checker"
             rgb1="0.85 0.85 0.85" rgb2="0.65 0.65 0.65" width="300" height="300"/>
    <material name="floor_material" texture="grid_tex" texrepeat="6 6"
              reflectance="0.05" specular="0.2" shininess="0.1"/>
  </asset>

  <worldbody>
    <light name="ambient" pos="0 0 2.0" dir="0 0 -1" diffuse="0.4 0.4 0.4"
           specular="0.05 0.05 0.05" castshadow="false"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="floor_material"/>
    <camera name="external_cam" mode="targetbody" target="wall"
            pos="1.4 -1.1 1.0" fovy="50"/>
  </worldbody>

{_WELD_AND_KEYFRAME}</mujoco>
"""


def _scene_wall_only(wall_xml: Path) -> str:
    return f"""<mujoco model="generated_wall_preview">
  <option gravity="0 0 -9.81"/>
  <visual><global offwidth="1024" offheight="768"/><quality shadowsize="2048"/></visual>
  <include file="{wall_xml}"/>
  <worldbody>
    <light name="preview_key" pos="0.2 -0.6 1.4" dir="0.3 0.6 -1" diffuse="0.8 0.8 0.8"/>
    <light name="preview_fill" pos="0 0 2" dir="0 0 -1" diffuse="0.4 0.4 0.4" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.05" rgba="0.6 0.6 0.65 1"/>
    <camera name="front" pos="-0.10 0 0.45" xyaxes="0 -1 0 0 0 1"/>
    <camera name="angled" mode="targetbody" target="wall" pos="0.10 -0.55 0.85"/>
  </worldbody>
</mujoco>
"""


def compose_scene(
    wall_xml: str | Path, out_path: str | Path | None = None, *, with_robot: bool = True
) -> Path:
    """Write a runnable scene around ``wall_xml`` and return its path.

    Defaults to writing ``scene_task.xml`` / ``scene_wall.xml`` beside the wall.
    """
    wall_xml = Path(wall_xml).resolve(strict=True)
    xml = _scene_with_robot(wall_xml) if with_robot else _scene_wall_only(wall_xml)
    if out_path is None:
        name = "scene_task.xml" if with_robot else "scene_wall.xml"
        out_path = wall_xml.parent / name
    out_path = Path(out_path)
    out_path.write_text(xml)
    return out_path
