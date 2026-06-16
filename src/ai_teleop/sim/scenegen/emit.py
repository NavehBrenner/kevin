"""Emit a MuJoCo MJCF wall file (and its OBJ meshes) from a resolved WallSpec.

Mirrors the structure of the hand-written ``wall_with_holes.xml`` it replaces:
a single ``wall`` body welded to the world at ``(0.80, 0, 0.45)``, with the
mesh assets expressed in wall-centre metre coordinates (so the geoms need no
offset). One collision geom per CoACD part (group 3, collides with the peg) and
one non-colliding visual mesh (group 2). A ``hole_i`` site marks each hole
centre on the robot-facing surface (body-local ``x = -thickness/2``); SimEnv
reads these for the privileged hole-pose observation.

Meshes are written as bare-filename OBJs alongside the XML, so the file loads
standalone (default meshdir = the XML's own directory) and carries no
``<compiler>`` directive that could clash when ``<include>``-d into a scene.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from .config import WallSpec

# World anchor of the wall body — unchanged from the hand-written wall so the
# Panda home keyframe and camera framing stay valid.
WALL_BODY_POS = (0.80, 0.0, 0.45)


def _mesh_assets(collision_names: list[str], visual_name: str) -> str:
    lines = [f'    <mesh name="{visual_name}" file="{visual_name}.obj"/>']
    lines += [f'    <mesh name="{name}" file="{name}.obj"/>' for name in collision_names]
    return "\n".join(lines)


def _collision_geoms(collision_names: list[str]) -> str:
    return "\n".join(
        f'      <geom name="{name}_geom" class="wall_solid" type="mesh" mesh="{name}"/>'
        for name in collision_names
    )


def _hole_sites(spec: WallSpec) -> str:
    surface_x = -0.5 * spec.wall_size[0]
    lines = []
    for index, hole in enumerate(spec.holes):
        y, z = hole.pos
        rgba = "0 1 0 0.5" if hole.is_target else "1 1 0 0.4"
        lines.append(
            f'      <site name="hole_{index}" pos="{surface_x:.5f} {y:.5f} {z:.5f}" '
            f'size="0.003" rgba="{rgba}" group="3"/>'
        )
    return "\n".join(lines)


def write_meshes(
    out_dir: Path,
    visual_mesh: trimesh.Trimesh,
    collision_parts: list[trimesh.Trimesh],
) -> tuple[str, list[str]]:
    """Write the visual and collision OBJs; return their (visual, [collision]) names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_name = "wall_visual"
    visual_mesh.export(out_dir / f"{visual_name}.obj")

    collision_names = []
    for index, part in enumerate(collision_parts):
        name = f"wall_col_{index:03d}"
        part.export(out_dir / f"{name}.obj")
        collision_names.append(name)
    return visual_name, collision_names


def _orientation_quat_attr(spec: WallSpec) -> str:
    """MJCF ``quat="w x y z"`` attribute for the wall tilt, or "" when upright.

    Emitted as a quaternion (not ``euler``) so the standalone wall.xml needs no
    ``<compiler angle>`` directive to be interpreted correctly.
    """
    rx, ry, rz = spec.orientation
    if rx == 0.0 and ry == 0.0 and rz == 0.0:
        return ""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_euler("xyz", [rx, ry, rz]).as_quat()
    return f' quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}"'


def build_mjcf(spec: WallSpec, visual_name: str, collision_names: list[str]) -> str:
    """Return the MJCF XML string for the generated wall."""
    px, py, pz = WALL_BODY_POS
    return f"""<mujoco model="wall_generated">
  <!--
    Procedurally generated wall (see ai_teleop.sim.scenegen). Geometry lives in
    OBJ meshes expressed in wall-centre metre coordinates. Collision = {len(collision_names)}
    convex CoACD parts; visual = one non-colliding mesh. Holes: {len(spec.holes)}
    (hole_0 is the target). Seed {spec.seed}. See header.json for the full spec.
  -->

  <asset>
    <material name="wall_material" rgba="0.78 0.78 0.80 1" specular="0.2" shininess="0.2"/>
{_mesh_assets(collision_names, visual_name)}
  </asset>

  <default>
    <default class="wall_solid">
      <geom type="mesh" material="wall_material" condim="3"
            friction="0.8 0.005 0.0001" group="3"/>
    </default>
    <default class="wall_visual">
      <geom type="mesh" material="wall_material"
            contype="0" conaffinity="0" group="2"/>
    </default>
  </default>

  <worldbody>
    <body name="wall" pos="{px} {py} {pz}"{_orientation_quat_attr(spec)}>
      <geom name="wall_visual_geom" class="wall_visual" mesh="{visual_name}"/>
{_collision_geoms(collision_names)}
{_hole_sites(spec)}
    </body>
  </worldbody>
</mujoco>
"""


def write_mjcf(out_dir: Path, spec: WallSpec, visual_name: str, collision_names: list[str]) -> Path:
    """Write ``wall.xml`` into ``out_dir`` and return its path."""
    xml = build_mjcf(spec, visual_name, collision_names)
    path = out_dir / "wall.xml"
    path.write_text(xml)
    return path
