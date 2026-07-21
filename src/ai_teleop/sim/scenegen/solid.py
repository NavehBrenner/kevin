"""Build the parametric wall-with-holes solid in CadQuery, and tessellate it to
metre-scale triangle meshes.

CadQuery / OpenCASCADE is built in **millimetres** here: OCCT's tessellation
deflection tolerances are absolute, so meshing curved features (the round bore,
the chamfer cone) is far cleaner at mm scale than at metre scale. The mm->metre
conversion happens exactly once, at tessellation time, so every mesh written to
disk and every coordinate outside this module is in metres.

Frame: built on the ``YZ`` workplane so the plate thickness runs along world X
and the face spans world (Y, Z) — see ``config`` for the full convention. The
robot-facing surface is the ``<X`` face; offset holes are drilled there. Note
the ``<X`` face's local +x axis points along world **-Y**, so a hole at world
``(y, z)`` is pushed at ``(-y, z)`` (verified in
``scripts/dev/probe_cadquery_hole_frame.py``).
"""

from __future__ import annotations

from typing import cast

import cadquery as cq

from .config import HoleSpec, WallSpec

MM_PER_M = 1000.0
M_PER_MM = 0.001


def _drill_circle(workpiece: cq.Workplane, hole: HoleSpec) -> cq.Workplane:
    """Cut a round through-bore for ``hole`` on the robot-facing face (mm)."""
    y_mm, z_mm = hole.pos[0] * MM_PER_M, hole.pos[1] * MM_PER_M
    diameter_mm = hole.size["diameter"] * MM_PER_M
    # Local +x on the <X face is world -Y, so negate y to land at world (y, z).
    return workpiece.faces("<X").workplane().pushPoints([(-y_mm, z_mm)]).hole(diameter_mm)


def _chamfer_hole(workpiece: cq.Workplane, hole: HoleSpec, thickness_mm: float) -> cq.Workplane:
    """Chamfer just this hole's robot-facing rim, by its own width.

    Selects the single rim edge nearest the hole centre on the ``<X`` face (the
    front circular edge), so per-hole chamfer widths are honoured. The hole
    centre in CadQuery world coords is ``(-thickness/2, y, z)`` (the drill maps
    pushed ``(-y, z)`` back to world ``(y, z)`` — see ``_drill_circle``).
    """
    if hole.chamfer <= 0:
        return workpiece
    y_mm, z_mm = hole.pos[0] * MM_PER_M, hole.pos[1] * MM_PER_M
    front_x = -0.5 * thickness_mm
    rim = cq.NearestToPointSelector((front_x, y_mm, z_mm))
    return workpiece.faces("<X").edges(rim).chamfer(hole.chamfer * MM_PER_M)


def build_wall_solid(spec: WallSpec) -> cq.Workplane:
    """Return the wall solid (with every hole cut and its rim chamfered) in mm."""
    thickness, width, height = (v * MM_PER_M for v in spec.wall_size)
    workpiece = cq.Workplane("YZ").box(width, height, thickness)

    # Cut + chamfer each hole individually so per-hole chamfer widths are honoured.
    for hole in spec.holes:
        workpiece = _drill_circle(workpiece, hole)
        workpiece = _chamfer_hole(workpiece, hole, thickness)

    return workpiece


def tessellate_to_metres(
    workpiece: cq.Workplane, tolerance_mm: float = 0.2
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Tessellate the solid and return ``(vertices_m, triangles)``.

    Vertices are converted mm -> metres. Triangles index into ``vertices_m``.
    """
    # .val() is typed as a union (Vector|Location|Shape|Sketch); a built solid
    # is always a Shape, which is what carries .tessellate().
    verts_mm, tris = cast(cq.Shape, workpiece.val()).tessellate(tolerance=tolerance_mm)
    vertices_m = [(v.x * M_PER_MM, v.y * M_PER_MM, v.z * M_PER_MM) for v in verts_mm]
    return vertices_m, tris
