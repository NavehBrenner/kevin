"""Build the wall's convex collision parts analytically (no voxel decomposer).

The wall is a prism, so its collision geometry is exact and cheap:

  * **Bulk** — the face minus each hole's *rim* (outer) polygon, extruded
    through the full thickness. Convex 2D pieces (from ``partition2d``) become
    convex prisms.
  * **Back-ring wedges** — per hole, per facet: fill the annulus between the
    bore and the rim over the *straight* depth (rim plane .. back face), so the
    actual bore narrows to the peg-fit diameter behind the chamfer.
  * **Collar wedges** — per hole, per facet: the chamfer funnel itself, sloping
    from the bore at the chamfer depth out to the rim at the robot-facing face.
    This is the passive-alignment ramp (the project's difficulty knob).

Every part is returned as the convex hull of its corner points; MuJoCo also
hulls mesh collision geoms, so giving it already-convex corner sets is exact.
All coordinates are wall-centre metres with x = thickness (matching the CadQuery
visual mesh), so collision and visual geoms need no per-geom offset.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from .config import COLLISION_FACETS, WallSpec
from .partition2d import convex_pieces
from .shapes2d import hole_rings, outer_outline

Vertices = list[tuple[float, float, float]]
Triangles = list[tuple[int, int, int]]


def to_trimesh(vertices: Vertices, triangles: Triangles) -> trimesh.Trimesh:
    """Wrap raw ``(vertices, triangles)`` as a single Trimesh (the visual mesh)."""
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(triangles, dtype=np.int64),
        process=False,
    )


def _hull(points: np.ndarray) -> trimesh.Trimesh:
    """Convex hull of a corner-point set, as a Trimesh collision part.

    Uses scipy directly (not ``Trimesh.convex_hull``, which drags in networkx
    for normal-fixing we don't need — MuJoCo re-hulls mesh collision geoms, so
    face winding is irrelevant here).
    """
    points = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(points)
    return trimesh.Trimesh(vertices=points, faces=hull.simplices, process=False)


def _prism(polygon_yz: np.ndarray, x0: float, x1: float) -> trimesh.Trimesh:
    """Extrude a convex 2D (y, z) polygon between thickness planes x0 and x1."""
    front = np.column_stack([np.full(len(polygon_yz), x0), polygon_yz])
    back = np.column_stack([np.full(len(polygon_yz), x1), polygon_yz])
    return _hull(np.vstack([front, back]))


def _ring_wedges(
    inner: np.ndarray,
    outer: np.ndarray,
    *,
    bore_x: tuple[float, float],
    rim_x: tuple[float, float],
) -> list[trimesh.Trimesh]:
    """Per-facet wedges filling between the ``inner`` and ``outer`` rings.

    ``bore_x`` = (x at inner-bottom, x at inner-top); ``rim_x`` likewise for the
    outer ring. A back-ring is a straight box (bore_x == rim_x spans). A collar
    slopes: the inner ring sits deeper (chamfer plane) than the rim's front lip.
    Degenerate facets (zero-area, e.g. a collar with no straight section) are
    skipped — the hull would be flat.
    """
    facets = len(inner)
    wedges: list[trimesh.Trimesh] = []
    for j in range(facets):
        k = (j + 1) % facets
        corners = np.array([
            [bore_x[0], *inner[j]],
            [bore_x[0], *inner[k]],
            [bore_x[1], *inner[j]],
            [bore_x[1], *inner[k]],
            [rim_x[0], *outer[j]],
            [rim_x[0], *outer[k]],
            [rim_x[1], *outer[j]],
            [rim_x[1], *outer[k]],
        ])
        # Drop duplicate rows (collapsed faces) before hulling.
        corners = np.unique(corners, axis=0)
        if len(corners) < 4:
            continue
        wedges.append(_hull(corners))
    return wedges


def wall_collision_parts(spec: WallSpec, facets: int = COLLISION_FACETS) -> list[trimesh.Trimesh]:
    """Return the full list of convex collision parts for ``spec``."""
    thickness = spec.wall_size[0]
    front_x = -0.5 * thickness  # robot-facing face
    back_x = +0.5 * thickness

    rim_outlines = [outer_outline(hole, facets) for hole in spec.holes]
    parts = [
        _prism(piece, front_x, back_x)
        for piece in convex_pieces(spec.wall_size[1], spec.wall_size[2], rim_outlines)
    ]

    for hole in spec.holes:
        inner, outer = hole_rings(hole, facets)
        # 45-degree chamfer: depth equals its width, clamped to the thickness.
        chamfer_x = min(front_x + hole.chamfer, back_x)
        # Straight section: bore..rim annulus from chamfer plane to back face.
        parts += _ring_wedges(inner, outer, bore_x=(chamfer_x, back_x), rim_x=(chamfer_x, back_x))
        # Funnel: a solid wedge bounded inside by the cone (bore ring at the
        # chamfer plane, collapsing to the rim ring at the front face) and
        # outside by the rim cylinder spanning chamfer plane .. front face.
        parts += _ring_wedges(
            inner, outer, bore_x=(chamfer_x, chamfer_x), rim_x=(chamfer_x, front_x)
        )
    return parts
