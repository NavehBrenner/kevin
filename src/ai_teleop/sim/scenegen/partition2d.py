"""Convex partition of the wall face: a rectangle minus the hole rim polygons.

The wall is a prism, so its collision geometry is found by partitioning the 2D
cross-section (the face, in the y-z plane) into convex polygons and extruding
each through the thickness. This module returns those convex 2D pieces.

v1 uses a constrained triangulation (every triangle is trivially convex), which
is exact, fast (<1 ms), and deterministic. A later optimisation can merge
adjacent triangles into larger convex polygons (Hertel-Mehlhorn) to cut the
geom count; the extrude builder does not care how convex the pieces are, only
that they are.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import trimesh.creation
from shapely.geometry import Polygon


def convex_pieces(
    width: float, height: float, hole_outlines: list[np.ndarray]
) -> list[np.ndarray]:
    """Partition the ``width x height`` face (centred on origin) minus the given
    hole rim polygons into convex 2D pieces.

    Args:
        width, height: full face dimensions (m), along y and z.
        hole_outlines: list of ``(k, 2)`` rim polygons (y, z) to subtract.

    Returns:
        list of ``(m, 2)`` convex polygons (counter-clockwise) tiling the face.
    """
    half_w, half_h = 0.5 * width, 0.5 * height
    face = Polygon(
        [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    )
    for outline in hole_outlines:
        face = face.difference(Polygon(outline))

    vertices, faces = cast(
        "tuple[np.ndarray, np.ndarray]",
        trimesh.creation.triangulate_polygon(face, engine="earcut"),
    )
    return [vertices[triangle] for triangle in faces]
