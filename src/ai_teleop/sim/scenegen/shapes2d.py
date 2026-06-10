"""2D hole outlines in the wall face plane (y, z).

Each hole is reduced to two index-paired rings of polygon vertices:

    inner ring -> the bore boundary (the tight opening the peg passes through)
    outer ring -> the rim boundary = bore inflated by the chamfer width

The collision builder extrudes the bulk against the *outer* ring and fills the
inner annulus (bore..rim) with per-facet wedges, so these two rings drive the
whole collision decomposition. They share vertex count and ordering so facet
``j`` is the quad ``inner[j], inner[j+1], outer[j+1], outer[j]``.

Only ``circle`` is implemented here for now; the rect/slot/keyhole/polygon
shapes are added with the shape library (they reduce to the same paired-ring
contract, e.g. via a shapely offset of the base polygon).
"""

from __future__ import annotations

import numpy as np

from .config import HoleSpec


def hole_rings(hole: HoleSpec, facets: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(inner_ring, outer_ring)``, each ``(facets, 2)`` in (y, z) metres."""
    if hole.shape == "circle":
        return _circle_rings(hole, facets)
    raise NotImplementedError(f"hole shape {hole.shape!r} not implemented yet")


def _circle_rings(hole: HoleSpec, facets: int) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(hole.pos, dtype=np.float64)
    bore_radius = 0.5 * hole.size["diameter"]
    rim_radius = bore_radius + hole.chamfer
    # Half-facet offset keeps the bore's inscribed radius closer to the true
    # circle so the faceted opening never bites into the peg clearance.
    angles = np.linspace(0.0, 2.0 * np.pi, facets, endpoint=False)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    inner = center + bore_radius * unit
    outer = center + rim_radius * unit
    return inner, outer


def outer_outline(hole: HoleSpec, facets: int) -> np.ndarray:
    """The rim polygon (outer ring) — what the bulk plate is cut against."""
    return hole_rings(hole, facets)[1]
