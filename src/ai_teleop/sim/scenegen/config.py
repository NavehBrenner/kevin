"""Configuration, sampling ranges, and the resolved-spec dataclasses for the
parametric wall/hole scene generator.

Frame convention (matches the hand-written ``wall_with_holes.xml`` it replaces):
the wall body is anchored in the world and its *local* axes are

    local +x = wall thickness / insertion depth   (world +x; robot looks down +x)
    local +y = wall width   (horizontal)
    local +z = wall height  (vertical)

The robot-facing surface is the local ``-x`` face. Holes are placed on the
``(y, z)`` plane and drilled along ``x``; the chamfer funnel opens toward the
robot on the ``-x`` face. A hole's ``pos`` is therefore a 2-tuple ``(y, z)`` in
metres, measured from the wall centre.

All lengths are metres. CadQuery itself happens to work in millimetres; the
conversion lives in ``solid.py`` so nothing outside it sees mm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Literal

HoleShape = Literal["circle", "rect", "slot", "keyhole", "polygon"]

# --- Defaults -------------------------------------------------------------

# (thickness_x, width_y, height_z) in metres. Matches the current wall:
# thickness 0.02, 0.40 x 0.40 face. Never sampled — used verbatim when the
# caller omits wall_size.
DEFAULT_WALL_SIZE: tuple[float, float, float] = (0.02, 0.40, 0.40)

# Number of distractors when the caller passes ``distractors=None``: an integer
# drawn uniformly from this inclusive range.
DISTRACTOR_COUNT_RANGE: tuple[int, int] = (0, 10)

# Facets used to approximate a curved hole boundary (e.g. a round bore) in the
# *collision* geometry. Each facet contributes a back-ring + a collar wedge per
# hole, so higher = smoother funnel but more geoms. The visual mesh is
# tessellated independently (and more finely) by CadQuery.
COLLISION_FACETS: int = 16


# --- Sampling ranges ------------------------------------------------------
# The active SamplingRanges instance is snapshotted into header.json so a scene
# stays reproducible even if these module defaults change later.


@dataclass(frozen=True)
class SamplingRanges:
    """Bounds used to draw any hole field the caller leaves unspecified."""

    # Circle bore diameter (m). Default peg is 8 mm Ø; keep openings comfortably
    # above peg+clearance.
    diameter: tuple[float, float] = (0.010, 0.030)
    # Rect / slot footprint (m): (min, max) applied independently per axis.
    rect_side: tuple[float, float] = (0.012, 0.040)
    # Slot aspect: length is rect_side; width is this fraction of length.
    slot_width_frac: tuple[float, float] = (0.30, 0.60)
    # Regular-polygon circumradius (m) and vertex count.
    polygon_radius: tuple[float, float] = (0.008, 0.020)
    polygon_sides: tuple[int, int] = (3, 8)
    # Rim chamfer width (m). The primary difficulty knob.
    chamfer: tuple[float, float] = (0.001, 0.004)
    # Clear margin (m) kept between any hole's bounding circle and the wall edge.
    edge_margin: float = 0.015
    # Minimum gap (m) between the bounding circles of any two holes.
    min_hole_gap: float = 0.010
    # Wall tilt per axis (radians). Each of the 3 body-local axes gets an
    # independent angle drawn as |Normal(0, std)| * random_sign, clipped to
    # +-max — so small tilts dominate and large ones are rare / capped.
    wall_rotation_std: float = 0.1745  # ~10 deg
    wall_rotation_max: float = 0.3491  # ~20 deg


DEFAULT_RANGES = SamplingRanges()


# --- Resolved spec --------------------------------------------------------


@dataclass
class HoleSpec:
    """A fully-resolved hole. Every field is concrete (nothing left to sample).

    ``size`` is shape-dependent:
        circle  -> {"diameter": d}
        rect    -> {"width": w, "height": h}
        slot    -> {"length": l, "width": w}
        polygon -> {"radius": r, "sides": n}
        keyhole -> {"diameter": d, "slot_width": w, "slot_length": l}
    """

    shape: HoleShape
    pos: tuple[float, float]  # (y, z) in metres, wall-centre origin
    size: dict[str, float]
    chamfer: float  # rim chamfer width (m)
    is_target: bool = False

    def bounding_radius(self) -> float:
        """Radius of the smallest circle (centred on ``pos``) enclosing the hole,
        chamfer rim included.

        Used for overlap / margin tests — cheap and shape-agnostic. The chamfer
        is added because the rim extends ``chamfer`` beyond the nominal bore, so
        spacing must keep rims (not just bores) apart.
        """
        size = self.size
        if self.shape == "circle":
            bore = 0.5 * size["diameter"]
        elif self.shape == "rect":
            bore = 0.5 * (size["width"] ** 2 + size["height"] ** 2) ** 0.5
        elif self.shape == "slot":
            bore = 0.5 * (size["length"] ** 2 + size["width"] ** 2) ** 0.5
        elif self.shape == "polygon":
            bore = size["radius"]
        elif self.shape == "keyhole":
            # Bore plus the slot reaching out along +z; bound generously.
            bore = 0.5 * size["diameter"] + size["slot_length"]
        else:
            raise ValueError(f"unknown hole shape: {self.shape!r}")
        return bore + self.chamfer


@dataclass
class WallSpec:
    """The complete, resolved description of one generated wall — everything
    needed to rebuild it byte-for-byte, independent of the module defaults."""

    seed: int
    wall_size: tuple[float, float, float]
    holes: list[HoleSpec]  # holes[0] is always the target
    ranges: SamplingRanges = field(default_factory=lambda: DEFAULT_RANGES)
    # Body tilt (rx, ry, rz) in radians about the wall-local axes; (0,0,0) = upright.
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    seed_was_given: bool = True

    @property
    def target_hole(self) -> HoleSpec:
        return self.holes[0]


@dataclass
class WallScene:
    """What ``generate_wall`` returns: the resolved spec plus paths to the
    artifacts written to disk."""

    spec: WallSpec
    mjcf_path: str
    visual_mesh_path: str
    collision_mesh_paths: list[str]
    header_path: str
    from_cache: bool = False


def resolve_seed(seed: int | None) -> tuple[int, bool]:
    """Return ``(seed, was_given)``; falls back to wall-clock time when None."""
    if seed is None:
        return int(time.time()), False
    return int(seed), True


def with_ranges(spec: WallSpec, ranges: SamplingRanges) -> WallSpec:
    """Functional update — used by tests to pin non-default ranges."""
    return replace(spec, ranges=ranges)
