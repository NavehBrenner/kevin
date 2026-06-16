"""Resolve a (possibly sparse) user request into a fully-concrete WallSpec.

Implements the locked ``generate_wall`` contract:

  * Any hole field the caller omits is drawn from the seeded RNG using
    ``SamplingRanges``.
  * ``distractors`` is ``list[dict]`` (explicit), ``int`` (that many sampled),
    or ``None`` (a count drawn uniformly from ``DISTRACTOR_COUNT_RANGE``).
  * Holes given an explicit ``pos`` are validated first: if any pair overlaps,
    or one breaches the wall edge margin, generation fails loudly. Holes with
    no ``pos`` are then rejection-sampled so they overlap nothing already placed.

The target hole is always ``holes[0]`` (``is_target=True``).
"""

from __future__ import annotations

import numpy as np

from .config import (
    DEFAULT_RANGES,
    DEFAULT_WALL_SIZE,
    DISTRACTOR_COUNT_RANGE,
    HoleSpec,
    SamplingRanges,
    WallSpec,
    resolve_seed,
)

# Shapes the generator can currently realise. Extended with the shape library;
# the sampler only ever draws from this set.
IMPLEMENTED_SHAPES = ("circle",)

_MAX_PLACEMENT_ATTEMPTS = 2000


def _sample_orientation(
    rng: np.random.Generator, ranges: SamplingRanges
) -> tuple[float, float, float]:
    """Independent per-axis tilt: |Normal(0, std)| * random_sign, clipped to +-max."""
    magnitude = np.minimum(
        np.abs(rng.normal(0.0, ranges.wall_rotation_std, size=3)), ranges.wall_rotation_max
    )
    sign = np.where(rng.random(3) < 0.5, -1.0, 1.0)
    tilt = magnitude * sign
    return (float(tilt[0]), float(tilt[1]), float(tilt[2]))


def sample_wall_spec(
    seed: int | None = None,
    true_hole: dict | None = None,
    distractors: list[dict] | int | None = None,
    wall_size: tuple[float, float, float] | None = None,
    *,
    ranges: SamplingRanges = DEFAULT_RANGES,
) -> WallSpec:
    """Resolve the request into a concrete WallSpec (no disk I/O)."""
    resolved_seed, seed_was_given = resolve_seed(seed)
    rng = np.random.default_rng(resolved_seed)
    resolved_wall_size: tuple[float, float, float] = (
        DEFAULT_WALL_SIZE
        if wall_size is None
        else (float(wall_size[0]), float(wall_size[1]), float(wall_size[2]))
    )

    requests = _collect_requests(rng, true_hole, distractors)

    # Resolve shape/size/chamfer for every hole up front (placement needs the
    # bounding radius); pos stays None until the placement pass.
    holes: list[_PendingHole] = []
    for given, is_target in requests:
        shape, size, chamfer = _resolve_shape_size_chamfer(rng, ranges, given)
        pos = tuple(given["pos"]) if (given and given.get("pos") is not None) else None
        holes.append(_PendingHole(shape, pos, size, chamfer, is_target))

    placed = _place_holes(rng, holes, resolved_wall_size, ranges)
    orientation = _sample_orientation(rng, ranges)
    return WallSpec(
        seed=resolved_seed,
        wall_size=resolved_wall_size,
        holes=placed,
        ranges=ranges,
        orientation=orientation,
        seed_was_given=seed_was_given,
    )


class _PendingHole:
    """A hole with everything resolved except (maybe) its position."""

    def __init__(self, shape, pos, size, chamfer, is_target):
        self.shape = shape
        self.pos = pos  # tuple | None
        self.size = size
        self.chamfer = chamfer
        self.is_target = is_target

    def bounding_radius(self) -> float:
        return HoleSpec(self.shape, (0.0, 0.0), self.size, self.chamfer).bounding_radius()

    def finalize(self, pos) -> HoleSpec:
        return HoleSpec(self.shape, tuple(pos), self.size, self.chamfer, self.is_target)


def _collect_requests(
    rng: np.random.Generator,
    true_hole: dict | None,
    distractors: list[dict] | int | None,
) -> list[tuple[dict | None, bool]]:
    """Return [(request_dict_or_None, is_target)], target first."""
    requests: list[tuple[dict | None, bool]] = [(true_hole, True)]

    if distractors is None:
        low, high = DISTRACTOR_COUNT_RANGE
        count = int(rng.integers(low, high + 1))
        requests += [(None, False)] * count
    elif isinstance(distractors, int):
        if distractors < 0:
            raise ValueError(f"distractors count must be >= 0, got {distractors}")
        requests += [(None, False)] * distractors
    else:
        requests += [(dict(d), False) for d in distractors]
    return requests


def _resolve_shape_size_chamfer(
    rng: np.random.Generator, ranges: SamplingRanges, given: dict | None
) -> tuple[str, dict[str, float], float]:
    given = given or {}
    shape = given.get("shape") or str(rng.choice(IMPLEMENTED_SHAPES))
    if shape not in IMPLEMENTED_SHAPES:
        raise NotImplementedError(f"hole shape {shape!r} not implemented yet")
    chamfer = given.get("chamfer")
    if chamfer is None:
        chamfer = float(rng.uniform(*ranges.chamfer))
    size = given.get("size") or _sample_size(rng, ranges, shape)
    return shape, size, float(chamfer)


def _sample_size(rng: np.random.Generator, ranges: SamplingRanges, shape: str) -> dict[str, float]:
    if shape == "circle":
        return {"diameter": float(rng.uniform(*ranges.diameter))}
    raise NotImplementedError(f"size sampling for {shape!r} not implemented yet")


def _place_holes(
    rng: np.random.Generator,
    pending: list[_PendingHole],
    wall_size: tuple[float, float, float],
    ranges: SamplingRanges,
) -> list[HoleSpec]:
    """Validate explicit positions, then rejection-sample the rest."""
    half_w = 0.5 * wall_size[1]
    half_h = 0.5 * wall_size[2]
    margin = ranges.edge_margin
    gap = ranges.min_hole_gap

    placed_centers: list[tuple[np.ndarray, float]] = []  # (center, radius)
    result: list[HoleSpec | None] = [None] * len(pending)

    def fits_edges(center: np.ndarray, radius: float) -> bool:
        return (
            abs(center[0]) + radius <= half_w - margin
            and abs(center[1]) + radius <= half_h - margin
        )

    def clear_of_placed(center: np.ndarray, radius: float) -> bool:
        return all(np.linalg.norm(center - c) >= radius + r + gap for c, r in placed_centers)

    # Pass 1: explicit positions, validated loudly.
    for index, hole in enumerate(pending):
        if hole.pos is None:
            continue
        center = np.asarray(hole.pos, dtype=np.float64)
        radius = hole.bounding_radius()
        if not fits_edges(center, radius):
            raise ValueError(
                f"hole {index} at {hole.pos} (radius {radius:.4f} m incl. chamfer) "
                f"breaches the {margin * 1000:.0f} mm edge margin of the "
                f"{wall_size[1]:.3f}x{wall_size[2]:.3f} m wall face."
            )
        if not clear_of_placed(center, radius):
            raise ValueError(
                f"explicit hole {index} at {hole.pos} overlaps a previously given "
                f"hole (need >= {gap * 1000:.0f} mm gap between rims). Adjust the "
                f"positions or remove one."
            )
        placed_centers.append((center, radius))
        result[index] = hole.finalize(center)

    # Pass 2: rejection-sample the rest.
    for index, hole in enumerate(pending):
        if hole.pos is not None:
            continue
        radius = hole.bounding_radius()
        free_y = half_w - margin - radius
        free_z = half_h - margin - radius
        if free_y < 0 or free_z < 0:
            raise ValueError(
                f"hole {index} (radius {radius:.4f} m incl. chamfer) is too large "
                f"to fit the wall face with a {margin * 1000:.0f} mm margin."
            )
        for _ in range(_MAX_PLACEMENT_ATTEMPTS):
            center = np.array([rng.uniform(-free_y, free_y), rng.uniform(-free_z, free_z)])
            if clear_of_placed(center, radius):
                placed_centers.append((center, radius))
                result[index] = hole.finalize(center)
                break
        else:
            raise ValueError(
                f"could not place hole {index} without overlap after "
                f"{_MAX_PLACEMENT_ATTEMPTS} attempts — the wall is too crowded for "
                f"{len(pending)} holes. Use fewer distractors or a larger wall_size."
            )

    return [hole for hole in result if hole is not None]
