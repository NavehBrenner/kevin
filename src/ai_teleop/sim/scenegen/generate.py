"""End-to-end pipeline: resolved WallSpec -> on-disk MJCF + meshes + header.

The high-level sampling entrypoint (``generate_wall(seed, true_hole, ...)``)
that draws unspecified fields is layered on top of this in the sampler step.
``generate_from_spec`` is the deterministic core: given a fully-resolved spec it
builds the CadQuery solid for the *visual* mesh, derives the *collision* parts
analytically (prism extrude + chamfer wedges), and emits all artifacts into
``out_dir``.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import emit, meta, solid
from .config import DEFAULT_RANGES as _DEFAULT_RANGES
from .config import SamplingRanges, WallScene, WallSpec
from .decompose import to_trimesh, wall_collision_parts
from .sampler import sample_wall_spec

# Default root for generated wall libraries; each wall lands in a seed-named
# subdirectory unless the caller overrides out_dir.
DEFAULT_OUT_ROOT = Path("outputs/walls")


def generate_wall(
    seed: int | None = None,
    true_hole: dict | None = None,
    distractors: list[dict] | int | None = None,
    wall_size: tuple[float, float, float] | None = None,
    *,
    out_dir: str | Path | None = None,
    ranges: SamplingRanges = _DEFAULT_RANGES,
    cache: bool = True,
) -> WallScene:
    """Sample a wall from a (possibly sparse) request and write all artifacts.

    The public entrypoint: resolves omitted fields from the seed, places holes
    without overlap, then runs the deterministic build. See ``sampler`` for the
    resolution rules and ``generate_from_spec`` for the build.
    """
    spec = sample_wall_spec(
        seed=seed,
        true_hole=true_hole,
        distractors=distractors,
        wall_size=wall_size,
        ranges=ranges,
    )
    if out_dir is None:
        out_dir = DEFAULT_OUT_ROOT / f"wall_{spec.seed}"
    if cache:
        cached = _load_cached_scene(Path(out_dir), spec)
        if cached is not None:
            return cached
    return generate_from_spec(spec, out_dir)


def generate_from_spec(
    spec: WallSpec,
    out_dir: str | Path,
    *,
    tessellation_tolerance_mm: float = 0.2,
) -> WallScene:
    """Build every artifact for ``spec`` under ``out_dir`` and return a WallScene."""
    out_dir = Path(out_dir)

    workpiece = solid.build_wall_solid(spec)
    vertices, triangles = solid.tessellate_to_metres(
        workpiece, tolerance_mm=tessellation_tolerance_mm
    )

    visual_mesh = to_trimesh(vertices, triangles)
    collision_parts = wall_collision_parts(spec)

    visual_name, collision_names = emit.write_meshes(out_dir, visual_mesh, collision_parts)
    mjcf_path = emit.write_mjcf(out_dir, spec, visual_name, collision_names)
    header_path = meta.write_header(out_dir, spec)

    return WallScene(
        spec=spec,
        mjcf_path=str(mjcf_path),
        visual_mesh_path=str(out_dir / f"{visual_name}.obj"),
        collision_mesh_paths=[str(out_dir / f"{name}.obj") for name in collision_names],
        header_path=str(header_path),
    )


def _load_cached_scene(out_dir: Path, spec: WallSpec) -> WallScene | None:
    """Return a WallScene for an already-built wall in ``out_dir`` iff its
    ``header.json`` exactly matches ``spec`` and its artifacts are all present.

    The expensive part of generation is the CadQuery solid + tessellation +
    mesh export; sampling is cheap. So we always re-sample (to get the spec to
    compare against) but skip the build when a byte-identical wall already
    exists on disk. Returns ``None`` on any mismatch or missing file so the
    caller falls back to a full rebuild.
    """
    header = out_dir / "header.json"
    mjcf = out_dir / "wall.xml"
    visual = out_dir / "wall_visual.obj"
    if not (header.exists() and mjcf.exists() and visual.exists()):
        return None
    try:
        # Round-trip the live spec through JSON so tuples (e.g. in
        # sampling_ranges) compare equal to the lists loaded from disk.
        spec_json = json.loads(json.dumps(meta.spec_to_dict(spec)))
        if json.loads(header.read_text(encoding="utf-8")) != spec_json:
            return None
    except (OSError, ValueError):
        return None
    collision_paths = sorted(str(p) for p in out_dir.glob("wall_col_*.obj"))
    if not collision_paths:
        return None
    return WallScene(
        spec=spec,
        mjcf_path=str(mjcf),
        visual_mesh_path=str(visual),
        collision_mesh_paths=collision_paths,
        header_path=str(header),
        from_cache=True,
    )
