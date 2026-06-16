"""Serialize a resolved WallSpec to ``header.json`` — the provenance record.

The header stores the seed *and* every resolved value (wall size, sampling
ranges actually in force, and each hole's concrete fields) so a scene is
reproducible even from a run where parameters were sampled rather than given,
and even if the module-default ranges change later.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import WallSpec


def spec_to_dict(spec: WallSpec) -> dict:
    return {
        "seed": spec.seed,
        "seed_was_given": spec.seed_was_given,
        "wall_size": list(spec.wall_size),
        "orientation": list(spec.orientation),
        "sampling_ranges": asdict(spec.ranges),
        "target_hole": _hole_to_dict(spec.holes[0]),
        "holes": [_hole_to_dict(hole) for hole in spec.holes],
    }


def _hole_to_dict(hole) -> dict:
    return {
        "shape": hole.shape,
        "pos": list(hole.pos),
        "size": hole.size,
        "chamfer": hole.chamfer,
        "is_target": hole.is_target,
    }


def write_header(out_dir: Path, spec: WallSpec) -> Path:
    path = out_dir / "header.json"
    path.write_text(json.dumps(spec_to_dict(spec), indent=2))
    return path
