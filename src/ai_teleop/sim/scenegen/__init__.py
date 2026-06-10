"""Parametric wall/hole scene generator (offline).

CadQuery builds a parametric wall-with-holes solid for the *visual* mesh; the
*collision* geometry is derived analytically (prism extrude + chamfer wedges)
and the result is emitted as a drop-in MJCF wall plus a provenance
``header.json``.

Heavy CAD deps (CadQuery) live behind the ``generate``/``solid`` modules and
are imported lazily: ``from ai_teleop.sim.scenegen import HoleSpec`` pulls in
only the dataclasses, while ``generate_wall`` / ``generate_from_spec`` are
resolved on first access (so the sim runtime can import the types without
dragging in CadQuery). ``sample_wall_spec`` is light (numpy only).

Public surface:
    config: HoleSpec, WallSpec, WallScene, SamplingRanges, defaults
    generate_wall: sparse request -> sampled, placed, on-disk wall (entrypoint)
    generate_from_spec: resolved spec -> on-disk artifacts
    sample_wall_spec: sparse request -> resolved spec (no I/O)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
    DEFAULT_RANGES,
    DEFAULT_WALL_SIZE,
    HoleSpec,
    SamplingRanges,
    WallScene,
    WallSpec,
)

if TYPE_CHECKING:  # for type-checkers/IDEs without triggering the CadQuery import
    from .generate import generate_from_spec, generate_wall
    from .sampler import sample_wall_spec

__all__ = [
    "DEFAULT_RANGES",
    "DEFAULT_WALL_SIZE",
    "HoleSpec",
    "SamplingRanges",
    "WallScene",
    "WallSpec",
    "generate_from_spec",
    "generate_wall",
    "sample_wall_spec",
]

_LAZY = {
    "generate_wall": ("generate", "generate_wall"),
    "generate_from_spec": ("generate", "generate_from_spec"),
    "sample_wall_spec": ("sampler", "sample_wall_spec"),
}


def __getattr__(name: str):
    """Resolve the build entrypoints on first access (PEP 562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    return getattr(module, attr)
