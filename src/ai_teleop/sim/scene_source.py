"""Resolve which scene XML a SimEnv runner should load.

Two sources, one return type (an absolute scene-path that `SimEnv` accepts):

  * the static hand-authored task scene (`assets/mjcf/full_scene.xml`) — the
    default everywhere; a fixed 3-hole wall.
  * a freshly procedurally-generated wall, wrapped in the same robot+peg task
    scene via the scenegen `compose_scene`. Reproducible from a seed.

This is the single place that knows how to turn "I want a generated wall" into
something `SimEnv(scene_path)` can open, so every runner (run_episode, the
recorders, future eval) can offer the same `--generated-wall` flag without
duplicating the generate -> compose plumbing.
"""

from __future__ import annotations

from pathlib import Path

from ai_teleop.common.log import get_logger
from ai_teleop.sim.scenegen.compose import compose_scene
from ai_teleop.sim.scenegen.generate import generate_wall

STATIC_TASK_SCENE = Path(__file__).resolve().parents[3] / "assets" / "mjcf" / "full_scene.xml"

# Module name avoids shadowing the `log: bool` emit-gate param below.
_logger = get_logger("scene")


def resolve_scene_path(
    *,
    generated: bool = False,
    wall_seed: int | None = None,
    distractors: int | None = None,
    log: bool = True,
) -> Path:
    """Return an absolute scene path for `SimEnv`.

    `generated=False` -> the static task scene. `generated=True` -> generate a
    wall (reproducible from `wall_seed`; `distractors=None` lets scenegen pick)
    and compose the full robot+peg task scene around it. With `log`, prints a
    one-line status (cache hit vs built from scratch).
    """
    if not generated:
        if log:
            _logger.info("static task wall: %s", STATIC_TASK_SCENE.name)
        return STATIC_TASK_SCENE
    import numpy as np

    wall = generate_wall(seed=wall_seed, distractors=distractors)
    if log:
        status = "cache hit" if wall.from_cache else "built from scratch"
        tilt = np.round(np.rad2deg(wall.spec.orientation), 1)
        _logger.info(
            "generated wall seed %s: %s (%d holes, tilt %s°)",
            wall.spec.seed,
            status,
            len(wall.spec.holes),
            tilt,
        )
    return compose_scene(Path(wall.mjcf_path), with_robot=True).resolve()
