"""Bridge from the offline wall generator to a runnable SimEnv.

`make_env` turns an `EnvConfig` into a runnable env (static or generated wall);
`make_wall_task_env` is the richer call that turns a (possibly sparse) wall
request into a ready-to-step task environment: it generates a procedural wall,
composes it into the full task scene (Panda + pre-grasped peg + the generated
wall), and returns a SimEnv pointed at that scene. The generator always places
the target at `hole_0` — but which hole is the goal is the task layer's choice,
not the env's (the env just reports every hole's pose).

This keeps the heavy CAD/generation deps (the `scenegen` extra) out of the sim
runtime's import path: they are only pulled in when this bridge is called.
"""

from __future__ import annotations

from pathlib import Path

from .config import EnvConfig
from .scene import RenderMode, SimEnv

# Mirrors `scene_source.STATIC_TASK_SCENE`, defined locally so the static-wall
# path through `make_env` does not import `scene_source` (whose module-level
# scenegen imports would drag in the CadQuery extra — the very thing this bridge
# keeps off the non-generated path).
_STATIC_TASK_SCENE = Path(__file__).resolve().parents[3] / "assets" / "mjcf" / "full_scene.xml"


def make_env(
    config: EnvConfig,
    *,
    render_mode: RenderMode = "headless",
    camera_height: int = 224,
    camera_width: int = 224,
) -> SimEnv:
    """Build a runnable :class:`SimEnv` from an :class:`EnvConfig`.

    Resolves the config's ``wall_seed`` to a scene: ``None`` → the static
    hand-authored wall (no ``scenegen``/CadQuery import); an integer → a
    procedurally generated wall (lazily importing the optional ``scenegen`` extra
    only on that path, so static/scenegen-free runs never pull CadQuery).
    """
    if config.wall_seed is None:
        scene_path: Path = _STATIC_TASK_SCENE
    else:
        # Lazy — `resolve_scene_path` imports the scenegen (CadQuery) extra at
        # module load, which we keep off the static path.
        from .scene_source import resolve_scene_path  # noqa: PLC0415

        scene_path = resolve_scene_path(generated=True, wall_seed=config.wall_seed)
    return SimEnv(
        str(scene_path),
        render_mode=render_mode,
        camera_height=camera_height,
        camera_width=camera_width,
        config=config,
    )


def make_wall_task_env(
    seed: int | None = None,
    *,
    true_hole: dict | None = None,
    distractors: list[dict] | int | None = None,
    wall_size: tuple[float, float, float] | None = None,
    wall_dir: str | Path | None = None,
    render_mode: RenderMode = "headless",
    camera_height: int = 224,
    camera_width: int = 224,
) -> SimEnv:
    """Generate (or reuse) a procedural wall and return a SimEnv on the task scene.

    Args:
        seed, true_hole, distractors, wall_size: forwarded to ``generate_wall``
            (ignored when ``wall_dir`` is given).
        wall_dir: reuse a previously generated wall directory instead of
            generating a fresh one. Must contain ``wall.xml``.
        render_mode, camera_*: forwarded to ``SimEnv``.

    Returns:
        A ``SimEnv`` whose scene is Panda + peg + the generated wall, with the
        target hole at index 0.
    """
    # Imported lazily so importing the sim runtime never drags in CadQuery.
    from .scenegen.compose import compose_scene

    if wall_dir is not None:
        wall_xml = Path(wall_dir) / "wall.xml"
        if not wall_xml.exists():
            raise FileNotFoundError(f"no wall.xml in {wall_dir!r}")
    else:
        from .scenegen.generate import generate_wall

        scene = generate_wall(
            seed=seed, true_hole=true_hole, distractors=distractors, wall_size=wall_size
        )
        wall_xml = Path(scene.mjcf_path)

    scene_path = compose_scene(wall_xml, with_robot=True)
    return SimEnv(
        scene_path,
        render_mode=render_mode,
        camera_height=camera_height,
        camera_width=camera_width,
        config=EnvConfig(wall_seed=seed),
    )
