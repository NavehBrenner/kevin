"""Bridge from the offline wall generator to a runnable SimEnv.

`make_wall_task_env` is the one call that turns a (possibly sparse) wall request
into a ready-to-step task environment: it generates a procedural wall, composes
it into the full task scene (Panda + pre-grasped peg + the generated wall), and
returns a SimEnv pointed at that scene. The generator always places the target
at `hole_0`, so the env is configured with `target_hole_index=0`.

This keeps the heavy CAD/generation deps (the `scenegen` extra) out of the sim
runtime's import path: they are only pulled in when this bridge is called.
"""

from __future__ import annotations

from pathlib import Path

from .scene import RenderMode, SimEnv


def make_wall_task_env(
    seed: int | None = None,
    *,
    true_hole: dict | None = None,
    distractors: list[dict] | int | None = None,
    wall_size: tuple[float, float, float] | None = None,
    wall_dir: str | Path | None = None,
    render_mode: RenderMode = "headless",
    camera_height: int = 128,
    camera_width: int = 128,
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
        target_hole_index=0,
        seed=seed or 0,
    )
