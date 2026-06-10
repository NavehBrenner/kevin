"""Generate (or load) a procedural wall and view it — interactively or as PNGs.

Examples:
    # Generate a random wall and open it in the task scene (robot + peg + wall):
    uv run python scripts/view_generated_wall.py --seed 7

    # Wall on its own (no robot), interactive:
    uv run python scripts/view_generated_wall.py --seed 7 --no-robot

    # An explicit target + 3 distractors, render PNGs instead of a window:
    uv run python scripts/view_generated_wall.py --seed 1 --distractors 3 --render

    # Re-view a wall already generated under outputs/walls/<id>/:
    uv run python scripts/view_generated_wall.py --wall-dir outputs/walls/wall_7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from ai_teleop.sim.scenegen.compose import compose_scene
from ai_teleop.sim.scenegen.generate import generate_wall


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: time-based)")
    ap.add_argument("--distractors", type=int, default=None,
                    help="number of distractor holes (default: random 0-10)")
    ap.add_argument("--wall-dir", type=Path, default=None,
                    help="view an existing wall dir instead of generating")
    ap.add_argument("--no-robot", action="store_true",
                    help="preview the wall alone (no Panda/peg)")
    ap.add_argument("--render", action="store_true",
                    help="render PNGs instead of opening an interactive window")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.wall_dir is not None:
        wall_xml = args.wall_dir / "wall.xml"
        print(f"using existing wall: {wall_xml}")
    else:
        scene = generate_wall(seed=args.seed, distractors=args.distractors)
        wall_xml = Path(scene.mjcf_path)
        print(f"generated wall (seed={scene.spec.seed}, holes={len(scene.spec.holes)}): {wall_xml}")

    scene_path = compose_scene(wall_xml, with_robot=not args.no_robot)
    print(f"composed scene: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    if not args.no_robot:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    if args.render:
        from PIL import Image
        cams = ["external_cam"] if not args.no_robot else ["front", "angled"]
        renderer = mujoco.Renderer(model, height=768, width=1024)
        for cam in cams:
            renderer.update_scene(data, camera=cam)
            out = scene_path.parent / f"view_{cam}.png"
            Image.fromarray(renderer.render()).save(out)
            print(f"  wrote {out}")
        renderer.close()
    else:
        print("opening interactive viewer (close the window to exit)...")
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
