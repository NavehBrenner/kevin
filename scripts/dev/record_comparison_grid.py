"""Record a 2x2 comparison clip: Expert vs No-assist, third-person + wrist cam.

Runs two episodes through the M3/M4 seam with the *same* scripted operator
(same seed) so the only difference is the assist source:

    row 0 : EXPERT     [ third-person | wrist camera ]
    row 1 : NO ASSIST  [ third-person | wrist camera ]

Each panel is labelled. The third-person camera is placed behind the wall,
looking back toward the arm. Output defaults to MP4 (seekable / pausable);
pass --format gif for a GIF instead. Pass --generated-wall to run on a freshly
procedurally-generated wall instead of the static task scene. --max-dpos raises
the controller's per-step command clamp (the free-space approach-speed knob).

Run: uv run python scripts/dev/record_comparison_grid.py
     uv run python scripts/dev/record_comparison_grid.py --format gif
     uv run python scripts/dev/record_comparison_grid.py --generated-wall --wall-seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    _FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
except OSError:  # fall back to the bitmap default if the TTF is unavailable
    _FONT = ImageFont.load_default()

from ai_teleop.control import Controller
from ai_teleop.domain import NoAssist, apply_delta
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.scene import SimEnv
from ai_teleop.sim.scene_source import resolve_scene_path

PANEL = 480
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
STEM = "comparison_grid"


def make_camera(args: argparse.Namespace) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, 0.0, 0.5]
    cam.distance = args.cam_distance
    cam.azimuth = args.cam_azimuth  # behind-the-wall view by default
    cam.elevation = args.cam_elevation
    return cam


def _progress(prefix: str, done: int, total: int) -> None:
    """In-place ASCII progress bar on one line; newline when complete."""
    filled = int(28 * done / total)
    bar = "█" * filled + "░" * (28 - filled)
    end = "\n" if done >= total else ""
    print(f"\r  {prefix:11} [{bar}] {done:3d}/{total} frames", end=end, flush=True)


def run_views(
    scene_path: Path,
    assist,
    seed: int,
    steps: int,
    every: int,
    cam: mujoco.MjvCamera,
    max_dpos: float,
    progress_label: str = "",
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Run one episode; return (third_person_frames, wrist_frames, final_mm)."""
    env = SimEnv(str(scene_path), render_mode="headless", camera_height=PANEL, camera_width=PANEL)
    observation = env.reset()
    controller = Controller(env, max_dpos_per_step=max_dpos)
    target = observation.hole_poses[observation.target_hole_index][:3].copy()
    human = ScriptedNoisyHuman(
        np.concatenate([target, controller.home_pose[3:]]),
        position_bias_std=0.012,
        orientation_bias_std=np.deg2rad(4),
        seed=seed,
    )
    third = mujoco.Renderer(env.model, height=PANEL, width=PANEL)
    third_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    total_frames = len(range(0, steps, every))
    for t in range(steps):
        base = human.get_command(observation)
        command = apply_delta(base, assist.get_delta(observation, base))
        controller.compute(observation, command)
        env.step()
        observation = env.get_observation()
        if t % every == 0:
            third.update_scene(env.data, camera=cam)
            third_frames.append(third.render().copy())
            wrist = env.render_wrist_camera()
            wrist_frames.append(
                np.asarray(Image.fromarray(wrist).resize((PANEL, PANEL), Image.NEAREST))
            )
            if progress_label:
                _progress(progress_label, len(third_frames), total_frames)
    third.close()
    final_mm = float(np.linalg.norm(observation.peg_pose[:3] - target)) * 1000
    env.close()
    return third_frames, wrist_frames, final_mm


def label(frame: np.ndarray, text: str) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, PANEL, 34], fill=(0, 0, 0, 170))
    draw.text((10, 7), text, fill=(255, 255, 255, 255), font=_FONT)
    return img


def save_animation(frames: list[np.ndarray], path: Path, fps: float) -> None:
    """Write `frames` (list of HxWx3 uint8) as MP4 or GIF, inferred from suffix."""
    if path.suffix == ".mp4":
        import imageio.v3 as iio

        # even dimensions required by the H.264 encoder.
        h, w = frames[0].shape[:2]
        stack = np.stack(frames)[:, : h - (h % 2), : w - (w % 2)]
        iio.imwrite(path, stack, fps=fps, codec="libx264", quality=8)
    else:
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument(
        "--format",
        choices=("mp4", "gif"),
        default="mp4",
        help="output container (mp4 is seekable/pausable; default)",
    )
    ap.add_argument("--fps", type=float, default=20.0, help="playback frame rate")
    ap.add_argument(
        "--max-dpos",
        type=float,
        default=0.025,
        help="controller per-step command clamp in metres (approach-speed knob)",
    )
    ap.add_argument(
        "--generated-wall",
        action="store_true",
        help="run on a freshly generated wall instead of the static scene",
    )
    ap.add_argument("--wall-seed", type=int, default=7)
    ap.add_argument("--distractors", type=int, default=None)
    ap.add_argument("--cam-azimuth", type=float, default=-40.0)
    ap.add_argument("--cam-elevation", type=float, default=-18.0)
    ap.add_argument("--cam-distance", type=float, default=1.6)
    args = ap.parse_args()

    scene_path = resolve_scene_path(
        generated=args.generated_wall,
        wall_seed=args.wall_seed,
        distractors=args.distractors,
    )
    cam = make_camera(args)

    print("rendering 2 episodes:")
    ex_third, ex_wrist, ex_mm = run_views(
        scene_path, Expert(), args.seed, args.steps, args.every, cam, args.max_dpos, "expert"
    )
    na_third, na_wrist, na_mm = run_views(
        scene_path, NoAssist(), args.seed, args.steps, args.every, cam, args.max_dpos, "no-assist"
    )
    print(f"expert final peg->hole = {ex_mm:.0f} mm   no-assist = {na_mm:.0f} mm")

    n = min(len(ex_third), len(na_third))
    grid_frames: list[np.ndarray] = []
    for i in range(n):
        top = np.concatenate(
            [
                np.asarray(label(ex_third[i], "EXPERT  -  third person")),
                np.asarray(label(ex_wrist[i], "EXPERT  -  wrist camera")),
            ],
            axis=1,
        )
        bottom = np.concatenate(
            [
                np.asarray(label(na_third[i], "NO ASSIST  -  third person")),
                np.asarray(label(na_wrist[i], "NO ASSIST  -  wrist camera")),
            ],
            axis=1,
        )
        grid_frames.append(np.concatenate([top, bottom], axis=0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{STEM}_{args.seed}_{args.wall_seed}.{args.format}"
    print(f"encoding {n} frames → {args.format} ...")
    save_animation(grid_frames, out, args.fps)
    Image.fromarray(grid_frames[int(n * 0.85)]).save(OUT_DIR / f"{STEM}.still.png")
    print(f"wrote {out}  ({n} frames, {grid_frames[0].shape[1]}x{grid_frames[0].shape[0]})")


if __name__ == "__main__":
    main()
