"""Measure MuJoCo offscreen-render cost on this box (hardware vs software GL).

The question: would replacing the passive viewer with offscreen-render + cv2.imshow be
cheaper? That hinges on whether the offscreen `mujoco.Renderer` (EGL) gets hardware GL or
falls back to software (llvmpipe) on WSL. A hardware render is ~1-3 ms/frame; software is
~20-50 ms. This times render_wrist_camera (which drives the offscreen Renderer) and also
times the equivalent of a viewer sync (mj_copyDataVisual via a second Renderer scene update).

    cd kevin && .venv/bin/python scripts/dev/render_cost_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ai_teleop.sim.scene import SimEnv  # noqa: E402
from ai_teleop.sim.scene_source import resolve_scene_path  # noqa: E402


def main() -> None:
    scene_path = resolve_scene_path(generated=False, wall_seed=7, distractors=None)
    env = SimEnv(str(scene_path), render_mode="headless", seed=0)
    env.reset()
    for _ in range(10):  # warm up (lazy Renderer creation, first-frame GL setup)
        env.render_wrist_camera()
    n = 100
    t0 = time.monotonic()
    for _ in range(n):
        env.render_wrist_camera()
    ms = (time.monotonic() - t0) / n * 1000
    print(f"offscreen render: {ms:.2f} ms/frame  ({1000 / ms:.0f} fps ceiling)")
    print("  ~1-3 ms => hardware GL; ~20-50 ms => software (llvmpipe) — imshow would be slower")
    env.close()


if __name__ == "__main__":
    main()
