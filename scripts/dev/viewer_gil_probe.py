"""Isolate whether MuJoCo's passive-viewer render thread starves the stereohand tracker
of GIL time (LAB-74 debug: native-Windows fps drop investigation).

`mujoco.viewer.launch_passive()` spawns its own background `threading.Thread` running a
continuous GLFW render loop on any non-macOS platform (see `_launch_internal` /
`launch_passive` in the installed `mujoco/viewer.py`) - decoupled from anything kevin's own
loop does, which is *why* the interactive viewer stays responsive however busy the rest of
the process is. The hypothesis under test: that thread (plus, in phase 3, the
`SimEnv.sync_viewer()` traffic on top of it) competes with stereohand's own background
capture/MediaPipe thread for the GIL, dropping its effective sensor rate - the standalone
`stereohand/scripts/demo.py` has no such competing thread and runs at full camera fps,
while `kvn episode --input vision` drops to ~20 fps.

Runs the same `StereoHandSource` (kevin's live sensor adapter) through three phases, each
for `--seconds`, and relies on its own sensor-health line (fresh fps / drop-out %, logged
on `close()`) to report each phase's effective rate:

  1. baseline      - tracker only, no viewer at all.
  2. idle viewer    - a launch_passive() viewer window open, never sync()'d.
  3. synced viewer  - the viewer open and sync()'d at run_episode's ~30 Hz default.

A big drop from (1) to (2) implicates the viewer's own render thread (its mere existence);
a further drop from (2) to (3) implicates `sync_viewer()` traffic on top of it.

Keep a hand visible to both cameras (any pose) throughout all three phases: the
sensor-health metric only counts *present* frames, so an empty frame reads as 0 fps
regardless of contention.

    cd kevin && uv run python scripts/dev/viewer_gil_probe.py \
        --stereo-calib ../stereohand/stereo_calib.json --cameras 0 2 --seconds 8
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ai_teleop.common.log import add_logging_arguments, configure_from_args  # noqa: E402
from ai_teleop.input.hand_tracker import StereoHandSource  # noqa: E402
from ai_teleop.sim.scene import SimEnv  # noqa: E402
from ai_teleop.sim.scene_source import resolve_scene_path  # noqa: E402

_SYNC_INTERVAL_S = 1.0 / 30.0  # matches run_episode's DEFAULT_RENDER_FPS
_POLL_INTERVAL_S = 0.002  # mirrors run_episode's 500 Hz physics-rate poll cadence


def _camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _run_phase(
    label: str,
    calib: str,
    left: int | str,
    right: int | str,
    seconds: float,
    *,
    on_tick: Callable[[], None] | None = None,
) -> None:
    """Poll a fresh StereoHandSource for `seconds`; its close() logs the sensor-health line."""
    print(f"\n--- {label} ({seconds:.0f}s; keep a hand in view) ---")
    source = StereoHandSource(calib, left=left, right=right, show_window=False)
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            source.read()
            if on_tick is not None:
                on_tick()
            time.sleep(_POLL_INTERVAL_S)
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-calib", required=True, help="stereohand stereo_calib.json path")
    parser.add_argument(
        "--cameras",
        nargs=2,
        default=["0", "2"],
        metavar=("LEFT", "RIGHT"),
        help="left/right camera sources",
    )
    parser.add_argument("--seconds", type=float, default=8.0, help="duration per phase")
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    left = _camera_source(args.cameras[0])
    right = _camera_source(args.cameras[1])

    _run_phase("phase 1/3: baseline (no viewer)", args.stereo_calib, left, right, args.seconds)

    scene_path = resolve_scene_path(generated=False, wall_seed=None, distractors=None)
    env = SimEnv(str(scene_path), render_mode="viewer")
    env.reset()
    env.launch_viewer()
    time.sleep(0.5)  # let the viewer's background render thread spin up

    try:
        _run_phase(
            "phase 2/3: idle viewer (open, never synced)",
            args.stereo_calib,
            left,
            right,
            args.seconds,
        )

        last_sync = 0.0

        def _tick() -> None:
            nonlocal last_sync
            now = time.monotonic()
            if now - last_sync >= _SYNC_INTERVAL_S:
                env.sync_viewer()
                last_sync = now

        _run_phase(
            "phase 3/3: synced viewer (~30 Hz sync)",
            args.stereo_calib,
            left,
            right,
            args.seconds,
            on_tick=_tick,
        )
    finally:
        env.close()

    print(
        "\nCompare the three 'fresh fps' numbers above (each phase's sensor-health log "
        "line). Big drop 1->2: the viewer's own render thread is the culprit. Further "
        "drop 2->3: sync_viewer() traffic adds on top of it. No drop at all: look "
        "elsewhere (camera/lighting/calibration) for the ~20 fps embedded figure."
    )


if __name__ == "__main__":
    main()
