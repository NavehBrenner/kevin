"""Check whether the demo's *visual* smoothness hides the same raw drop-out kevin measures
(LAB-74 debug, continued from `viewer_gil_probe.py` / `poll_rate_probe.py`).

Two threading hypotheses (the MuJoCo viewer's render thread; kevin's poll frequency) were
each cleanly falsified by controlled measurement, and the camera pair itself was ruled out
(the demo tracks "almost every frame" with the identical `--left/--right` args every kevin
probe has used). That leaves one variable no probe so far has controlled for: every prior
probe ran with `show_window=False` (no visual feed) so the cv2-window cost couldn't confound
the numbers - but that also means none of them let a human *watch* the tracking while the
sensor-health counters ran.

`stereohand`'s renderer applies an EMA (`RenderConfig(smooth=...)`, default 0.5) to the
*displayed* skeleton, and per `project-wiki/concepts/temporal-gating.md`, stereo fusion is
all-or-nothing (`present` requires **both** views to detect the hand **and** their capture
timestamps to fall within `max_skew_s`, 20 ms by default) - individually-normal single-frame
MediaPipe misses (see `project-wiki/entities/mediapipe-hands.md`) compound roughly as p^2
once fused, plus outright rejections from capture-timestamp skew. A raw present/absent rate
in the same range this investigation has already measured (~5-8 fresh fps, ~60-80%
drop-out) could very plausibly look smooth on screen - visual persistence and EMA smoothing
hide brief, frequent misses that a human eye doesn't consciously register - while a naive
downstream consumer (kevin's clutch: engage on `present`, release+re-anchor after
`dropout_grace_s`) turns the exact same underlying signal into visibly jerky robot motion.

This runs kevin's own `StereoHandSource` (the same sensor-health counters every prior probe
in this investigation used) with `show_window=True` - the same on-screen feed the demo
shows - so you can *watch* the tracking quality with your own eyes while it logs the same
"fresh fps / drop-out" numbers. If it looks smooth to you but still logs a similar ~5-8 fps
/ ~60-80% drop-out to the earlier runs, that confirms the raw sensor was never actually
different from the demo's - the "fps drop" you originally saw is downstream, not sensor
throughput. If it looks visibly choppy while you watch, or the numbers look genuinely much
better than the earlier runs, that reopens the throughput question.

Press 'q' or ESC in the preview window (or wait out --seconds) to stop.

    cd kevin && uv run python scripts/dev/visible_sensor_health_probe.py \
        --stereo-calib ../stereohand/stereo_calib.json --cameras 2 1 --seconds 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ai_teleop.common.log import add_logging_arguments, configure_from_args  # noqa: E402
from ai_teleop.input.hand_tracker import StereoHandSource  # noqa: E402


def _camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


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
    parser.add_argument("--seconds", type=float, default=15.0, help="how long to watch/measure")
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    left = _camera_source(args.cameras[0])
    right = _camera_source(args.cameras[1])

    print(f"\nWatch the preview window for {args.seconds:.0f}s. Does tracking look smooth?")
    source = StereoHandSource(args.stereo_calib, left=left, right=right, show_window=True)
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            source.read()
            time.sleep(0.002)  # same 500 Hz poll cadence run_episode's control loop uses
    finally:
        source.close()  # logs the sensor-health "fresh fps / drop-out" line


if __name__ == "__main__":
    main()
