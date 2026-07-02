"""Isolate whether the *polling rate* itself, not the MuJoCo viewer, degrades stereohand's
effective tracking rate when embedded in kevin (LAB-74 debug: native-Windows fps drop,
continued from `viewer_gil_probe.py`).

`viewer_gil_probe.py` already ruled out the viewer's own render thread: fresh fps was
similar (and non-monotonic) whether the viewer was closed, idle, or synced. But even its
"baseline (no viewer)" phase - polling `StereoHandSource.read()` at ~500 Hz, matching
`run_episode`'s physics-rate control loop - showed the same ~80-90% drop-out a live
`kvn episode --input vision` run does, while `stereohand/scripts/demo.py`'s much gentler
~10 Hz *event-driven* loop (wait for a new reading, then process - no fixed-rate busy-poll)
tracks the same hardware/calibration almost every frame.

That's the remaining variable: `run_episode`'s architecture calls the base command source
(and therefore `tracker.read()`) once per physics tick, at up to 500 Hz, deliberately - one
command per physics step is what makes a recorded episode replay tick-for-tick (see
`project-wiki/concepts/realtime-pacing-substepping.md`). A tight fixed-interval poll loop at
that rate, even though each `read()` call is cheap (a lock acquire + cached-object return),
wakes the main thread far more often than the demo's loop does; on Windows, CPython's GIL
reacquisition after each of those wake-ups is not FIFO-fair against a background thread
that's also frequently re-entering/exiting native calls (a documented Windows GIL "convoy"
effect) - so a busier poller can starve the tracker thread's scheduling even though neither
loop is CPU-bound.

Runs the same `StereoHandSource`, no viewer at all, at several fixed poll intervals for
`--seconds` each, and reports each phase's sensor-health line:

  1. 500 Hz  (2 ms)   - matches run_episode's physics-rate control loop.
  2. 100 Hz  (5 ms)   - matches calibrate_neutral's poll_interval_s default.
  3. ~10 Hz  (100 ms) - roughly matches stereohand demo.py's event-driven cadence.

A clear fresh-fps improvement (and drop-out reduction) from phase 1 -> 3 implicates poll
*frequency* itself (not the viewer) as the mechanism; no trend re-opens the question.

Keep a hand visible to both cameras (any pose) throughout all three phases.

    cd kevin && uv run python scripts/dev/poll_rate_probe.py \
        --stereo-calib ../stereohand/stereo_calib.json --cameras 2 1 --seconds 8
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


def _run_phase(
    label: str,
    calib: str,
    left: int | str,
    right: int | str,
    seconds: float,
    poll_interval_s: float,
) -> None:
    """Poll a fresh StereoHandSource at a fixed interval for `seconds`; close() logs sensor health."""
    print(
        f"\n--- {label}: poll every {poll_interval_s * 1000:.0f} ms "
        f"({seconds:.0f}s; keep a hand in view) ---"
    )
    source = StereoHandSource(calib, left=left, right=right, show_window=False)
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            source.read()
            time.sleep(poll_interval_s)
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

    _run_phase(
        "phase 1/3: 500 Hz poll (run_episode's rate)",
        args.stereo_calib,
        left,
        right,
        args.seconds,
        0.002,
    )
    _run_phase(
        "phase 2/3: 100 Hz poll (calibrate_neutral's rate)",
        args.stereo_calib,
        left,
        right,
        args.seconds,
        0.005,
    )
    _run_phase(
        "phase 3/3: ~10 Hz poll (roughly demo.py's cadence)",
        args.stereo_calib,
        left,
        right,
        args.seconds,
        0.1,
    )

    print(
        "\nCompare fresh fps / drop-out across the three phases. Clear improvement from "
        "500 Hz -> ~10 Hz implicates poll *frequency* (Windows GIL scheduling against the "
        "tracker's background thread), not the viewer, as the mechanism."
    )


if __name__ == "__main__":
    main()
