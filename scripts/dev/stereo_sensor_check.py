"""Probe the stereo HandReading the kevin teleop path actually sees (LAB-74 debug).

Runs StereoHandSource exactly as `kvn episode --input vision --stereo-calib` does,
but with no robot — just prints each reading so we can tell *why* the arm isn't
moving:

- present=False every tick  -> the hand isn't being triangulated in BOTH views
  (lighting, hand not in shared FoV, stale/wrong calibration, or the bg thread
  hasn't produced a frame yet). VisionInput holds the pose -> arm looks frozen.
- present=True but position barely changes -> sensor is live; look at mapping/scale.
- present=True and position moves with your hand -> sensor + conversion are fine;
  the problem is downstream (controller / command), investigate there.

Usage (same URLs/calib you pass to kvn):
    cd kevin
    .venv/bin/python scripts/dev/stereo_sensor_check.py \
        --calib ../stereohand/stereo_calib.json \
        --left "http://$WIN:8080/0" --right "http://$WIN:8080/1"
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ai_teleop.input.hand_tracker import StereoHandSource


def _source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib", required=True, help="stereohand stereo_calib.json path")
    p.add_argument("--left", default="0", help="left camera (index or stream URL)")
    p.add_argument("--right", default="2", help="right camera (index or stream URL)")
    p.add_argument("--seconds", type=float, default=15.0, help="how long to probe")
    p.add_argument("--hz", type=float, default=10.0, help="print rate")
    args = p.parse_args()

    print(f"opening stereo source: left={args.left!r} right={args.right!r} calib={args.calib}")
    # show_window=False keeps this headless; the bg tracker thread still runs.
    source = StereoHandSource(
        args.calib, left=_source(args.left), right=_source(args.right), show_window=False
    )

    present_count = 0
    total = 0
    period = 1.0 / args.hz
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            reading = source.read()
            total += 1
            if reading.present:
                present_count += 1
                pos = np.array2string(reading.position, precision=3, suppress_small=True)
                print(f"present=True  pos(m)={pos}  open_close={reading.open_close:.2f}")
            else:
                print("present=False  (no hand triangulated in both views)")
            time.sleep(period)
    finally:
        source.close()

    pct = 100.0 * present_count / total if total else 0.0
    print(f"\n{present_count}/{total} frames present ({pct:.0f}%).")
    if present_count == 0:
        print(
            "Diagnosis: the sensor never produced a hand. Verify stereohand alone first:\n"
            "  cd ../stereohand && .venv/bin/python scripts/demo.py "
            f"--calib {args.calib} --left {args.left!r} --right {args.right!r}\n"
            "If the 3D skeleton doesn't track there either, it's cameras/lighting/"
            "calibration, not kevin."
        )


if __name__ == "__main__":
    main()
