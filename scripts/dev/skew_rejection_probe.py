"""Isolate capture-timestamp skew rejection from MediaPipe detection misses
(LAB-74 debug, continued from `viewer_gil_probe.py` / `poll_rate_probe.py` /
`visible_sensor_health_probe.py`).

Every prior probe in this investigation measured the *fused* present/absent rate, which
conflates two independent causes of a "no reading" instant:

1. **MediaPipe not detecting a hand** in one or both 2D views this frame (normal, even with
   a plainly-visible hand - see `project-wiki/entities/mediapipe-hands.md`).
2. **`StereoCapture.read()` rejecting an otherwise-fine pair purely on timing**: the two
   cameras run on independent, uncoordinated grabber threads (`capture.py`), and a pair is
   only returned if their capture timestamps land within `max_skew_s` (20 ms default) of
   each other - unrelated to whether a hand was visible in either frame at all.

Three probes have now measured a stable ~5-8 fresh fps / ~75-85% drop-out figure regardless
of the MuJoCo viewer, poll rate, or the preview window - all *downstream* of capture. This
probe instead measures **`StereoCapture` alone, with no MediaPipe/landmarker in the loop at
all**: how often two independent 30 fps-ish cameras actually deliver a pair within the skew
window, purely as a hardware/timing question. If skew rejection alone is already high, that
directly implicates `max_skew_s` (an easy, one-line tuning knob) rather than detection
quality, camera hardware, or anything in kevin's control loop.

    cd kevin && uv run python scripts/dev/skew_rejection_probe.py --cameras 2 1 --seconds 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "stereohand" / "src"))


def _camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cameras",
        nargs=2,
        default=["0", "2"],
        metavar=("LEFT", "RIGHT"),
        help="left/right camera sources",
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to measure")
    parser.add_argument(
        "--max-skew-s",
        type=float,
        default=0.02,
        help="capture-pair skew tolerance to test (stereohand's default is 0.02).",
    )
    args = parser.parse_args()

    from stereohand.capture import StereoCapture

    left = _camera_source(args.cameras[0])
    right = _camera_source(args.cameras[1])

    print(
        f"opening cameras left={left!r} right={right!r}, max_skew_s={args.max_skew_s}"
        f" ({args.seconds:.0f}s, no hand needed - this measures raw capture pairing only)"
    )
    capture = StereoCapture(left, right, max_skew_s=args.max_skew_s)
    reads = 0
    accepted = 0
    skews: list[float] = []
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            pair = capture.read()
            reads += 1
            if pair is not None:
                accepted += 1
            if capture.last_skew_s is not None:
                skews.append(capture.last_skew_s)
            time.sleep(0.002)
    finally:
        capture.close()

    accept_rate = 100.0 * accepted / reads if reads else 0.0
    mean_skew_ms = 1000.0 * sum(skews) / len(skews) if skews else float("nan")
    print(f"\n{accepted}/{reads} reads had a within-skew pair ({accept_rate:.0f}% accepted).")
    print(
        f"mean observed skew: {mean_skew_ms:.1f} ms (tolerance was {args.max_skew_s * 1000:.0f} ms)"
    )
    if accept_rate < 50.0:
        print(
            "\nMost reads are being rejected on timing alone, before MediaPipe/detection even "
            "runs. Try a larger --max-skew-s (e.g. 0.05 or 0.1) and re-run to see if the "
            "downstream fresh-fps/drop-out numbers improve to match."
        )


if __name__ == "__main__":
    main()
