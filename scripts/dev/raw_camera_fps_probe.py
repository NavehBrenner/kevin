"""Measure each camera's own raw capture fps independently - no stereo pairing, no
MediaPipe, no stereohand tracker at all (LAB-90 debug follow-up).

After landing the `max_skew_s` fix, both the standalone `stereohand/scripts/demo.py` and
`kevin`'s `run_episode --input vision` regressed again (demo ~60 -> ~30 fps; run_episode back
toward ~20 fps) - even after unplugging/replugging the USB cameras. Neither `kevin` nor
`stereohand` had any code changes when this reappeared (confirmed via `git status`/`git diff`
in both repos), so this isolates the question to the *lowest possible layer*: is each camera,
on its own, actually still delivering frames at its earlier rate?

Opens each camera via `stereohand.capture.open_capture` (the exact same MJPG+DSHOW open path
the real pipeline uses, so this is apples-to-apples) and times raw `cv2.VideoCapture.read()`
calls per camera, on its own thread, with nothing else running - no stereo sync, no skew
tolerance, no MediaPipe. If each camera independently reports the same lower fps here, the
regression is upstream of all of kevin's/stereohand's code (hardware, driver, USB
negotiation, or lighting/auto-exposure). If each camera reports its earlier rate here but the
stereo pipeline still shows the drop, the bottleneck is in pairing/detection instead.

    cd kevin && uv run python scripts/dev/raw_camera_fps_probe.py --cameras 2 1 --seconds 10
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "stereohand" / "src"))


def _camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _fourcc_str(capture: object) -> str:
    import cv2

    code = int(capture.get(cv2.CAP_PROP_FOURCC))  # type: ignore[attr-defined]
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)) or "?"


def _measure(
    name: str, source: int | str, seconds: float, results: dict[str, tuple[int, float]]
) -> None:
    import cv2
    from stereohand.capture import open_capture

    capture = open_capture(source)
    if not capture.isOpened():
        results[name] = (-1, 0.0)
        return
    # What actually got negotiated, not what open_capture() requested (.set() calls can
    # silently fail) - a fallback off MJPG onto raw YUYV is a known way for two USB
    # webcams on one controller to starve each other's bandwidth (see open_capture()'s
    # own docstring in stereohand/capture.py).
    print(
        f"{name}: negotiated fourcc={_fourcc_str(capture)!r} "
        f"res={int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"driver-reported fps={capture.get(cv2.CAP_PROP_FPS):.1f}"
    )
    frames = 0
    start = time.monotonic()
    deadline = start + seconds
    try:
        while time.monotonic() < deadline:
            ok, _frame = capture.read()
            if ok:
                frames += 1
    finally:
        capture.release()
    elapsed = time.monotonic() - start
    results[name] = (frames, elapsed)


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
    args = parser.parse_args()

    left = _camera_source(args.cameras[0])
    right = _camera_source(args.cameras[1])

    print(
        f"Opening left={left!r} and right={right!r} independently for {args.seconds:.0f}s each..."
    )
    results: dict[str, tuple[int, float]] = {}
    threads = [
        threading.Thread(target=_measure, args=("left", left, args.seconds, results)),
        threading.Thread(target=_measure, args=("right", right, args.seconds, results)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for name in ("left", "right"):
        frames, elapsed = results[name]
        if frames < 0:
            print(f"{name}: FAILED TO OPEN")
            continue
        fps = frames / elapsed if elapsed > 0 else 0.0
        print(f"{name}: {frames} frames over {elapsed:.1f}s = {fps:.1f} fps (raw, no processing)")


if __name__ == "__main__":
    main()
