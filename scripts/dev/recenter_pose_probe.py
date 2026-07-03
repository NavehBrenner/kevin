"""Per-frame breakdown of why the open-palm recenter / centering hold never completes.

LAB-90 follow-up. Both stereohand's demo (``--recenter``) and kevin's startup centering
(``calibrate_neutral``) gate on the same two per-frame tests over the triangulated metric
landmarks — >=3 fingers extended (tip->wrist > 1.4x knuckle->wrist) and palm squareness
(|normal_z| > 0.7*|normal|) — plus a hold that restarts whenever the palm drifts >2 cm from
the hold anchor, with a short grace window for dropouts. Any of the three failing chronically
produces the same symptom ("the countdown never finishes", or never starts), and none of them
is visible in the preview window or the fps counter.

This probe opens the exact same tracker path as ``run_episode --input vision`` (no renderer)
and, while the operator holds an open palm square to the cameras, tallies per fresh
triangulated frame which sub-test failed, measures palm jitter against the 2 cm tolerance,
and replays the calibrate_neutral hold state machine — reporting every restart with its
reason. It also prints the negotiated camera mode next to the calibration's image size:
rectification maps assume the calibration resolution, and a silently different negotiated
mode makes triangulated *depth* garbage while 2-D tracking still looks perfect.

    cd kevin && uv run python scripts/dev/recenter_pose_probe.py --cameras 2 1 --seconds 20

Hold an open palm facing the cameras, as still as you can, for the whole run.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# Same landmark indices / thresholds as ai_teleop.input.hand_tracker._palm_open_facing and
# stereohand.renderer._palm_open_facing (they are copies of each other).
_WRIST = 0
_PALM_CENTER = 9  # what the demo's move-tolerance tracks; kevin tracks the wrist
_INDEX_MCP = 5
_PINKY_MCP = 17
_FINGERTIPS = (8, 12, 16, 20)
_FINGER_MCPS = (5, 9, 13, 17)
_EXTENDED_RATIO = 1.4
_SQUARENESS_MIN = 0.7
_HOLD_S = 3.0
_MOVE_TOL_M = 0.02
_POSE_GRACE_S = 0.3  # kevin's calibrate_neutral default (the demo uses 0.4)


def _camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _pose_breakdown(landmarks: np.ndarray) -> tuple[int, float]:
    """(extended finger count, squareness ratio |normal_z|/|normal|) for one frame."""
    wrist = landmarks[_WRIST]
    extended = sum(
        np.linalg.norm(landmarks[tip] - wrist)
        > _EXTENDED_RATIO * np.linalg.norm(landmarks[mcp] - wrist)
        for tip, mcp in zip(_FINGERTIPS, _FINGER_MCPS, strict=True)
    )
    normal = np.cross(landmarks[_INDEX_MCP] - wrist, landmarks[_PINKY_MCP] - wrist)
    norm = float(np.linalg.norm(normal))
    squareness = abs(float(normal[2])) / norm if norm > 0 else 0.0
    return int(extended), squareness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", default="../stereohand/stereo_calib.json")
    parser.add_argument(
        "--cameras", nargs=2, default=["2", "1"], metavar=("LEFT", "RIGHT"), help="camera sources"
    )
    parser.add_argument("--max-skew", type=float, default=0.02, help="StereoCapture max_skew_s")
    parser.add_argument("--seconds", type=float, default=20.0, help="how long to measure")
    args = parser.parse_args()

    import cv2
    import stereohand
    from stereohand import StereoCalibration, StereoHandTracker

    print(f"stereohand from: {stereohand.__file__}")
    calibration = StereoCalibration.load(args.calib)
    print(
        f"calibration: image_size={calibration.image_size} rms={calibration.rms:.3f}px "
        f"baseline={calibration.baseline * 100:.1f}cm"
    )

    left = _camera_source(args.cameras[0])
    right = _camera_source(args.cameras[1])
    tracker = StereoHandTracker.open(
        calibration, left=left, right=right, max_skew_s=args.max_skew, render=False
    )
    # Negotiated camera modes vs the calibration's image size (private access — dev probe).
    # A mismatch silently corrupts rectification (the remap tables index the wrong pixels),
    # which shows up as garbage *depth* while per-view 2-D detection still works.
    capture = tracker._capture  # noqa: SLF001
    for name, cam_thread in (("left", capture._left), ("right", capture._right)):  # noqa: SLF001
        vc = cam_thread._capture  # noqa: SLF001
        code = int(vc.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
        size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        match = "OK" if size == tuple(calibration.image_size) else "**MISMATCH vs calibration**"
        print(
            f"{name}: negotiated {size[0]}x{size[1]} fourcc={fourcc!r} "
            f"driver fps={vc.get(cv2.CAP_PROP_FPS):.1f} -> {match}"
        )

    print(f"\nHold an open palm facing the cameras, still, for {args.seconds:.0f}s...\n")

    # Aggregates.
    fresh = 0
    absent_reads = 0
    total_reads = 0
    pose_pass = 0
    extended_hist: dict[int, int] = {}
    squareness_values: list[float] = []
    wrist_positions: list[np.ndarray] = []
    step_moves_mm: list[float] = []  # frame-to-frame wrist displacement
    skews_ms: list[float] = []
    prev_landmarks: np.ndarray | None = None
    prev_wrist: np.ndarray | None = None

    # calibrate_neutral replay (kevin semantics: raw wrist, grace on pose loss, restart on move).
    hold_start: float | None = None
    hold_anchor: np.ndarray | None = None
    last_good: float | None = None
    restarts: list[str] = []
    completions = 0

    # Absence attribution (sampled every poll): is the capture layer even delivering
    # synced pairs, and when it is, which view's MediaPipe detector is missing the hand?
    # These read the tracker's async debug attributes — sampling, not exact counts.
    pair_ts_prev = 0.0
    pairs_seen = 0
    lm_samples = 0
    lm_left_missing = 0
    lm_right_missing = 0
    skew_samples = 0
    skew_rejects = 0

    start = time.monotonic()
    last_status = start
    deadline = start + args.seconds
    try:
        while time.monotonic() < deadline:
            reading = tracker.read()
            now = time.monotonic()
            total_reads += 1

            pair_ts = capture.latest_pair_timestamp()
            if pair_ts > pair_ts_prev:
                pair_ts_prev = pair_ts
                pairs_seen += 1
                if capture.last_skew_s is not None:
                    skew_samples += 1
                    if capture.last_skew_s > args.max_skew:
                        skew_rejects += 1
            landmarks_2d = tracker.last_landmark_2d
            if landmarks_2d is not None:
                lm_samples += 1
                if landmarks_2d[0] is None:
                    lm_left_missing += 1
                if landmarks_2d[1] is None:
                    lm_right_missing += 1

            new = reading.present and (
                prev_landmarks is None or not np.array_equal(reading.landmarks, prev_landmarks)
            )
            if not reading.present:
                absent_reads += 1
            good_pose = False
            if new:
                prev_landmarks = reading.landmarks
                fresh += 1
                landmarks = reading.landmarks
                extended, squareness = _pose_breakdown(landmarks)
                extended_hist[extended] = extended_hist.get(extended, 0) + 1
                squareness_values.append(squareness)
                good_pose = extended >= 3 and squareness > _SQUARENESS_MIN
                if good_pose:
                    pose_pass += 1
                wrist = landmarks[_WRIST].copy()
                wrist_positions.append(wrist)
                if prev_wrist is not None:
                    step_moves_mm.append(float(np.linalg.norm(wrist - prev_wrist)) * 1000)
                prev_wrist = wrist
                if capture.last_skew_s is not None:
                    skews_ms.append(capture.last_skew_s * 1000)

            # Hold state machine, evaluated every poll like calibrate_neutral does.
            if new and good_pose:
                last_good = now
                assert prev_wrist is not None
                moved = (
                    hold_anchor is not None
                    and float(np.linalg.norm(prev_wrist - hold_anchor)) > _MOVE_TOL_M
                )
                if hold_start is None:
                    hold_start, hold_anchor = now, prev_wrist.copy()
                elif moved:
                    assert hold_anchor is not None
                    drift_cm = float(np.linalg.norm(prev_wrist - hold_anchor)) * 100
                    restarts.append(
                        f"t={now - start:5.1f}s  moved {drift_cm:.1f}cm > {_MOVE_TOL_M * 100:.0f}cm"
                    )
                    hold_start, hold_anchor = now, prev_wrist.copy()
                elif now - hold_start >= _HOLD_S:
                    completions += 1
                    print(f"t={now - start:5.1f}s  HOLD COMPLETED (centering would succeed here)")
                    hold_start, hold_anchor = now, prev_wrist.copy()  # keep measuring
            elif hold_start is not None and last_good is not None:
                if now - last_good > _POSE_GRACE_S:
                    restarts.append(
                        f"t={now - start:5.1f}s  pose lost > {_POSE_GRACE_S}s grace "
                        f"(dropout or pose-test failure)"
                    )
                    hold_start, hold_anchor = None, None

            if now - last_status >= 1.0:
                last_status = now
                held = 0.0 if hold_start is None else now - hold_start
                z = wrist_positions[-1][2] if wrist_positions else float("nan")
                print(
                    f"t={now - start:5.1f}s  fresh={fresh:4d}  pose_pass="
                    f"{(100 * pose_pass / fresh) if fresh else 0:5.1f}%  "
                    f"hold={held:4.1f}s  restarts={len(restarts)}  wrist_z={z:+.3f}m  "
                    f"lm_miss L={100 * lm_left_missing / lm_samples if lm_samples else 0:.0f}% "
                    f"R={100 * lm_right_missing / lm_samples if lm_samples else 0:.0f}%"
                )
            time.sleep(0.005)
    finally:
        tracker.close()

    elapsed = time.monotonic() - start
    print(f"\n=== summary over {elapsed:.1f}s ===")
    print(
        f"fresh frames: {fresh} ({fresh / elapsed:.1f} fps) | absent reads: "
        f"{100 * absent_reads / total_reads if total_reads else 0:.0f}% of {total_reads} polls"
    )
    print(
        f"capture: {pairs_seen} new frame arrivals ({pairs_seen / elapsed:.1f}/s); "
        f"skew>tolerance on {100 * skew_rejects / skew_samples if skew_samples else 0:.0f}% "
        f"of {skew_samples} sampled reads"
    )
    if lm_samples:
        print(
            f"per-view detection missing (sampled): left {100 * lm_left_missing / lm_samples:.0f}%"
            f"  right {100 * lm_right_missing / lm_samples:.0f}%"
        )
    if skews_ms:
        print(f"pair skew: mean {np.mean(skews_ms):.1f}ms  p95 {np.percentile(skews_ms, 95):.1f}ms")
    if fresh:
        print(f"pose test pass rate: {100 * pose_pass / fresh:.1f}%")
        print(f"extended-finger histogram (need >=3): {dict(sorted(extended_hist.items()))}")
    if squareness_values:
        sq = np.array(squareness_values)
        print(
            f"squareness |n_z|/|n| (need > {_SQUARENESS_MIN}): "
            f"mean {sq.mean():.2f}  p10 {np.percentile(sq, 10):.2f}  "
            f"frames>thresh {100 * float(np.mean(sq > _SQUARENESS_MIN)):.0f}%"
        )
    if wrist_positions:
        positions = np.array(wrist_positions)
        std_mm = positions.std(axis=0) * 1000
        print(
            f"wrist position: mean {np.round(positions.mean(axis=0), 3).tolist()}m  "
            f"std/axis [x y z] = [{std_mm[0]:.1f} {std_mm[1]:.1f} {std_mm[2]:.1f}]mm"
        )
    if step_moves_mm:
        moves = np.array(step_moves_mm)
        print(
            f"frame-to-frame wrist jump: p50 {np.percentile(moves, 50):.1f}mm  "
            f"p95 {np.percentile(moves, 95):.1f}mm  max {moves.max():.1f}mm "
            f"(hold restarts at {_MOVE_TOL_M * 1000:.0f}mm drift from anchor)"
        )
    print(f"hold completions: {completions}  restarts: {len(restarts)}")
    for line in restarts[:30]:
        print(f"  {line}")
    if len(restarts) > 30:
        print(f"  ... and {len(restarts) - 30} more")

    if fresh and pose_pass / fresh < 0.5:
        print(
            "\n=> pose test is the blocker: check which sub-test fails above "
            "(extended count -> detection quality; squareness -> depth geometry/calibration)."
        )
    elif any("moved" in r for r in restarts) and completions == 0:
        print(
            "\n=> jitter is the blocker: triangulated position wanders past the 2cm tolerance "
            "even for a still hand (depth noise -- check skew, baseline, calibration match)."
        )


if __name__ == "__main__":
    main()
