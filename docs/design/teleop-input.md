# Teleop Input — Stereo Hand-Tracking Design

> **History (2026-06-22, LAB-75):** a monocular MediaPipe baseline (LAB-50/51) shipped
> first and was then **removed** in favour of stereo-only input — the project drives the
> arm exclusively from the two-webcam [stereohand](https://github.com/NavehBrenner/stereohand)
> sensor. The monocular rationale is kept below only to explain *why* stereo was worth it;
> there is no longer a single-camera code path or fallback.

Companion docs: [problem-structure.md](problem-structure.md) · [evaluation-protocol.md](evaluation-protocol.md). The authoritative high-level scope is [`../../project-scope.md`](../../project-scope.md); milestone build-order lives in [`../milestones.md`](../milestones.md) (M8). This file pins down *how the human's hand becomes an EE command*, and locks the next step: a second camera for metric, stereo-triangulated hand pose.

The teleop input is **demo-enablement, not a core result** — the KPIs come from the scripted noisy-human, not a live operator (see [evaluation-protocol.md](evaluation-protocol.md)). So this path is allowed to be approximate; the bar is "a person can comfortably drive the arm and complete assisted insertions," not metric fidelity. The stereo upgrade below is what turns "drivable" into "feels like the robot mirrors my hand."

## Layering (locked, unchanged by the upgrade)

Two layers behind the `InputStrategy` seam, and the upgrade touches only the lower one:

- **Sensor** — `input/hand_tracker.py`. Webcam frame(s) → MediaPipe Hands 21 landmarks → a small typed `HandReading` (position, orientation estimate, open/close grip proxy, `present` flag). Pure sensing: no robot, no `Command`, no calibration. Per [`../../project-scope.md`](../../project-scope.md), MediaPipe is "treated as a sensor library," not a contribution.
- **Strategy** — `input/vision_input.py`. `HandReading` stream → base EE `Command`: relative clutched mapping, per-axis scale + axis remap/flip (`WorkspaceCalibration`), one-euro jitter filter, grip mapping, optional orientation. This is where camera-space becomes robot-space.

The seam means the input source is swappable at runtime (`--input {scripted,vision}`) with zero up/downstream change. The stereo upgrade keeps the `HandReading` contract and the strategy intact — it only changes *how well* the reading's position and orientation are estimated.

## Why a single camera wasn't enough (the monocular baseline, now removed)

The first iteration used one webcam. It drove the arm, but two axes are weak by construction — which is exactly what stereo fixes:

- **Depth is a proxy, not metric.** A single camera can't triangulate distance, so `HandReading.position[2]` is an *apparent-hand-size* proxy (wrist→middle-MCP pixel distance — larger ⇒ closer). It gives a usable forward/back axis but it is monotonic-only, not metric: it drifts with hand pose (a tilted hand "shrinks"), and it can't be scaled into real centimeters. MediaPipe's own landmark `z` is worse, so the proxy is the better of two bad options.
- **Orientation is jittery, so 6-DoF is off by default.** The hand-frame quaternion is estimated from a few palm landmarks in a single view; out-of-plane rotation is exactly where a monocular estimate is least observable. `VisionInput(track_orientation=False)` is the default because feeding that signal in tends to fight the controller. The peg being round (roll irrelevant) makes this acceptable for the baseline, but it means the operator can't actually *orient* the peg by hand.

Net: the baseline mirrors hand **translation in the image plane** well, fakes depth, and ignores rotation. That is the gap the second camera closes.

## The upgrade: a second camera → stereo metric hand pose

Add a second webcam, calibrated against the first, and run MediaPipe on both views. The 21 landmarks seen from two known viewpoints **triangulate** to a metric 3D hand skeleton in camera-rig space. That single change fixes both weak axes at once:

- **Depth becomes metric.** Triangulated wrist depth is real distance, in centimeters, with the same accuracy as the in-plane axes — no size proxy, no pose-dependent drift. `WorkspaceCalibration` scale becomes a true unit conversion instead of a hand-tuned fudge.
- **Orientation becomes observable.** A hand frame fit to the *3D* landmark cloud (e.g. wrist→index-MCP and wrist→pinky-MCP spanning the palm, normal = their cross product) is far steadier than the monocular estimate, because out-of-plane rotation is now seen by the second camera. This is what lets us finally turn `track_orientation=True` on and get true **6-DoF mirroring** — the "really good hand movement mirroring" goal.

### Why stereo pair, not an RGB-D camera

A depth camera (RealSense / Azure Kinect) would also give metric 3D and skip calibration. We choose a **second plain webcam** because: it keeps MediaPipe-as-sensor unchanged (still RGB in, landmarks out — no new SDK, no depth-alignment), the triangulation math is ~30 lines of OpenCV, depth cameras struggle with thin/fast-moving fingers at close range anyway (active IR is tuned for room-scale), and two identical cheap webcams is the lower-friction, lower-cost path. The cost is a one-time calibration step. If calibration proves painful in practice, an RGB-D camera is the documented fallback — the `HandReading` contract is identical either way, so the strategy layer never knows the difference.

### What stays the same

The strategy layer (`vision_input.py`) is **unchanged in shape**: same relative clutch, same one-euro filter, same `WorkspaceCalibration`, same grip mapping. Metric depth and trustworthy orientation just flow into the existing transform. The clutch matters *more* here, not less: metric mapping means the operator's reachable hand volume must clutch-tile across the larger robot workspace exactly as before. With metric input, the plain `mirror` mapping (direct 1:1, relative to the clutch anchor) is genuinely usable — so it's the only mapping kept.

### Hardware, sync, and calibration specifics

The decisions that aren't obvious from "add a second camera":

- **Two cameras, our own triangulation — no stereo model.** MediaPipe is monocular; we
  run it per view and triangulate the 21 (trivially-corresponded) landmarks ourselves
  via linear DLT (`cv2.triangulatePoints`). **Linear DLT is enough** — no bundle
  adjustment / iterative refinement unless a measured error demands it. The failure mode
  here is gold-plating the CV, not under-building it.
- **Matched pair, rigidly co-mounted.** Cameras need not be identical for the math
  (triangulation uses each camera's own intrinsics + the extrinsics), but a matched pair
  on one rigid bar is the low-friction path for two reasons: (1) accuracy degrades to the
  *weaker* camera, and (2) — more importantly — **synchronization dominates.** The hand
  moves, so the two frames must be captured near-simultaneously or we triangulate two
  different poses. Best-effort software sync (timestamp both grabs, reject pairs whose
  skew exceeds a threshold) is sufficient given teleop is approximate; hardware sync is
  overkill. Rolling-shutter mismatch between unmatched cameras is the nastiest case.
- **Calibration is two separate things, don't conflate them.** *Camera calibration*
  (new, LAB-54): per-camera intrinsics + lens distortion, then stereo extrinsics —
  **ChArUco over a plain checkerboard** (robust to the partial/occluded board views you
  get with two cameras at an angle) → `cv2.stereoCalibrate` / `cv2.stereoRectify` → P1/P2,
  persisted once; valid until the rig is physically disturbed. *Workspace calibration*
  (already shipped, `WorkspaceCalibration`): camera-rig→robot mapping; unchanged in shape,
  but its scale knob becomes a true cm→robot unit conversion instead of a proxy fudge.
- **Baseline/geometry is the real accuracy knob.** Too-small baseline → poor depth
  resolution; too-wide → less view overlap and MediaPipe may lose the hand in one view.
  Leave this tunable.
- **Single hand only** (`max_num_hands=1`). Two detected hands, or left/right handedness
  flipping *between* views, silently breaks landmark correspondence.
- **Drop out, don't freeze.** Hand missing/low-confidence in *either* view → that tick is
  `present=False` (the clutch + one-euro filter already absorb gaps). There is no monocular
  fallback any more: if a camera is unplugged, stereo simply reports drop-out and the arm
  holds (clutched) until both views see the hand again.

## Build order (the stereo issues)

Three issues, mirroring the sensor/strategy split, sequenced:

1. **Stereo capture + calibration.** Synchronized dual-webcam capture; one-time stereo calibration (checkerboard → intrinsics + extrinsics + rectification), persisted to a config file. Output: a rectified, time-aligned frame pair and the projection matrices. Pure infra — no hand, no MediaPipe.
2. **Stereo hand triangulation → metric `HandReading`.** Run MediaPipe on both rectified views; match the 21 landmarks (same indices, trivially corresponded); triangulate to metric 3D; fit the palm frame. Replaces the size-proxy depth and the monocular orientation estimate inside `hand_tracker.py`, behind the same `HandReading` type. Drop-out = hand missing in *either* view.
3. **Enable 6-DoF mirroring in `VisionInput`.** With metric position and a trustworthy orientation, flip `track_orientation` on by default, re-tune `WorkspaceCalibration` scale to the now-metric input, and confirm orientation mirroring doesn't fight the controller. This is the issue that delivers the "good hand mirroring" payoff.

The deterministic core stays unit-tested (landmark→reading math; palm-frame fit on a known pose); the live two-camera path is manual.

## Status / placement (stereo-only, LAB-74 / LAB-75)

Stereo capture + calibration + triangulation ship in the standalone
[stereohand](https://github.com/NavehBrenner/stereohand) package (metric `(21, 3)` landmarks
out of `StereoHandTracker`). kevin consumes them via:

- `StereoHandSource` in `hand_tracker.py` — adapts `stereohand.StereoHandTracker` to the
  `read() -> HandReading` seam, running `reading_from_landmarks` on the triangulated metric
  landmarks. The sole live sensor (the monocular `MediaPipeHandTracker` was removed).
- `WorkspaceCalibration` (the now-default, metric) — near-1:1 metre→metre scale, depth→world-x
  remap, forward axis sign-flipped (metric depth grows *away* from the camera). The CLI
  enables `track_orientation=True` for the 6-DoF payoff and `recenter=True` for the
  open-palm re-anchor gesture.
- `kvn episode --input vision --stereo-calib <path> --cameras <L> <R>` — `--stereo-calib`
  is required; there is no single-camera mode.

Packaging note: the `stereo-input` extra pulls stereohand, which brings a modern MediaPipe
(Tasks API) + opencv-contrib transitively — the only vision stack now (the old `vision-input`
extra and its legacy `mediapipe==0.10.21` pin are gone). The live two-camera path is manual;
the deterministic core (metric landmark→reading, calibration mapping) is unit-tested.

> Provenance: design locked 2026-06-20 from direct observation of the monocular baseline's behaviour (depth-proxy drift and orientation jitter), then the baseline was removed 2026-06-22 (LAB-75) in favour of stereo-only. Not from a `raw/` source.
