# Recentering Handoff — kevin

Entry point for the teleop **recentering / calibration-feedback** work. The WSL latency
work that preceded it is done (below); start here for the next problem.

## Status (2026-06-23)
- Branch: `feat/lab-74-stereo-teleop-debug` (Linear **LAB-74**).
- **Latency: SOLVED.** Live stereo teleop runs at **1.000× real-time** via catch-up
  substepping in `run_episode` (commit `2e52314`). Sim-time now tracks wall-time, so the
  `sim_time`-anchored gesture timings are no longer stretched; viewer is responsive.
- A debug real-time `print` is still in `run_episode` — **strip before PR**.
- Sibling handoff: `../stereohand/docs/recentering-handoff.md`
  (branch `feat/tracker-event-gate-fps-cap`).

## Problem to solve
With `--input vision`, the operator gets **no feedback about `VisionInput`'s state** — can't
tell if a recenter is happening, whether the clutch is engaged, or where "neutral" is
("no idea if it's even happening").

Key facts (verify against source):
- `recenter` is **off by default in `mirror` mode** (`--recenter` flag) — the open-palm
  recenter pose *is* the mirror driving pose, so it collides. In the run command below,
  nothing recenter-related fires. **Confirm what feedback is actually expected.**
- Clutch = lift the hand out of frame (dropout) to re-anchor.
- `VisionInput` timings are `observation.sim_time`-anchored (now real-time):
  `recenter_hold_s=3.0`, `recenter_lock_s=0.5`, `recenter_move_tol=0.02` (2 cm),
  `dropout_grace_s=0.2`, `lock_delay=0.2`, + a one-euro filter.

## Proposed first step (cheap, high value)
Log `VisionInput` **state transitions** via `get_logger` (clutch engage/release, recenter
arm/progress/lock, force-lock HOLD) so the terminal shows what's happening — no GUI work.
A richer on-screen HUD is harder: kevin **doesn't own a cv2 window** (the camera window is
stereohand's *generic* renderer), so a HUD means feeding kevin state into that renderer
(couples the dependency) or drawing kevin's own overlay.

## Files
- `src/ai_teleop/input/vision_input.py` — **primary.** `VisionInput`: clutch / recenter
  timer / lock, one-euro filter, `WorkspaceCalibration`, modes `mirror|expo|rate`.
- `src/ai_teleop/input/hand_tracker.py` — `StereoHandSource` (adapts stereohand),
  `max_fps`, cv2 window pump (`_WINDOW_PUMP_STRIDE`), `_palm_open_facing` (kevin's own
  copy of the recenter pose test).
- `scripts/run_episode.py` — CLI flags: `--recenter`, `--control-mode`, `--gain`,
  `--max-fps`, `--no-cam-window`, `--stereo-calib`, `--left/--right`.
- `src/ai_teleop/sim/runner.py` — substep loop (+ the debug print).
- `src/ai_teleop/sim/scene.py` — `SimEnv.sync_viewer()` (throttled, main-thread).

## Run it
```bash
kvn episode --input vision --no-force-cap \
  --stereo-calib ../stereohand/stereo_calib.json \
  --left "http://$WIN:8080/0" --right "http://$WIN:8080/1" \
  --max-steps 0 --control-mode mirror      # add --recenter to exercise the gesture
```
Needs the viewer (no `--headless`). `--max-steps 0` = unlimited (Ctrl-C to stop).

## Tooling
- Run tools via venv python (stale shebangs): `.venv/bin/python -m {ruff check, mypy, pytest}`.
- Branch-per-feature → PR → CI; never commit to `master`. Repo cheatsheet:
  `../.ai/src/skills/lib/repos.md`.
