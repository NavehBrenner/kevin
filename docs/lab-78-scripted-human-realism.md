# LAB-78 — Realistic ScriptedNoisyHuman command dynamics

**Issue:** [LAB-78](https://linear.app/naveh-brenner/issue/LAB-78) · **Milestone:** M7 (prerequisite) ·
**Blocks:** LAB-77 (difficulty recalibration) → LAB-82 (vision corpus).
**Companion design doc:** [`design/human-generation.md`](design/human-generation.md) (the form this refines).

Implementation spec — written to be picked up cold in a new session. Grounded against
the measured gap from `scripts/dev/compare_human_vs_scripted.py` (64 recorded real-human
episodes vs scripted `dataset_1`) and the current `input/scripted_noisy_human.py`.

---

## 1. Why

The scripted operator's **command stream** is structurally unlike a real operator's. The
command is a live policy input (the command-history GRU), so this is a covariate-shift gap
that bites **M7 vision** specifically — the wrist camera localizes the hole *during the
approach*, which the scripted command skips. Measured, apples-to-apples (both pre-assist):

| metric | real human | scripted | 
|---|---|---|
| net command travel | **442 mm** | 8 mm |
| fraction of steps the command moves | **46 %** | 1.6 % |
| near-field command speed (p90) | 372 mm/s | ~0 |
| median near-field lateral aim error | 18 mm | 13.5 mm |

(Phase-1 F/T was insensitive to this — the action is all at contact. Don't expect M5 to have
been "wrong"; this is an M7 readiness fix.)

## 2. Root cause (current code)

`ScriptedNoisyHuman.get_command()` (`input/scripted_noisy_human.py`) **returns the goal pose
directly** every tick: `command = held(goal + drift)`, `goal = hole + bias`. Two consequences:

1. **No approach phase.** The command is parked at the hole from tick 0 (net travel ≈ the
   bias+drift magnitude, ~8 mm). The *realized* arm approach you see is manufactured entirely
   by the **controller's 2 cm/step command clamp** (see the class docstring, lines 30–34), not
   by the operator. So the command *stream itself* never sweeps in.
2. **Staircase.** Drift is an OU process refreshed at `refresh_hz` (8 Hz) and **held** between
   refreshes (`_hold_steps`, `_refresh_held_target`). Between refreshes the command is
   constant → 1.6 % of steps move.

## 3. Design — new command model

Introduce the missing **coarse approach layer** (already described in
`human-generation.md` §2, "capped-rate proportional move … point-and-push") and make drift
**per-tick** so the command is continuous. The actor integrates a command that *chases* the
drifting biased goal at a capped rate, starting from where the arm actually is:

```
goal              = hole + bias                      # fixed per episode (unchanged)
drift_t           = OU(tau), advanced EVERY control tick   # was: per-refresh, held
target_t          = goal + drift_t
command_0.position = observation.ee_pose[:3]         # seed at the arm's start pose (~base)
step              = target_t - command_{t-1}
command_t.position = command_{t-1} + step * min(1, max_approach_speed * dt / (|step| + eps))
```

i.e. each tick the command moves toward the current drifting goal at up to
`max_approach_speed`, decelerating proportionally inside the last step. This yields:

- **Approach phase** — command starts ~400 mm out (seeded from `observation.ee_pose`) and
  sweeps to the goal continuously → fixes net-travel + far-field continuity.
- **Near-field continuity** — per-tick OU drift means the command keeps making small moves
  even after arrival (no hold) → fixes `moving_frac` near contact.
- **Aim** — terminal command ≈ `goal + drift`; bump bias σ to hit ~18 mm (§5).

Orientation: the rig is **position-only** (mirror off) and the measured rotation rate already
matches, so orientation can keep the current `goal_quaternion + drift` behavior. Optional: give
it the same capped-rate slerp toward the goal for symmetry — not required for acceptance.

Determinism is preserved: the stream is still fully determined by `seed`; the only new input is
`observation.ee_pose` on the first tick (the arm's deterministic reset pose).

## 4. Implementation changes

**`src/ai_teleop/input/scripted_noisy_human.py`**
- Add constructor param `max_approach_speed: float = 0.35` (m/s; starting value, tune in §5).
- Replace the per-refresh hold (`_hold_steps`, `refresh_hz`/`control_hz` hold logic) with a
  **per-tick OU update**: `beta = exp(-dt_control / drift_tau)`, `innovation = sqrt(1 - beta²)`,
  advance `_drift_position` (and `_drift_orientation`) every `get_command` call. (`dt_control =
  1 / control_hz`.) Keep `drift_tau`, `drift_position_std` semantics — stationary σ unchanged.
- Add command-integrator state `self._command_position`, seeded lazily on the first
  `get_command` from `observation.ee_pose[:3]` (a `self._initialized` flag). `observation` stops
  being unused — drop the `# noqa: ARG002`.
- `get_command`: advance drift → `target = goal + drift` → capped-rate move of
  `self._command_position` toward `target` → return `Command(self._command_position.copy(), …)`.
- Keep `tremor_std` (still per-tick, additive on the returned position) and `bore_aligned_grasp`
  unchanged.
- Update the module docstring (the "M2 controller clamp turns the full-goal command into the
  approach" note is now obsolete — the actor owns the approach).

**`src/ai_teleop/data/generate.py`** (`ScriptedNoisyHuman(...)`, ~line 268) and
**`docs/cli.md`** — thread `max_approach_speed` through the data-gen config/CLI like the other
noise params (so the calibration in LAB-77 can sweep it if needed). Bump the default
`position_bias_std` per §5.

**Note on dev scripts:** several `scripts/dev/*` build `ScriptedNoisyHuman(position_bias_std=
0.012, …)`. `max_approach_speed` has a default, so they keep working; no edits required.

## 5. Tuning targets (the fit loop)

`scripts/dev/compare_human_vs_scripted.py` **is the acceptance test.** Regenerate a small
scripted set, then iterate two knobs until the scripted distributions overlap the 64-episode
recorded ones:

- `max_approach_speed` → match `net_cmd_disp_mm` (~440), `moving_frac` (~0.46), `near_speed_*`.
  Start 0.35 m/s.
- `position_bias_std` (and/or `drift_position_std`) → match `cmd_tip_lat_near_med_mm` (~18 mm,
  IQR ~11–26). Start by bumping `DEFAULT_POSITION_BIAS_STD` 0.010 → ~0.015 and check.

Keep the difficulty operating point honest — these are *realism* knobs; the task-geometry
difficulty (chamfer / expert authority) is LAB-77's job, calibrated **after** this lands.

## 6. Test (one runnable check)

`tests/test_scripted_noisy_human_realism.py` — assert the three behaviors that would silently
regress (no framework beyond pytest):

1. **Approach exists:** first command position ≈ `observation.ee_pose[:3]` (within a tick's
   travel), and is **far** from `goal` (≫ chamfer band); a late command is **near** `goal`.
2. **Continuity:** over a representative rollout, the fraction of ticks with non-zero command
   movement is high (e.g. > 0.4) and each per-tick step ≤ `max_approach_speed * dt` — no holds,
   no jumps.
3. **Determinism:** two actors with the same `seed` and the same observation sequence emit
   identical command streams.

## 7. Risks / notes

- **Invalidates existing datasets.** `dataset_0/1` and any M5 corpus were generated with the
  old operator; they're git-ignored and regenerable. Regenerate before LAB-77/82. M5 results
  are unaffected as a deliverable (Phase-1 is F/T, insensitive to this) — no M5 rerun required
  unless you want refreshed numbers.
- **Precedes LAB-77 by design:** changing the operator shifts every success rate, so the
  difficulty recalibration must measure against this final operator (see the issue chain).
- **Eval pairing intact:** the actor stays seedable and open-loop, so paired-seed comparisons
  ([`design/evaluation-protocol.md`](design/evaluation-protocol.md)) still hold.

## 8. Out of scope

- DAgger / expert-relabeling of the recorded episodes (deferred; the recordings carry
  privileged poses so it stays available later).
- Orientation realism beyond the current behavior (rig is position-only).
- Task-geometry difficulty calibration (LAB-77).
