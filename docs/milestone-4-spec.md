# Milestone 4 — Expert + Data Generation Online

**Goal**: turn the M3 plumbing into an **unattended data factory**. M3 left a
loop that runs end-to-end with a crude scripted human and a zero-Δ `NoAssist`.
M4 fills that loop with *behaviour* and *recording*: a realistic noisy operator,
a closed-form **analytical expert** that supplies the correction Δ from
privileged geometry, per-episode **coverage randomization**, and a
**data-generation driver** that logs structured per-step rows to disk. The
deliverable is the **behavioral-cloning training corpus** M5 trains against.

This is the milestone where the project stops being scaffolding. By the end of
it we can run hundreds of episodes overnight and wake up to a directory of
trajectory files, each one a complete `(observation, expert Δ, privileged
state, success)` record. Nothing here learns yet — the expert *cheats* (it reads
true poses) — but everything needed to teach a non-cheating policy is now on
disk. M4 closes the **Foundation** phase and is the substrate for Design Review
D1.

The contract M4 must keep stable for the future is narrow: **the on-disk
trajectory schema**. Everything else (noise magnitudes, gate constants, scene
layout) is allowed to move; M5 only depends on the columns being there and
meaning what the schema says.

## Definition of done

By the end of M4 we can:

- Run a `scripts/` data-generation driver that executes **N episodes
  unattended** (noisy-human → expert → controller → sim) and writes **one
  trajectory file per episode**, producing **hundreds** of files in one run.
- Drive each episode with the **realistic** `ScriptedNoisyHuman` — biased,
  drifting, low-frequency operator noise — instead of M3's per-step Gaussian
  jitter, fully reproducible from `(master_seed, episode_index)`.
- Inject the **analytical expert** through the M3 seam in place of `NoAssist`,
  with **no** change to the runner, the input strategy, or the controller — and
  watch it visibly improve seating in a headless spot-check.
- **Randomize coverage** per episode: target/distractor holes, initial peg
  offset, and the noisy-human noise pattern, all derived deterministically from
  the episode seed.
- Detect each episode's **terminal condition** (insertion depth → success,
  force-cap → abort, timeout → failure) and stamp **per-episode and per-step
  `success` flags** into the log, **keeping all episodes** (failures included).
- Spot-check a handful of episodes and confirm they are visually sane and that
  the logged privileged ground truth lines up with the logged sensors.

## What's in M4

- **Realistic `ScriptedNoisyHuman`** (LAB-39, `input/`) — structured
  low-frequency noise: per-episode position/orientation **bias**, correlated
  **drift**, optional **tremor**, ~5–10 Hz refresh held to the control rate,
  grip release at episode end. Replaces the M3 stub behind the same
  `InputStrategy` interface.
- **Coverage randomization** (LAB-40, `sim/`) — per-episode randomization of the
  reset state: target hole index, hole layout (via `scenegen`), initial peg
  offset, noise seed — all keyed off `(master_seed, episode_index)`.
- **Analytical privileged-info expert** (LAB-27, `expert/`) — closed-form
  align-then-advance geometric law as an `AssistProvider`, with the
  far-field-zero distance gate `g(d)`. The behavioral-cloning *teacher*.
- **Data-generation rollout + trajectory schema** (LAB-28, `scripts/` + `data/`)
  — wraps `run_episode` with structured per-step logging to disk, terminal-
  condition detection, the keep-all-episodes policy, and deterministic
  per-episode seeding. The schema is the stable contract M5 reads.

## What's not in M4 — explicit anti-scope

- **Image rendering into the pipeline.** The schema reserves a wrist-camera
  column, but actual frame *rendering* + decimation into the corpus is **M7**.
  Phase 1 (M5) trains on F/T + proprioception + command only; M4 logs no pixels.
- **The dataset loader / windowing.** Turning these flat per-step rows into
  batched, windowed training tensors (`H_c×7` command history, `H_f×6` F/T
  history, …) is **M5** (LAB-32). M4 logs *flat* per-step rows; it does not
  assemble histories.
- **The learned policy.** M4 ships only the analytical expert; the BC-trained
  residual `AssistProvider` is **M5+**.
- **The evaluation harness / KPIs.** Trial-level KPIs computed by a *passive,
  non-privileged* observer are **M6** (LAB-36). M4's terminal-condition
  detection is a *privileged* convenience inside the data-gen driver (it reads
  true poses), used only to label episodes and stop the loop — it is **not** the
  eval harness and the controller stays mode-less (`project-scope.md` *Runtime
  state*).
- **Real input devices** (MediaPipe vision, keyboard) → **M8**.
- **Drag / themed environments, vision domain randomization** → M7 / stretch.
- **DAgger / expert-action noise injection for recovery.** Held in reserve
  (`docs/design/problem-structure.md`); only adopted if open-loop BC
  underperforms in M5/M6. M4 ships clean expert labels.

## Design — the four pieces

### 1. Realistic noisy human (LAB-39)

Per `docs/design/human-generation.md`. The command stream `c_t` is a composition
of three layers, **fully determined by `(master_seed, episode_index)`**:

```
c_t = coarse_move_toward(g)  ⊕  drift_t  ⊕  tremor_t
g   = (p_hole + bias_episode,  R_hole · ΔR_bias_episode)     # fixed per episode
```

- **Intent / bias** — `bias_episode ~ N(0, σ_bias²)` (a few mm–cm) drawn **once
  per episode**, plus a constant angular bias. This is a *systematic*,
  consistent misjudgement of where the hole is — it does **not** resample each
  step. This consistency is what makes the correction problem non-trivial.
- **Trajectory** — a capped-rate proportional ("point-and-push") move from the
  current commanded pose toward the biased goal `g`. Deliberately
  contact-unaware: it will keep pushing the peg into flat wall if `g` is off the
  hole — exactly the situation the assist must rescue.
- **Noise** — correlated **drift** (Ornstein–Uhlenbeck or band-limited /
  smoothed random walk) on position + orientation, wandering coherently over
  hundreds of ms; optional small high-frequency **tremor**. The command target
  refreshes at **~5–10 Hz** and is held / interpolated to the control rate,
  mimicking discrete human intents under a fast controller.
- **Grip** — baseline-closed; an optional open/release event at episode end. No
  micro-modulation (that is the expert's / policy's job).

**The trap we reject**: per-step i.i.d. Gaussian noise (M3's stub) makes the
expert's optimal correction *exactly the negative of the injected noise* — a
trivial denoising task that learns nothing. M4 freezes the *form* (biased +
drifting + coarse); the noise **magnitudes** (`σ_bias`, drift time-constant and
amplitude, tremor) stay placeholders calibrated post-baseline against the
human-only failure rate.

### 2. Coverage randomization (LAB-40)

Today `SimEnv.reset()` always loads the fixed `home` keyframe — **no
randomization exists**, so this is new work on the sim/reset path. Per episode,
derive from `(master_seed, episode_index)`:

- **Target / distractor holes** — randomize the active target hole index and,
  where affordable, the hole layout. The procedural `sim/scenegen` generator
  already emits multi-hole walls; reuse it rather than hand-placing.
- **Initial peg offset** — perturb the grasped-peg / starting EE pose within a
  bounded range so approaches start from varied positions, not one canonical
  config. Diverse starting states are what give BC its coverage.
- **Per-episode seed** — one seed derives the scene, the peg offset, **and** the
  noisy-human noise pattern. No global mutable RNG state leaks across episodes.

**Cost tradeoff** (a known unknown): randomizing hole *geometry* means
regenerating + recompiling the MuJoCo model (scenegen emits MJCF), which is not
free. The cheapest in-reach randomization — target index + peg offset + noise
seed against a fixed multi-hole wall — lands first; per-episode (or per-batch)
geometry regen is layered on if the compile cost is acceptable. `reset()` must
stay **deterministic given the episode seed**, and randomization ranges are
config (constructor args / YAML).

`Observation`'s privileged channels (`peg_pose`, `hole_poses`,
`target_hole_index`) must remain correct after randomization — a logged
spot-check must line up.

### 3. The analytical expert (LAB-27)

Per `docs/design/expert-corrections.md`. A deterministic state-feedback law: in,
the privileged true state `s_t` + the noisy command `c_t`; out, a clamped
correction `Δ*_t` with the **same signature** as the future policy. It is the BC
*teacher*, not a research contribution.

**Geometry, per step.** With true peg tip `p_tip`, peg long-axis `a`, hole entry
`p_hole`, insertion axis `n`:

```
e      = p_hole − p_tip          # tip→hole position error
e_ax   = (e · n) · n             # component ALONG the insertion axis
e_lat  = e − e_ax                # component LATERAL to the axis (kill this first)
d      = ‖e‖                     # approach distance (drives the gate g)
```

**Align-then-advance phased law** (push in only once aligned, else the peg jams
on the rim):

1. **Lateral alignment** — desired lateral shift `= e_lat` (move the tip onto
   the hole axis line).
2. **Angular alignment** — `R_align` = smallest rotation taking `a` onto `n`
   (off-axis pitch/roll only; yaw irrelevant for a round peg);
   `R_des = R_align · R_cmd`.
3. **Axial advance — gated by alignment** — advance along `−n` at a capped speed
   **only when** `‖e_lat‖ < ε_lat` and angular error `< ε_ang`; otherwise
   suppress axial motion. The chamfer + lateral compliance of the M2 backbone do
   the final rim-guided seating physically.
4. **Grip** — hold baseline; **reduce** on a detected jam signature; restore once
   seated.

**Residual on the command, not an absolute pose:**

```
Δ_full = pose_des ⊖ c_t
  .position    = pose_des.position − c_t.position
  .orientation = log( R_des · R_cmdᵀ )      # axis-angle, world-frame convention apply_delta expects
  .grip        = grip_des − c_t.grip
```

**Distance gate `g(d)` — far-field zero by construction:**

```
Δ*_t = clamp(  g(d) · Δ_full  )
g(d) = 0                              for d ≥ d_far
     = smoothstep(d_far, d_near, d)   for d_near < d < d_far   # Hermite 3x²−2x³, C¹
     = 1                              for d ≤ d_near
```

`g(d_far) = 0` **exactly and by construction**, so the expert's far-field
correction is *structurally* zero, not approximately zero — matching what the
deployed policy can support (F/T ≈ 0 in free space, no exteroception in Phase 1).
The final `clamp` uses the **same** residual-interface bounds as the policy
(`±2 cm / ±10° / ±5 N` per step) via `domain.clamp_delta`.

**Convention hazard** (called out in M3's handoff and the seam wiki):
`Δ_full.orientation` must be expressed in the **world-frame** convention
`apply_delta` composes with (left-multiply). Get it wrong and the correction
rotates the wrong way. Unit-test a known rotation through `apply_delta`.

**Geometry the expert needs that isn't handed to it directly:**

- `p_tip` is the *tip* of the peg, not the peg body origin in `Observation.peg_pose`
  — offset the body pose along the peg long-axis by the half-length read from the
  model. (Known unknown: confirm the peg geometry / which body axis is `a`.)
- `n` and `p_hole` come from `Observation.hole_poses[target_hole_index]` — the
  hole *site* frame. Which site axis points *into* the hole is a convention to
  confirm (see `scripts/probe_cadquery_hole_frame.py` for prior probing).

### 4. Data-generation rollout + trajectory schema (LAB-28)

A `scripts/` driver wraps the M3 `run_episode` (which stays logging-free — the
shape M3 deliberately left) and bolts logging around it. For each of N episodes:
seed → randomize scene (LAB-40) → build noisy human (LAB-39) + expert (LAB-27)
→ run the loop, recording one row per control step → detect the terminal
condition → write one file.

**Per-step row schema** (flat; windowing is M5). All poses follow the repo
convention — position in metres, quaternion `(w,x,y,z)`, world frame at the
robot base:

| Group | Fields | Source | Notes |
|---|---|---|---|
| index | `step` (int), `sim_time` (s) | loop / `Observation.sim_time` | |
| F/T | `wrist_ft` (6: Fx,Fy,Fz,Mx,My,Mz) | `Observation.wrist_ft` | **bias-subtracted** (see below) |
| proprioception | `joint_positions` (7), `joint_velocities` (7), `ee_pose` (7), `gripper_width` (1) | `Observation` | `gripper_width` is a **schema gap** today — see below |
| command | `cmd_position` (3), `cmd_quaternion` (4), `cmd_grip` (1) | base `c_t` (pre-Δ) | the operator stream |
| expert label | `delta_position` (3), `delta_orientation` (3 axis-angle), `delta_grip` (1) | expert `Δ*_t` (clamped) | the **BC target** |
| privileged | `peg_pose` (7), `target_hole_pose` (7), `target_hole_index` (int), `d` (1) | `Observation` (analysis only) | never fed to a deployed policy |
| label | `step_success` (bool) | terminal logic | per-step success flag |
| placeholder | `wrist_camera` | — | reserved; rendering is **M7**, logged null/absent in M4 |

**Per-episode metadata** (file-level): `master_seed`, `episode_index`,
`n_steps`, `terminal_reason` ∈ {success, force_abort, timeout},
`episode_success` (bool), target hole index, randomization params, schema
version.

**Terminal conditions** (privileged, computed in the driver — not the
controller):

- **success** — peg tip past the insertion-depth threshold along `−n` below the
  hole entry.
- **abort** — wrist force magnitude exceeds a force cap `F_max` (protect against
  a wedged peg grinding forever).
- **timeout / failure** — step budget reached without success.

**Policies**: **keep all episodes** (failures included — diverse state coverage
helps BC); **deterministic per-episode seeding**; the driver owns all
console/IO, `run_episode` stays pure.

**Two schema gaps this milestone must resolve** (both small, both feed M5):

1. **`gripper_width`** is in the proprioception stream (`problem-structure.md`)
   but **not** in today's `Observation`. M4 adds it to `Observation` /
   `SimEnv.get_observation` (read the finger joint) so the schema column is real.
2. **F/T bias subtraction** — `Observation.wrist_ft` is the *raw* sensor wrench,
   which includes the peg's static gravity load. The schema's `wrist_ft` is
   defined as **bias-subtracted** (contact-only). Compute the bias as the wrench
   measured at reset in free space (before contact) and subtract it in the
   logger (or expose a bias-subtracted channel). Decide ownership (logger vs
   `SimEnv`) during the step; document it in the schema doc.

**File format**: **Parquet** (one file per episode, per-step rows as columns +
per-episode metadata in the file's key-value metadata), chosen because it is
columnar, self-describing, pandas/pyarrow-native, and exactly what the M5 loader
(LAB-32) expects to read. NPZ is the fallback if a Parquet dependency is
unwelcome. The schema is documented in `docs/` (or `data/`'s docstring) as the
**stable M5 contract**.

## Build order (estimated effort in parentheses)

Order chosen so each step is testable against the previous and the heavy
integration (data-gen) comes last. Each step is its own branch → PR → CI →
merge.

### Step 1 — Realistic noise model · LAB-39 (~2–3 h)

File: `src/ai_teleop/input/scripted_noisy_human.py` (rewrite), tests in
`tests/test_scripted_noisy_human.py`.

- Add per-episode `bias_episode` (constant within an episode), a drift process
  (OU / band-limited), optional tremor, and a ~5–10 Hz refresh-and-hold.
- Keep the `InputStrategy` signature; the M3 runner must drive it unchanged.
- **Per-step acceptance**: same `(seed, episode_index)` ⇒ identical stream; bias
  constant within / varying across episodes; drift autocorrelation ≫ white
  noise; `position_noise → 0` recovers the coarse trajectory; `poe check` green.

### Step 2 — Coverage randomization · LAB-40 (~2–3 h)

Files: `src/ai_teleop/sim/scene.py` (reset path) + `scenegen` glue; tests in
`tests/`.

- Add per-episode randomization of target hole index + initial peg offset
  (cheap path first), keyed off the episode seed; layer hole-geometry regen via
  `scenegen` if compile cost allows.
- **Per-step acceptance**: different `episode_index` ⇒ different target/peg start;
  same ⇒ exact reproduction; `Observation` privileged poses stay correct; the
  M3 runner consumes the randomized reset with no `Controller` change.

### Step 3 — `gripper_width` + F/T bias plumbing (~1 h)

Files: `src/ai_teleop/common/observation.py`, `src/ai_teleop/sim/scene.py`.

- Add `gripper_width` to `Observation` and populate it from the finger joint.
- Decide + implement F/T bias subtraction ownership; document the convention.
- **Acceptance**: `get_observation` returns the new field; a reset free-space
  wrench tares to ≈0 after bias subtraction; existing tests still pass. (Folded
  into LAB-28's PR if trivial, or its own small commit.)

### Step 4 — Analytical expert · LAB-27 (~3–4 h)

Files: `src/ai_teleop/expert/` (new module, re-exported from `__init__`); tests
in `tests/test_expert.py`.

- Implement the geometry (tip/hole error split), the phased align-then-advance
  desired pose, the residual `Δ_full = pose_des ⊖ c_t`, the smoothstep gate
  `g(d)`, and the final `clamp_delta`. Reuse `domain` quaternion/clamp helpers —
  no hand-rolled quaternion math.
- **Per-step acceptance**:
  - `isinstance(Expert(...), AssistProvider)` holds; it slots into `run_episode`
    in place of `NoAssist` with **no** runner/input/controller change.
  - Unit test: `g(d) == 0` (hence `Δ* == 0`) across a grid of `d ≥ d_far`
    far-field states — the by-construction far-field-zero property.
  - Unit test: a known mis-alignment yields a Δorientation that, through
    `apply_delta`, rotates the peg axis *toward* `n` (convention check).
  - Headless spot-check (a `scripts/dev/` probe): driven through the seam, the
    expert visibly improves seating vs `NoAssist` (smaller final `d` / deeper
    insertion).

### Step 5 — Data-generation rollout + schema · LAB-28 (~3–4 h)

Files: `scripts/generate_dataset.py` (driver), `src/ai_teleop/data/` (schema +
writer), `docs/` schema note; tests in `tests/`.

- Driver: loop N episodes, each seeded `(master_seed, episode_index)`, compose
  randomized scene + noisy human + expert + controller, call `run_episode` with a
  per-step row callback (or a thin logging wrapper), detect the terminal
  condition, write one Parquet file + per-episode metadata.
- Keep-all-episodes; `run_episode` stays logging-free; all IO in the driver.
- **Acceptance (this is the M4 milestone acceptance)**:
  - A run produces **hundreds** of episode files.
  - The schema is stable + documented; a unit test round-trips a written file and
    asserts the columns / dtypes / metadata.
  - A spot-check script reads a few episodes, confirms they are visually sane and
    that privileged `d`/`peg_pose` line up with the logged sensors (e.g. F/T rises
    as `d → 0`).

## Acceptance criteria

- `uv run python scripts/generate_dataset.py --episodes 200 --out data/runs/<name>`
  runs unattended and writes ~200 trajectory files; rerunning with the same
  `--seed` reproduces them byte-for-meaning (same per-step values).
- The analytical expert swaps in for `NoAssist` in `run_episode` with **no edit**
  to `ScriptedNoisyHuman` or `Controller`, and visibly improves seating headless.
- `g(d) == 0` for `d ≥ d_far` (far-field Δ ≈ 0 by construction) — a unit test.
- The realistic noisy human is deterministic from `(seed, episode_index)` and
  its noise is *correlated*, not per-step white (autocorrelation test).
- Coverage randomization varies target hole + peg start across episodes,
  reproducibly; `Observation` privileged poses stay correct.
- The trajectory schema is documented and a written file round-trips through a
  reader test with the expected columns + metadata.
- `uv run poe check` is green; the M3 runner (`scripts/run_episode.py`),
  the M2 dev harness, and the M1 smoke test still pass — M4 adds layers and
  changes no existing contract except the **additive** `Observation.gripper_width`
  field.

## Total estimated effort

**10–14 hours**, 2–4 sessions, across five PRs. The long poles are the expert
geometry (getting the frames/conventions right) and the data-gen integration
(schema + terminal logic + the two `Observation` gaps). No new tuning marathon —
the noise/gate constants are deliberately left as post-baseline calibration.

## Files this milestone touches

```
src/ai_teleop/input/
└── scripted_noisy_human.py     (rewrite — structured noise model)            LAB-39

src/ai_teleop/sim/
└── scene.py                    (reset randomization + gripper_width + F/T bias) LAB-40 / Step 3

src/ai_teleop/common/
└── observation.py              (add gripper_width field)                      Step 3

src/ai_teleop/expert/
├── __init__.py                 (populate — re-export Expert)                  LAB-27
└── expert.py                   (new — analytical align-then-advance expert)   LAB-27

src/ai_teleop/data/
├── __init__.py                 (populate)                                     LAB-28
└── schema.py / writer.py       (new — trajectory schema + Parquet writer)     LAB-28

scripts/
└── generate_dataset.py         (new — unattended data-gen driver)            LAB-28

docs/
└── (schema note — the stable M5 contract)                                    LAB-28

tests/
├── test_scripted_noisy_human.py (extend — bias/drift/determinism)            LAB-39
├── test_expert.py               (new — gate, conformance, convention)        LAB-27
└── test_dataset.py              (new — schema round-trip)                     LAB-28
```

`src/ai_teleop/control/` is **not** modified — M2's
`Controller.compute(observation, command)` contract is sufficient; the expert
composes *above* the controller through the M3 seam, exactly as `NoAssist` did.

## Known unknowns / things to figure out during M4

- **Peg-tip offset + axis `a`** — confirm the peg half-length and which body axis
  is the long axis, to compute `p_tip` from `Observation.peg_pose`.
- **Hole-site frame convention** — which axis of the hole site is `n` (into the
  hole) and whether `p_hole` is the entry or the bottom. Reuse the
  `probe_cadquery_hole_frame.py` finding; re-probe if unsure.
- **Gate + alignment constants** — `d_near`, `d_far`, `ε_lat`, `ε_ang`, axial
  speed cap: rough values now, calibrated alongside contact/chamfer tuning. The
  *form* (smoothstep, far-field-zero) is fixed; the numbers are not.
- **Jam-detection signature** — the F/T pattern that triggers grip reduction;
  start with a simple lateral-force threshold, refine against logged jams.
- **F/T bias ownership** — does `SimEnv` expose a bias-subtracted channel, or does
  the logger tare at reset? Pick the one that keeps `Observation` honest about
  what a real sensor gives.
- **Scene-regen cost** — whether hole-geometry randomization per episode is
  affordable, or whether to regenerate per batch / use a fixed multi-hole wall.
- **Parquet vs NPZ** — default Parquet; fall back to NPZ if the dependency is
  unwelcome. Decide before writing the writer so the M5 loader targets one format.

## Handoff to Milestone 5

M5 (Phase-1 F/T-only residual) consumes exactly **one** thing from M4: the
**on-disk trajectory schema**. M5 adds:

- The **dataset loader + windowing** (LAB-32) — reads these flat per-step
  Parquet rows and assembles the Phase-1 windowed streams (command history
  `H_c×7`, F/T history `H_f×6` bias-subtracted, proprioception), with the
  expert's `delta_*` columns as the regression target.
- The **BC-trained residual** `AssistProvider` that reproduces the expert's `Δ`
  from non-privileged observation alone, slotting into the same M3 seam the
  expert used here.

M4 does **not** require the noise magnitudes or gate constants to be final — only
the **schema columns + their meanings** need to be stable. That is the
deliverable of this milestone.
