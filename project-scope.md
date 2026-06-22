# Project Scope — AI-Assisted Robotic Teleoperation for Precision Insertion

*Course 20973 — Workshop in Autonomous Systems Simulation, fall 2026*
*Author: Naveh — solo project*
*Status: scope draft, pre-proposal*

---

## One-line summary

A simulated robotic arm performs peg-in-hole insertions under **shared-autonomy** control: a human operator provides coarse 6-DoF commands via webcam-tracked hand motion, and a **vision-conditioned residual policy** issues real-time micro-corrections (sub-centimeter pose deltas and grip-force modulation) so insertions reliably succeed even when the human's input is noisy.

## Problem statement

Teleoperated robotic manipulation breaks down at the last few millimeters. Humans are good at coarse motion, poor at fine alignment under uncertain feedback. Operators routinely fail at sub-centimeter contact-rich tasks like inserting plugs, keys, or assembly pieces. This project demonstrates — in simulation — that a learned residual correction policy, conditioned on a wrist-mounted camera and contact force, can close that gap: taking the human's noisy coarse command and adding a small, physics-aware correction that turns failed attempts into successful insertions.

## What we are building

A single integrated MuJoCo simulation in which:

- A Franka Emika Panda manipulator attempts to insert a peg into a fixtured hole on a tabletop.
- The peg's coarse trajectory is driven by a human-issued command stream.
- **Three input modalities** are interchangeable behind a common interface: stereo webcam (two-camera metric hand tracking via the [stereohand](https://github.com/NavehBrenner/stereohand) package, MediaPipe under the hood), keyboard fallback, and a scripted "noisy human" for repeatable benchmarking.
- **Two assistance modes** are interchangeable behind a common interface: no assistance (Δ=0) and the learned residual policy. Both run on the always-on impedance backbone + Δ-clamp / force cap.
- The learned residual policy is **vision-conditioned**: it reads a wrist-mounted RGB camera, plus wrist force/torque, plus proprioception, and outputs bounded pose/grip corrections. It is trained offline via behavioral cloning from a scripted privileged-info expert.

### Build order — two phases

Built and evaluated incrementally to derisk the project:

- **Phase 1 (Option A) — F/T-only residual, contact-reactive alignment.** No camera. The operator's coarse command brings the peg to the hole vicinity (command history encodes which hole is intended); the residual policy uses force/torque + proprioception to *feel the contact and correct alignment* as the peg seats. It cannot localize the hole from afar — its value is the last-millimeter insertion. **This is the safety net — guaranteed deliverable.**
- **Phase 2 (Option B, headline) — Vision-conditioned residual.** A wrist-mounted simulated RGB camera lets the policy *locate the target hole and sharpen the approach* before contact; the residual is retrained with the image as an input, on top of F/T + proprio. The same contact-alignment ability then carries the peg home.

If Phase 2 stalls, Phase 1 still produces a publishable project.

## Simulation environment

### Robot

- **Franka Emika Panda** manipulator (7-DoF), from MuJoCo Menagerie.
- Standard **Franka Hand** (2-finger parallel gripper, 1-DoF).
- **Pre-grasped peg**: each trial starts with the peg already firmly held in the gripper, at a slightly randomized pose relative to the end-effector. Grasping is *not* part of this project's scope.
- **Wrist-mounted RGB camera**: fixed offset from the gripper, configurable resolution and field of view.
- **Wrist-mounted light source**: co-located with the wrist camera so the field of view is always illuminated. Plus a soft ambient fill light in the scene.
- **Six-axis wrist force/torque sensor** (built into the Panda model in MuJoCo Menagerie).

### Workpiece

- **Vertical wall** in front of the robot, containing **multiple holes** at different positions on the surface.
- Only **one hole is the target** per trial; the other holes are visible distractors that train the vision policy to attend to the right one (this is *not* multi-task — the target is specified by the controller, not inferred by the robot).
- Holes are **chamfered** (rim bevel ~1–2 mm) so the peg can slide down the rim into the hole on contact. **The chamfer angle is the primary difficulty knob** — start generous, tighten as we tune.
- **Default geometry**: round, 10 mm hole / 8 mm peg (1 mm radial clearance).
- Each hole has a known nominal pose, given to the controller at trial start with a small random offset to represent prior uncertainty.

### Conventions & physics

- **World frame** at the robot base, z up. All poses (EE, peg, hole, target) reported in this frame. One convention, declared once, documented in the README.
- Default MuJoCo timestep (~2 ms / 500 Hz) for contact-rich stability.
- **F/T baseline calibration** at trial start: with no contact, the wrist F/T reading is dominated by the pre-grasped peg's weight (gravity bias). We record the no-contact reading and subtract it from the live signal, so the residual policy sees only contact-induced wrenches.

### Two render modes wired in from day 1

- **Interactive viewer** — for human-driven trials, demos, and visual debugging.
- **Headless offscreen rendering** — for training data collection and benchmark evaluation; only the wrist camera is rendered, no GUI window. Same simulation, different rendering surface — no code-path divergence between the two.

### Per-trial logging

Saved as one structured file per trial (Parquet or NPZ):

- Every sensor at every step: wrist F/T, joint state, EE pose.
- Decimated wrist-camera frames.
- Human command stream and residual-policy inputs/outputs.
- Ground-truth peg pose and target-hole pose (privileged — for offline analysis only, never fed to the deployed policy).

## Controller architecture

The control stack is layered, with the residual policy bolted on top of a fully-functional classical controller. The arm runs in either of two assistance modes (off / learned policy) by swapping just the topmost layer.

### Command pipeline

- **Operational-space differential control.** The active input strategy (human, scripted, etc.) commands an EE pose — position + orientation, plus an optional grip force. The controller computes a small Δpose from current to commanded each timestep and solves an incremental IK for joint motion.
- **Null-space resolution.** The Panda's redundant 7th DoF is resolved with a soft posture cost in the IK step — the elbow is biased toward a nominal configuration, keeping the arm out of singular postures for free.
- No global IK, no trajectory planning. Just frame-by-frame incremental control. Small steps + continuous configuration → no branch-jumping between IK solutions.

### Compliance / contact behavior

- **Impedance control with direction-dependent stiffness.** The arm behaves like a mass-spring-damper toward its commanded pose:
  - Along the insertion axis: **stiff** (we want to push in).
  - Lateral to insertion: **compliant** (the chamfered hole rim physically guides the peg in — passive alignment).
  - Off-axis rotation (pitch/roll): **compliant** (the peg can tilt to fit).
  - On-axis rotation (yaw): irrelevant for a round peg.
- Peak contact force is bounded by stiffness × max deflection. The arm is *physically incapable* of generating large forces — safety by design, not by software clamps alone.

### Runtime state — two modes only

The runtime controller is **mode-less in the autonomy sense.** It has no notion of task progress, success, or failure. It has two states:

- **Active** — input strategy in control, residual policy assisting, arm tracks the combined command.
- **Locked** — input ignored. Two variants:
  - *Hold lock* — arm freezes in place (safety trips, trial setup, manual pause).
  - *Park lock* — arm autonomously returns to a known safe pose (between trials, after the human disengages).

Trial-level concepts (start, end, success, failure) live in the **evaluation harness** — a passive observer that watches the runtime and computes KPIs offline. The controller has no dependency on the harness; they are developed and changed independently. This decoupling is a deliberate Dependency-Inversion choice and an architectural pillar of the project.

### Residual policy interface

The residual policy is structured as a **pose-delta + grip-force-delta** layer, sitting on top of the active input command:

- **Δposition**: 3D, clamped (e.g. ±2 cm/step).
- **Δorientation**: 3D axis-angle, clamped (e.g. ±10°/step).
- **Δgrip-force**: 1D, clamped (e.g. ±5 N/step).

These deltas are added to the input strategy's command before it reaches the impedance controller. Clamping is enforced *before* the controller sees the augmented command — the policy is safe-by-construction even if it outputs nonsense.

Setting all deltas to zero recovers "no-assist" mode for free. The same interface accepts deltas from any source: the analytical expert (deltas computed from privileged geometry, used only to generate training data) and the learned policy (deltas predicted by the residual network) share this exact output signature.

### Gripper behavior

- Default: closed at a baseline grip force, just enough to hold the pre-grasped peg.
- The residual policy can modulate grip force via Δgrip-force — including reducing it to allow the peg to slip into alignment when angled. This is one of the micro-adjustment behaviors we expect the policy to learn from the expert.
- Open trigger: detected by the input strategy (e.g., the human opens their hand in the webcam input). There is no internal task-completion state.

## What "AI" means in this project

Two ML components, in deliberately different roles:

- **MediaPipe Hands** — off-the-shelf, treated as a sensor library. Converts webcam frames into hand pose. Not a research contribution.
- **Vision-conditioned residual correction policy** — the project's headline ML contribution. A multi-stream network: GRUs over the command and force/torque histories, an MLP over proprioception, and a fine-tuned CNN (pretrained init) over the wrist image, fused by an MLP head; trained via behavioral cloning. Inputs: command history, force/torque history, proprioception, and (Phase 2) the wrist-camera image — four independent streams. Outputs: bounded pose and grip-force deltas. This is the component the KPI evaluation centers on. Full architecture and rationale: [`docs/design/policy-model.md`](docs/design/policy-model.md).

## Expert and data generation

The deployed residual policy is trained via behavioral cloning against a scripted **privileged-info expert** that runs in simulation. The expert is the source of all supervision for BC training.

### Expert algorithm

**Analytical / closed-form**, not RL-trained, not human-recorded. Given the privileged state (true peg pose, true target-hole pose, current arm state) plus the noisy-human's current command, the expert computes the optimal correction directly from geometry. Standard for peg-in-hole in the BC literature. The expert is a *tool*, not a research contribution — the research contribution is the deployed policy that has to do the same job *without* privileged info.

### Expert interface

- **Inputs**: privileged ground truth (true peg + hole poses), current sensor readings, current scripted-noisy-human command.
- **Outputs**: same signature as the residual policy — Δposition (3D), Δorientation (3D), Δgrip-force (1D), all clamped to the same bounds.

The symmetric output contract is what makes BC clean: the policy mimics the expert's output exactly.

### Data-generation rollout

One **episode** = one complete trial, from arm-at-base to terminal condition.

Initialization: scene parameters randomized (see *Coverage*), arm at base, peg pre-grasped, scripted noisy-human's intent set to "navigate to and insert into hole #k". Random seed fixed for the episode.

Per-step loop (every control timestep, ~100 Hz):
1. Read full sim state and sensor readings.
2. Scripted noisy-human computes a commanded pose for this timestep.
3. Expert sees (state + noisy command), computes Δ.
4. Final command (noisy + Δ) goes to the impedance controller.
5. Sim steps forward.
6. Log a row: (sensors, noisy command, expert Δ, ground-truth state for offline analysis).

Episode terminates on: peg inserted past threshold depth → success; force cap exceeded → safety abort; timeout → failure; noisy-human's "intent" completes (releases peg, withdraws) → end.

Episode length: 5–30 simulated seconds, 500–3000 logged rows per episode.

### Coverage / randomization

Base scope (randomized per episode):
- Target hole position (which hole + small offset within the noise envelope).
- Distractor hole positions on the wall.
- Initial peg pose in the gripper (small randomization).
- Scripted noisy-human's noise pattern (σ, drift, seed).

Stretch — environmental physics & visual randomization:
- **Underwater-style drag**: a velocity-dependent damping force on the peg/gripper (in MuJoCo, just a damping parameter). Trains the policy to handle different physical environments and interacts interestingly with the residual — policy can learn to modulate delta magnitudes by environment.
- **Lighting variation** (Phase 2 only).
- **Wall material / texture variation** (Phase 2 only).

### Data quality / filtering

**Keep all episodes**, log a `success` flag per episode and per step. The expert's action is always meaningful — even on a failing trajectory, the expert is still doing the right thing given the state. BC benefits from diverse state coverage. If failure trajectories later prove harmful, filtering them is a one-line change; the data is already on disk.

### Volume

Effectively unlimited free data — **the "noisy-human" used for data generation is a scripted programmatic actor, not a real human**. Generation runs at sim speed, unattended. Constraint is training compute, not human time or data availability.

Starting targets:
- Phase 1 (F/T-only residual): ~1,000 episodes ≈ 500K–1M frames; training ≈ a few GPU-hours.
- Phase 2 (vision-conditioned residual): ~5,000 episodes ≈ 2.5M–5M frames; training ≈ 10–20 GPU-hours.

Calibrate by validation curves. Sim throughput supports overnight regeneration if we change something significant.

## Sensing strategy

Layered by purpose, not pooled:

| Sensor | Used by | Purpose |
|---|---|---|
| Wrist-mounted RGB camera | Deployed policy (Phase 2) | Locates the hole during approach; resolves "where is the world" without external cameras |
| Wrist force/torque | Deployed policy (both phases) | Primary signal for inferring contact geometry — "where is the peg catching?" |
| Proprioception (joint angles/velocities, EE pose) | Deployed policy (both phases) | Where the arm currently is |
| Privileged true peg/hole pose | Expert (data-generation only) | Lets the scripted expert produce supervision; **never available to the deployed policy** |

Deliberately **not** used: external (top-down) scene camera streaming continuous hole pose. It would trivialize perception and ruin the project's framing.

## Tech stack

Python end-to-end, intentionally avoiding low-level layers:

- **MuJoCo** via official `mujoco` Python bindings (model loading, stepping, sensors, camera rendering).
- **MediaPipe Hands** via its Python package (webcam → 2D/3D hand keypoints).
- **PyTorch** for the residual policy (per-stream encoders + fusion MLP head; fine-tuned image CNN; behavioral cloning training loop).
- **OpenCV** for webcam I/O and image preprocessing.
- **NumPy / SciPy** for the backbone controller (IK refinement, force capping, filtering).
- **Hydra** or plain YAML for run/experiment configuration.
- **Pytest** for tests.
- **Matplotlib + pandas** for KPI plots and ablation tables.

No C/C++/Rust extensions; no ROS. All glue code is plain Python.

## In scope

- MuJoCo simulation as described in **Simulation environment** above (Franka Panda + parallel gripper + chamfered vertical-wall fixture with distractor holes + wrist camera + wrist light + pre-grasped peg, ~8 mm peg / 10 mm hole round geometry to start).
- Three swappable input strategies (vision, keyboard, scripted noisy human).
- Two swappable assistance modes (none, learned residual).
- Wrist-mounted simulated RGB camera, rendered by MuJoCo at training and eval time.
- Scripted privileged-info expert that produces training trajectories.
- Behavioral cloning training pipeline for the residual policy (image CNN fine-tuned end-to-end from a pretrained init; freeze held as fallback).
- Two trained policy variants for ablation: F/T-only (Phase 1) and vision+F/T (Phase 2).
- KPI logging and ablation runs comparing assist off vs on, plus the Phase-1/Phase-2 vision ablation, head-to-head.
- All booklet-required deliverables: design document with architecture diagram and sequence chart, design alternatives writeup, README, runnable code.
- Self-evaluation & reflective analysis writeup (5% bonus).
- **Public GitHub repository** with a polished project page (README + media) suitable for showcasing in a portfolio context.
- **1–2 minute demo video** showing webcam → robot, assistance toggling on/off, and a KPI summary montage. Embedded in the project page.

## Explicitly out of scope (anti-scope)

- Real hardware. Sim only.
- Vision models trained from scratch. MediaPipe Hands is used as-is; the policy's image CNN is **pretrained-initialized and fine-tuned end-to-end** as part of the BC policy (a deliberate decision — see [`docs/design/policy-model.md`](docs/design/policy-model.md) Decision B — superseding the earlier "pretrained backbone used as-is" plan). No bespoke vision model is trained from random init in isolation, and a frozen backbone remains the fallback.
- Reinforcement learning. Behavioral cloning only.
- Multi-step assembly. One peg, one hole.
- Multi-user studies. Human-in-the-loop runs are qualitative (a handful of demo recordings by the author); statistical comparisons use the scripted noisy-human under matched conditions.
- Task inference / multi-task generalization. The task is known to be "insertion".
- External scene camera, motion-capture suit, EEG, EMG, gloves, VR controllers, force-feedback haptics.
- Domain randomization beyond what's needed to make the BC policy not crash in eval (cosmetic robustness is a stretch goal).

## Failsafe and safety envelope

Safety is enforced in three layers, strongest first:

1. **Passive (impedance control).** Peak contact force is bounded by stiffness × maximum deflection. With the chosen stiffness profile, no command — including garbage from a misbehaving residual policy — can produce large forces. This is the strongest guarantee because it is mechanical, not algorithmic. See *Controller architecture → Compliance*.
2. **Active clamps on the residual.** Δposition ≤ 2 cm/step, Δorientation ≤ 10°/step, Δgrip-force ≤ 5 N/step. Enforced before the controller sees the augmented command. See *Controller architecture → Residual policy interface*.
3. **Trip-and-lock watchdog.** Monitors the runtime; if peak force exceeds a configurable threshold (e.g., 30 N), the trial timeout (e.g., 60 s) is hit, or the residual emits NaN / out-of-distribution values, the runtime enters **hold lock** and the trial is recorded as a failure.

Between trials, the runtime executes a **park lock** routine returning the arm to a base pose.

## Experimental design

Three headline configurations, all driven by the same scripted noisy-human under matched conditions (~100 trials each):

1. **Human-only (off)** — no assistance layer, Δ=0. Baseline.
2. **Human + residual policy (vision + F/T)** — policy on. Headline configuration.

Plus one **ablation**: the Phase 1 F/T-only residual, evaluated on the same trial set, to isolate the contribution of vision.

### Evaluation randomization

What changes per trial (to make 100 trials statistically meaningful):

- **Hole pose**: small random offset from the "known" position given to the controller — represents prior uncertainty.
- **Initial peg pose**: randomized starting position for the arm.
- **Noisy-human command stream**: a biased coarse trajectory toward the target with **structured low-frequency noise** (per-episode constant bias + correlated drift on position and orientation), at a realistic update rate (~5–10 Hz) — *not* per-step i.i.d. Gaussian noise (which would make the expert a trivial noise-negator). Same noise seeds across configurations for paired comparisons. See [`docs/design/human-generation.md`](docs/design/human-generation.md).
- **Fixed master seed list** so results are reproducible.

> Specific magnitudes (hole-noise σ, human-noise σ, force cap, trial timeout) are deliberately left as placeholders. They will be **calibrated after the human-only baseline runs**, so they describe a problem that is genuinely hard for the unassisted human-only baseline — not trivially easy and not trivially impossible.

### Per-trial KPIs

- Insertion success (bool)
- Time-to-insert (seconds)
- Peak contact force (N) — proxy for safety
- Number of contact events before success
- Trajectory smoothness (integrated jerk on end-effector)

Plus a small qualitative section: 3–5 recorded vision-input demos showing the system working end-to-end with a real webcam.

## Success criteria

- Working integrated demo: webcam-driven teleop produces visible insertion attempts in MuJoCo, with assistance mode toggleable at runtime.
- Phase 1 (F/T-only residual) outperforms human-only on success rate; peak force bounded by construction.
- Phase 2 (vision-conditioned residual) outperforms human-only on success rate **and** peak force, statistically meaningful; and beats Phase 1 (the vision ablation).
- Architecture cleanly separates input layer / backbone controller / assistance layer; Strategy pattern at each seam; SOLID compliance defensible during the design review.
- All booklet-required deliverables submitted on time and to professional quality, including self-evaluation writeup.

## Constraints

- Solo project. Scope is deliberately right-sized for one person at 10–15 hrs/week.
- Total runway: 2026-05-18 → 2026-08-31. ~15 weeks, ~150–225 hours total, ~100–150 hours pure implementation after onboarding/prep/eval overhead.
- Adding vision (Phase 2) costs ~30–40 hours over Phase 1; still inside the time envelope, but tighter.
- Topic approval ~2026-06-08; mid-semester design-review presentation mid-July; final submission 2026-08-31.
- Grading rubric weights architecture, SOLID, and code quality heavily (~55% of implementation grade). Scope is sized to leave time for clean engineering, not just a working demo.

## Deferred design — components 4–7

Components 4 and 5 have since been **designed in detail** — see the design docs in [`docs/design/`](docs/design/) ([problem structure](docs/design/problem-structure.md), [human generation](docs/design/human-generation.md), [expert corrections](docs/design/expert-corrections.md), [policy model](docs/design/policy-model.md)). Components 6 and 7 still have open decisions. The interface contracts established by components 1–3 (sim environment, controller, expert/data generation) are sufficient to begin implementation regardless.

### Component 4 — Residual policy  *(designed — see [`docs/design/policy-model.md`](docs/design/policy-model.md))*

- **Settled**: input signature (four streams — command history, F/T history, proprioception, wrist image), output signature (clamped Δpose + Δgrip-force), training method (BC from the expert); multi-stream-encoder + fusion-head architecture; **GRU** temporal encoders (1D-CNN fallback); **fine-tuned image CNN** (pretrained init, freeze fallback) with an optional all-holes auxiliary loss; **implicit** goal (no explicit goal input); per-channel rotation-aware Huber loss with episode-level train/val split.
- **Still open** (tuning, not architecture): history lengths, image resolution, loss weights, aux-loss `λ` schedule, GRU sizing, measured inference latency.

### Component 5 — Input strategies

- **Settled**: three swappable strategies (vision, keyboard, scripted noisy-human) behind a common interface; vision uses pretrained MediaPipe Hands; scripted noisy-human noise model is **structured low-frequency biased noise** (per-episode bias + correlated drift), per [`docs/design/human-generation.md`](docs/design/human-generation.md).
- **Still open**: live stereo workspace-calibration tuning (metric scale, axis signs, gain — rig-dependent knobs in `WorkspaceCalibration`), noise magnitudes (deferred to post-baseline calibration), keyboard control bindings. *(The mapping/clutch/jitter-filter design itself is settled and metric.)*

### Component 6 — Evaluation harness

- **Pending**: trial-start detection logic, success/failure classification rules, KPI computation specifics, ablation orchestration, statistical-analysis approach.
- **Already settled**: passive-observer architecture (no controller→harness dependency); three configurations evaluated; per-trial KPIs defined.

### Component 7 — Overall architecture / module boundaries

- **Pending**: final repo layout, dependency-direction enforcement strategy, configuration system (Hydra vs plain YAML), CLI/entry-point design, test strategy.
- **Already settled**: Strategy pattern at three seams (input / assistance / lock); Dependency Inversion between controller and eval harness; Python end-to-end.

## Implementation roadmap

The scope-level design for components 1–3 is locked. Implementation can proceed in parallel with continued scoping of 4–7.

**Milestone 1 — Sim environment online.** MuJoCo scene with Franka Panda + chamfered wall + holes loads. Viewer runs; headless renderer runs. Peg-pre-grasp logic working. F/T sensor + wrist camera return values. *(Component 1.)*

**Milestone 2 — Controller online.** Operational-space differential IK + direction-dependent impedance controller working end-to-end. Manual pose commands move the arm. Force-cap watchdog working. Park-lock / hold-lock implemented. *(Component 2 — backbone controller.)*

**Milestone 3 — Assistance seam + scripted input online.** The assistance Strategy seam is wired so deltas can come from any source (none / expert / policy), and a stub scripted noisy-human drives the system. The no-assist (zero-Δ) mode is runnable for contrast. *(Assistance seam + a stub scripted noisy-human.)*

**Milestone 4 — Expert and data generation online.** Analytical expert implemented. Full data-generation pipeline runs unattended: hundreds of episodes → structured log files on disk. Spot-check: a few example episodes are visually sane. *(Component 3.)*

By Milestone 4 we have: the human-only baseline mode (zero correction), the assistance seam wired with the analytical expert, and a data-generation pipeline ready for when the residual policy lands. That's already a working system — Phase 1 of the project is feasible from this state without any of the deferred components.

## Stretch goals (prioritized, only if time remains)

1. **Tighter tolerance** (9 mm peg in 10 mm hole; eventually 9.5/10) — most impressive demo polish for least extra code.
2. **Additional peg/hole geometries**:
   - Square peg / square hole (orientation along the insertion axis now matters — harder).
   - Multiple peg sizes (e.g., 6 mm, 8 mm) for a tolerance-gradient demo in the final video.
   - Hex or keyed geometries as further exotic options.
3. **Mild domain randomization at training time** — randomize texture, hole color, lighting intensity, and wall material across training episodes. Trains the vision policy to ignore cosmetic variation; supports a "real-world transferable" framing.
4. **"Themed" demo environments** — exaggerated lighting scenarios layered on top of the base scene for the final video:
   - **Deep-underwater**: bluish dim ambient, strong wrist-light cone, particulate fog.
   - **Storm**: dark ambient with occasional bright lightning flashes (strobe-like global light).
   - **Industrial low-light** / dusty warehouse.
   These stress the vision policy and produce strong demo-reel material. Built on top of the same MuJoCo scene — just shaders + lighting parameters.
5. **Brief RL fine-tuning comparison** of the residual policy on top of the BC-trained weights — only as a final tail experiment, never the centerpiece.
6. **MediaPipe Holistic** in place of MediaPipe Hands — adds full-arm pose tracking for richer teleop input.

*(Formerly listed here: stereo hand tracking — now **shipped** as the project's sole vision input, replacing the monocular baseline. See [`docs/design/teleop-input.md`](docs/design/teleop-input.md).)*

## Key risks (named, mitigation deferred to implementation plan)

- **Contact dynamics tuning.** MuJoCo's tight-tolerance contact is fiddly.
- **Workspace calibration for vision input.** Mapping camera-space hand motion to robot workspace needs a clutch + scaling system.
- **Vision-conditioned BC data hunger.** Likely needs more demonstrations than F/T-only — risk of underfitting.
- **MuJoCo camera rendering throughput.** Rendering at every step slows training; may need to batch or downsample.
- **Residual policy fails to outperform human-only.** Mitigation: the paired-seed design has high power to detect even small gains, and difficulty is calibrated to leave headroom above the human-only baseline.
- **Scope creep toward RL.** Tempting but time-fatal; held as explicit anti-scope.

---

*Next document: implementation plan — how each of the components above is actually built, at the level of algorithms, data flow, and modules (but not yet specific libraries / hyperparameters).*
