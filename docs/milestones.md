# Project Milestones — Roadmap

The complete milestone roadmap for the AI-assisted teleoperation project. This is the
**parent document**: each milestone here is defined at the scope level (goal, what's in,
what's deferred, acceptance, dependencies, rough effort). Detailed per-milestone specs
(`milestone-N-spec.md`) are created from these definitions one at a time, as each
milestone is started.

- Authoritative design decisions live in [`../project-scope.md`](../project-scope.md).
- This file is the *plan*; the scope doc is the *definition*. If they conflict, the scope
  doc wins and this file should be reconciled.
- Per the root `CLAUDE.md` convention: at the end of each milestone, record durable,
  transferable findings (tool quirks, techniques) in `project-wiki/` — but milestone
  *status* itself stays here and in git history, not in the wiki.

## Milestone map

| # | Milestone | Phase | Component(s) | Status |
|---|-----------|-------|--------------|--------|
| M1 | Sim environment online | Foundation | 1 | ✅ done |
| M2 | Backbone controller online | Foundation | 2 (partial) | ⬜ next |
| M3 | Assistance seam + scripted input online | Foundation | seam, 5 (stub) | ⬜ |
| M4 | Expert + data generation online | Foundation | 3 | ⬜ |
| M5 | Residual policy — Phase 1 (F/T-only) | Phase 1 | 4 | ⬜ |
| M6 | Evaluation harness + Phase 1 results | Phase 1 | 6 | ⬜ |
| M7 | Vision-conditioned residual — Phase 2 | Phase 2 | 4 (vision) | ⬜ |
| M8 | Human teleop input (MediaPipe + keyboard) | Phase 2 | 5 | ⬜ |
| M9 | Final evaluation + polish | Delivery | 6, 7 | ⬜ |

Two **graded course checkpoints** (D1, D2) draw on these — see *Course checkpoints* below.

---

## Foundation phase

### M1 — Sim environment online ✅

**Goal**: a working MuJoCo scene (Panda + chamfered wall + holes + pre-grasped peg +
wrist camera + wrist light + F/T sensor) behind a clean `SimEnv` API, with viewer and
headless render paths. No control logic.

**Status**: done in a prior session. Detailed spec: [`milestone-1-spec.md`](milestone-1-spec.md).
Durable findings recorded in `project-wiki/entities/{mujoco,franka-panda}.md`.

**Feeds**: every later milestone (the `SimEnv` API contract).

---

### M2 — Backbone controller online

**Goal**: turn an EE-pose command into arm motion, compliantly. The arm tracks manual
pose targets and behaves safely on contact.

**In scope**:
- Operational-space differential IK (small Δpose per step) with null-space posture cost.
- Direction-dependent impedance control (stiff along insertion axis, compliant laterally
  and off-axis).
- Force-cap watchdog; `hold-lock` / `park-lock` runtime states.
- Manual pose commands (a dev harness) to drive the arm for visual confirmation.

**Deferred (anti-scope)**: any input strategy (M3), the assistance seam (M3), expert/policy.

**Acceptance**: commanded EE pose is tracked smoothly; pushing the peg into the wall
produces bounded forces (impedance gives, doesn't crush); force-cap trip → hold-lock;
park-lock returns to base pose.

**Depends on**: M1. **Component**: 2 (backbone controller). **Rough effort**: 12–16 h.

---

### M3 — Assistance seam + scripted input online

**Goal**: establish the assistance seam — one interface through which *any* Δ source
(none, expert, or learned policy) plugs in — and a stub automated command stream to
drive the system end-to-end without a real human.

**In scope**:
- Wire the assistance seam so deltas can come from any source (no-assist now; expert and
  policy later) behind one interface — dependency inversion, so swapping the Δ source
  touches nothing upstream or downstream.
- A stub `ScriptedNoisyHuman` input strategy (target pose + simple noise) — just enough
  to drive the system without a real human.
- The no-assist (zero Δ) mode runnable end-to-end through the seam.

**Deferred**: the full noisy-human noise model (refined in M4), vision/keyboard input,
the expert and the learned policy.

**Acceptance**: with the stub noisy-human commanding, the system runs end-to-end in
no-assist mode through the seam; the seam accepts an injected Δ from a dummy source
(proving the interface) without upstream/downstream changes.

**Depends on**: M2. **Components**: assistance seam, 5 (stub). **Rough effort**: 6–10 h.

---

### M4 — Expert + data generation online

**Goal**: an unattended pipeline that produces behavioral-cloning training data.

**In scope**:
- Analytical privileged-info expert (closed-form correction from true peg/hole geometry +
  current noisy-human command). Output signature = the residual interface (Δpose + Δgrip).
- Data-generation rollout loop (noisy-human → expert → controller → sim; log structured
  rows per step).
- Coverage randomization: target/distractor hole positions, initial peg offset,
  noisy-human noise pattern. Per-episode + per-step `success` flags.
- Refine the `ScriptedNoisyHuman` to a realistic noise model.

**Deferred**: vision randomization (M7), drag/themed environments (stretch).

**Acceptance**: running the pipeline produces hundreds of episodes as on-disk trajectory
files; a spot-check of a few episodes is visually sane; data schema is stable enough to
train against.

**Depends on**: M3. **Component**: 3. **Rough effort**: 10–14 h.

> **End of Foundation**: at this point we have a human-only baseline mode, the analytical
> expert wired into the assistance seam, and a data-generation pipeline. This is already
> enough to drive the **Design Review (D1)** with a real preliminary prototype.

---

## Phase 1 — learned assist (F/T only)

### M5 — Residual policy, Phase 1 (F/T-only)

**Goal**: the headline ML component, first version — a BC-trained residual that corrects
the noisy-human command using force/torque + proprioception (no vision yet).

**In scope**:
- BC training pipeline (dataset loader, model, train/val loop, checkpointing).
- Policy (Phase-1, no image): GRU encoders over command history + F/T history, MLP over
  proprioception, fused → clamped Δpose + Δgrip. See [`design/policy-model.md`](design/policy-model.md).
- Integrate the trained policy as the "learned assist" mode behind the assistance seam.

**Deferred**: vision conditioning (M7), RL (anti-scope).

**Acceptance**: the trained policy runs in the loop in real time; on held-out episodes it
qualitatively improves insertion over human-only; training/validation curves are sane.

**Depends on**: M4. **Component**: 4 (Phase 1). **Rough effort**: 12–18 h.

---

### M6 — Evaluation harness + Phase 1 results

**Goal**: turn "it seems to work" into measured, defensible numbers.

**In scope**:
- Passive-observer eval harness (trial start/end detection, success/failure
  classification, KPI computation) — no controller→harness dependency.
- Ablation orchestration: run human-only vs F/T-residual under matched
  noisy-human conditions (~100 trials each, paired seeds).
- **Calibrate** the deferred eval-randomization magnitudes (hole-noise σ, human-noise σ,
  force cap, timeout) so the task is genuinely hard for human-only.
- KPI tables + plots (success rate, time-to-insert, peak force, contacts, smoothness).

**Acceptance**: a reproducible two-way KPI comparison with the F/T residual measurably
beating human-only. **This is the first publishable result — Phase 1 complete.**

**Depends on**: M5. **Component**: 6. **Rough effort**: 10–14 h.

---

## Phase 2 — vision

### M7 — Vision-conditioned residual

**Goal**: upgrade the residual to the headline vision-conditioned policy — the project's
main ML contribution.

**In scope**:
- Wrist-camera frames rendered into the data pipeline + fed to the policy.
- Vision-conditioned policy: add a **fine-tuned image CNN** (pretrained init, trained
  end-to-end with the rest; freeze fallback) over the existing GRU/MLP streams
  (image + F/T + proprioception + command). Optional all-holes auxiliary loss as a
  training stabilizer. See [`design/policy-model.md`](design/policy-model.md) Decision B.
- Regenerate data with images; retrain; ablate vision+F/T vs F/T-only on the same trials.
- Optional: training-time domain randomization (lighting/texture) for robustness.

**Deferred**: themed demo environments (stretch), drag physics (stretch).

**Acceptance**: vision-conditioned policy trained and evaluated; ablation shows the
contribution of vision; runs in real time within the control budget.

**Depends on**: M6 (eval harness reused). **Component**: 4 (Phase 2). **Rough effort**: 14–20 h.

---

### M8 — Human teleop input (MediaPipe + keyboard)

**Goal**: let a real human drive the arm — for the demo and qualitative validation.

**In scope**:
- `VisionInput` strategy via MediaPipe Hands (webcam → hand pose → EE command).
- Workspace calibration (camera-space → robot-space), clutching, gain, jitter filter
  (one-euro), drop-out handling.
- `KeyboardInput` fallback strategy.
- All three input strategies interchangeable behind the common interface.

**Deferred**: MediaPipe Holistic / full-arm tracking (stretch).

**Acceptance**: a person can drive the arm via webcam and complete assisted insertions;
keyboard fallback works; a handful of qualitative runs are recordable.

**Depends on**: M3 (input interface), works with any assist mode. **Component**: 5.
**Rough effort**: 12–16 h. *(Lower priority than M5–M7 for the core results; needed for
the final demo video.)*

---

## Delivery

### M9 — Final evaluation + polish

**Goal**: produce the final, submission-grade results and artifacts.

**In scope**:
- Full final KPI runs across all configurations; statistical analysis.
- 3–5 recorded webcam-driven qualitative demo runs.
- Demo video (webcam → robot, assistance toggled on/off, KPI montage).
- README polish; public repo finalization; reproducibility check (clean clone → run).
- Self-evaluation & reflective analysis writeup (5% bonus).

**Acceptance**: all final-submission artifacts complete and reproducible (see D2).

**Depends on**: M6 (+ M7, M8 for the full story). **Components**: 6, 7. **Rough effort**: 12–16 h.

---

## Course checkpoints (graded)

These are **deliverable** milestones, not code milestones — they require dedicated
writing/diagram/slide work that draws on the implementation milestones.

### D1 — Design Review Package (~mid-July, ~35% of grade)

A professional design-review document + ~slides. Required content (from the booklet):
project requirements, system architecture with **architecture diagram + sequence chart**,
**≥2 design alternatives** with trade-offs and rationale, simulation scenarios, KPI
definitions, challenges/risks, a preliminary prototype/demo, evaluation criteria, timeline.

**Source material**: most of this already exists in [`../project-scope.md`](../project-scope.md);
D1 is largely repackaging it into review form + adding diagrams + a live demo.
**Milestone readiness target**: M1–M4 done (working expert-driven prototype to demo); M5–M6
in progress is a bonus. **Rough effort**: 8–12 h (mostly writing + diagrams).

### D2 — Final Submission (2026-08-31, ~65% of grade)

Required: a design document (architecture, components, scenarios, metrics, etc.), a README
(install + run), and the code itself. We additionally deliver the demo video and self-eval.

**Milestone readiness target**: M1–M9 (Phase 2 ideally; Phase 1 is the floor — a complete,
well-engineered Phase-1 project is a strong submission on its own). **Rough effort**:
6–10 h packaging on top of M9.

---

## Indicative timeline

Solo, ~10–15 h/week, today ≈ 2026-05-21 → deadline 2026-08-31 (~15 weeks). M1 already done.

| Window | Target |
|---|---|
| late May | M2 |
| early June | M3 |
| mid June | M4 (Foundation complete) |
| ~June 8 | *Topic approval* — covered by `project-scope.md` |
| late June | M5 |
| early July | M6 (**Phase 1 complete — publishable**) |
| ~mid July | **D1 Design Review** (demo the expert-driven prototype + early policy) |
| mid–late July | M7 (vision) |
| early August | M8 (human teleop) |
| mid–late August | M9 (final eval + polish) |
| **2026-08-31** | **D2 Final Submission** |

## Sequencing & risk notes

- **Critical path to a passing project**: M1→M6 (Phase 1). If time gets tight, a polished
  Phase-1 project (F/T residual + clean off-vs-on KPI comparison + good
  engineering) is a complete, defensible submission. M7–M8 are the upside.
- **Biggest risk milestone**: M7 (vision-conditioned BC is data-hungry and the place solo
  projects most often miss KPI targets). Mitigation: M5/M6 give a working F/T baseline and
  calibrated targets before vision is attempted.
- **M8 is demo-enablement**, not core-results — the statistical KPIs come from the scripted
  noisy-human. M8 can slip latest without endangering the core contribution.
- **Parallelizable**: M8 (human input) is independent of M5–M7 and could be built earlier
  if a live demo is wanted for D1.
