# Milestone 5 — Residual Policy, Phase 1 (F/T-only)

**Goal**: the project's headline ML component, first version. Train a residual
correction network `π_θ` by **behavioral cloning** to reproduce M4's analytical
expert's `Δ` from **non-privileged** observation — command history, force/torque
history, proprioception, **no vision** — and run it in the loop behind the M3
assistance seam, where it qualitatively beats human-only on held-out episodes.

M4 produced the training corpus (the trajectory schema) and proved an *expert*
that cheats (reads true poses) can seat the peg. M5 is where a network learns to
do the same from what a real robot actually senses. This is the
privileged-to-non-privileged transfer that *is* the project (see
`docs/design/problem-structure.md`). Phase 1 deliberately omits the camera: the
operator's coarse command brings the peg to the hole vicinity and the policy
supplies contact-reactive alignment from F/T — it isolates the contact-reasoning
contribution and is the **guaranteed deliverable**. Vision (Phase 2) is M7.

The architecture is **locked** in `docs/design/policy-model.md` (multi-stream
encoder + fusion head; GRU temporal encoders with a 1D-CNN fallback). M5 builds
exactly that skeleton minus the image branch, so Phase 2 only *widens* the input
— keeping the Phase-1-vs-Phase-2 ablation clean.

## Definition of done

By the end of M5 we can:

- **Load** the M4 NPZ corpus into batched, windowed training samples — command
  history, F/T history, proprioception in; the expert's `Δ` as the target — with
  an **episode-level** train/val split.
- **Train** the Phase-1 residual via BC (per-channel, rotation-aware loss) with
  sane train/val curves, checkpointing, and early stopping.
- **Run the trained policy in the loop in real time** as an `AssistProvider`
  behind the M3 seam — swapped in for `NoAssist`/`Expert` with no runner, input,
  or controller change — within the control-step latency budget.
- Show the learned residual **qualitatively beats human-only** (`NoAssist`) on
  held-out episodes (deeper seating / lower lateral error), the same headless
  spot-check shape M4 used for the expert. (Rigorous KPI numbers are **M6**.)

## What's in M5

- **BC dataset loader + windowing** (LAB-32, `data/`) — turns flat per-step NPZ
  rows into windowed `(streams, Δ*)` tensors; episode-level split; normalization.
- **Phase-1 residual model** (LAB-33, `policy/`) — GRU(command) + GRU(F/T) +
  MLP(proprio) → concat → MLP fusion head → 7-vector `Δ_raw`. No image branch.
- **BC train/val loop + checkpointing + seam integration** (LAB-34,
  `policy/` + `scripts/`) — the training pipeline, plus a `ResidualPolicy`
  `AssistProvider` that wraps a checkpoint for stateful real-time inference and
  slots into the seam.

These three implementation issues **already exist** in the M5 milestone; this
spec is their detailed expansion, one build step each.

## What's not in M5 — explicit anti-scope

- **Vision conditioning.** No wrist-camera stream, no image CNN branch, no aux
  heatmap head — that is **M7** (Phase 2). The schema reserves the column; M5
  trains on F/T + proprioception + command only.
- **The evaluation harness + KPI numbers.** Success-rate/force/time tables and
  the paired-counterbalanced ablation are **M6** (LAB-36/37/38). M5's "beats
  human-only" is a qualitative spot-check, not the measured result.
- **DAgger / expert-action-noise recovery.** Held in reserve
  (`problem-structure.md`); only escalated if open-loop BC rollouts drift. M5
  ships open-loop BC.
- **RL.** Out entirely.
- **Re-generating the corpus.** M5 trains on whatever M4 wrote; tuning data
  volume/coverage is a calibration knob, not new M5 scope.

## Design — the four pieces

### Inputs — the Phase-1 streams (recap from `policy-model.md`)

Three independent streams, each with its own encoder. Histories are
**zero-padded at episode start** (the model must tolerate a partially-filled
buffer — itself realistic).

| Stream | Shape (tunable) | Source columns | Encoder |
|---|---|---|---|
| Command history | `H_c×7` (~50 steps) | `cmd_position` (3) + `cmd_quaternion` (4) | GRU → `e_cmd` |
| F/T history | `H_f×6` (~20 steps) | `wrist_ft` (6, already bias-subtracted) | GRU → `e_ft` |
| Proprioception | `~24` | `ee_pose`→3+6D, `joint_positions` (7), `joint_velocities` (7), `gripper_width` (1) | MLP → `e_pro` |

The command stream carries the **base** operator command `c_t` (pre-Δ), exactly
what the schema's `cmd_*` columns log and what the seam hands `get_delta` — never
the policy's own Δ. Quaternion inputs are converted to a continuous **6D**
representation in the loader (raw quaternions are admissible as *inputs* but 6D
avoids the sign ambiguity and matches the proprio EE-rotation encoding).

### Output and the BC target

`Δ_raw = π_θ(o_t)` is **7 numbers**: `(Δposition ∈ ℝ³, Δorientation ∈ ℝ³
axis-angle, Δgrip ∈ ℝ¹)` — identical signature to the expert. The training
target is the schema's `delta_position` / `delta_orientation` / `delta_grip`. The
hard safety clamp (`±2 cm / ±10° / ±5 N`) is applied **outside** the network via
`domain.clamp_delta`, so the policy is safe-by-construction even if it emits
garbage; an optional `tanh`-scaled head keeps raw outputs near range to ease
training but is *not* the safety bound.

### Model — multi-stream encoder + fusion head (LAB-33)

```
command (H_c×7) ─► GRU ─► e_cmd ┐
F/T     (H_f×6) ─► GRU ─► e_ft  ├─► concat ─► MLP fusion head ─► Δ_raw (7)
proprio (~24)   ─► MLP ─► e_pro ┘
```

Late fusion of separately-encoded modalities. The fusion head is where
cross-modal reasoning happens ("F/T says catching on the +x rim, command says
still pushing +x → correct toward −x"). **GRU** is the locked temporal encoder
(Decision A); the **1D-CNN over a fixed zero-padded window** is the documented
fallback sharing the same window-in/embedding-out contract, so swapping is a
localized encoder change. The skeleton is deliberately the Phase-2 skeleton minus
the `e_img` branch.

### Training — behavioral cloning (LAB-34)

- **Loss**: per-channel weighted, rotation-aware:
  `L = w_pos·Huber(Δ̂.pos, Δ*.pos) + w_ori·rot_loss(Δ̂.ori, Δ*.ori) + w_grip·Huber(Δ̂.grip, Δ*.grip)`.
  Channels differ in units/scale (cm vs rad vs N) and importance → separate
  weights. Huber (smooth-L1) for robustness to the occasional large expert Δ
  (MSE is the simpler fallback). Orientation loss is a proper rotation difference
  (geodesic / `log(R̂·R*ᵀ)`), **never** naive component subtraction or raw
  quaternion regression.
- **Episode-level split.** Train/val split at the **episode** level, never the
  step level — steps within an episode are highly correlated and a step-level
  split leaks and inflates validation scores.
- **Volume / schedule** (scope target): ~1,000 episodes, a few CPU/GPU-hours;
  calibrate by validation curves. M5 trains on the existing M4 corpus; regenerate
  more episodes (M4 driver) if the val curve is data-starved.
- **Checkpointing + early stopping** on the val curve.

### Real-time inference + seam integration (LAB-34)

A `ResidualPolicy` in `policy/` wraps a trained checkpoint as an
`AssistProvider`:

- Maintains rolling **history buffers** (command, F/T) and the **GRU hidden
  state**, advancing them each `get_delta` call; returns `clamp_delta(Δ_raw)`.
- **Per-episode reset** of buffers + hidden state — see the known-unknown on the
  reset hook below.
- Forward pass must fit inside one control tick (the sim runs at 500 Hz ⇒ ~2 ms;
  the design's nominal budget is ~10 ms). A small GRU+MLP on CPU is well under
  this; measure once encoder sizes are fixed.
- Slots into `run_episode` in place of `NoAssist`/`Expert` with **no** runner,
  input, or controller change — the dependency-inversion property M3 established.

### Dependency note — the `ml` extra

Phase-1 training/inference needs **PyTorch**. `torch` is already pinned in the
`ml` optional-dependency group (`pyproject.toml`); M5's first task is to make it
available to the test/CI environment (extend the CI install and the `dev`/test
extras so the loader + model + a tiny train-step test run in CI). Keep the heavy
import inside `policy/`/`data/` modules so the rest of the package still imports
without torch.

## Build order (estimated effort in parentheses)

Each step is its own branch → PR → CI → merge, in dependency order.

### Step 1 — BC dataset loader + windowing · LAB-32 (~3–4 h)

Files: `src/ai_teleop/data/dataset.py` (+ `data/__init__` re-export); tests in
`tests/test_dataset_loader.py`. Add `torch` to the test/CI environment.

- Read the M4 NPZ episodes via `ai_teleop.data.load_episode`; assemble per-step
  windows (`H_c`, `H_f` zero-padded at episode start), proprio vector (quat→6D),
  and the `Δ*` target. Expose as a `torch.utils.data.Dataset` + `DataLoader`.
- **Episode-level** train/val split; normalization stats computed on train only.
- **Per-step acceptance**: loads a real M4 run; window/target shapes are correct;
  zero-padding at episode start verified; no episode appears in both splits; a
  fixed seed reproduces batches; runs without a GPU.

### Step 2 — Phase-1 residual model · LAB-33 (~3–4 h)

Files: `src/ai_teleop/policy/model.py` (+ re-export); tests in
`tests/test_policy_model.py`.

- GRU(command), GRU(F/T), MLP(proprio) → concat → MLP fusion head → 7 outputs;
  optional `tanh`-scaled head. Hidden sizes/layers are hyperparameters. Encoder
  modules share a window-in/embedding-out contract so the 1D-CNN fallback drops
  in without head changes.
- **Per-step acceptance**: forward pass on a batch yields `(B, 7)`; handles a
  zero-padded (episode-start) window; stateful GRU path exercised; parameter
  count sane; CPU forward is fast; `isinstance` plays nice with the seam (the
  wrapper, not the raw `nn.Module`, is the `AssistProvider`).

### Step 3 — BC train/val loop + checkpointing + seam integration · LAB-34 (~5–7 h)

Files: `scripts/train_policy.py`, `src/ai_teleop/policy/residual_policy.py`
(the `AssistProvider` wrapper) (+ re-export); tests in `tests/test_residual_policy.py`.

- **Training**: per-channel rotation-aware Huber loss, Adam + schedule,
  train/val curves, early stopping, checkpoint (weights + normalization stats +
  hyperparameters + schema version). Calibrate size/epochs by the val curve.
- **Inference wrapper**: `ResidualPolicy(checkpoint)` — stateful history buffers
  + GRU hidden state, per-episode reset, `clamp_delta` on output.
- **Per-step acceptance**:
  - A tiny train run on a small corpus drives train + val loss **down** and
    checkpoints; resuming a checkpoint reproduces outputs.
  - `isinstance(ResidualPolicy(ckpt), AssistProvider)`; it runs in `run_episode`
    in place of `NoAssist` with no other change, within the latency budget.
  - **Headless spot-check**: paired vs `NoAssist` under a biased operator, the
    learned residual improves seating (deeper penetration / lower lateral error).

## Acceptance criteria

- `uv run poe check` green, including the new loader/model/policy tests, with
  `torch` available in CI.
- The loader produces correctly-shaped, episode-split, zero-padded windowed
  batches from a real M4 run, reproducibly.
- A BC training run on the M4 corpus drives train **and** validation loss down
  (sane curves), checkpoints, and early-stops.
- `ResidualPolicy` loads a checkpoint, satisfies `AssistProvider`, and runs in
  `run_episode` **in real time** in place of `NoAssist`/`Expert` with no
  runner/input/controller edit.
- Headless spot-check: the trained policy **qualitatively beats human-only** on
  held-out episodes.
- The M4 data-gen pipeline, M3 runner, M2 harness, and M1 smoke test all still
  pass — M5 adds `policy/` + `data/` layers and changes no existing contract.

## Total estimated effort

**12–18 hours**, 3–5 sessions, across three PRs. The long pole is LAB-34
(training loop + the stateful real-time wrapper + the qualitative win); the
loader and model are mechanical given the locked architecture. The genuine risk
is BC **covariate shift** (open-loop drift) — mitigations (keep-failures already
done in M4; expert-action noise; DAgger) are held in reserve and only escalated
if the spot-check shows drift.

## Files this milestone touches

```
src/ai_teleop/data/
├── __init__.py        (re-export the loader)                         LAB-32
└── dataset.py         (new — windowing Dataset/DataLoader, split)    LAB-32

src/ai_teleop/policy/
├── __init__.py        (populate — re-export model + ResidualPolicy)  LAB-33/34
├── model.py           (new — multi-stream encoder + fusion head)     LAB-33
└── residual_policy.py (new — AssistProvider inference wrapper)       LAB-34

scripts/
└── train_policy.py    (new — BC train/val loop + checkpointing)      LAB-34

tests/
├── test_dataset_loader.py   (new)                                   LAB-32
├── test_policy_model.py     (new)                                   LAB-33
└── test_residual_policy.py  (new — conformance + seam + spot-check) LAB-34

pyproject.toml / CI         (make torch available to tests)           LAB-32
```

`src/ai_teleop/{control,sim,domain,expert,input}/` are **not** modified — M5
consumes the M4 schema and the M3 seam through their existing contracts. The one
possible exception is an optional `reset()` hook on the seam for stateful
providers (see below).

## Known unknowns / things to figure out during M5

- **Stateful-policy episode reset.** The GRU hidden state + history buffers must
  reset per episode, but `AssistProvider.get_delta` has no reset signal. Options:
  (a) the policy sniffs `observation.sim_time` resetting toward 0; (b) add an
  optional `reset()` to the provider that `run_episode` calls at episode start.
  Prefer (b) — explicit, and a no-op for stateless providers (`NoAssist`,
  `Expert`). Decide in LAB-34; it's the only candidate change to a shared
  contract.
- **History lengths** `H_c`, `H_f` — calibrate against validation curves.
- **Loss specifics** — Huber vs MSE; per-channel weights `w_pos/w_ori/w_grip`;
  exact rotation loss (geodesic vs 6D-MSE).
- **GRU sizing** — hidden size, layers, shared vs separate GRU for command/F-T.
- **`tanh`-scaled output head** — whether it helps training.
- **Inference latency** — measure once encoder sizes are fixed; fall back to the
  1D-CNN encoder if the stateful GRU path is fiddly or too slow.
- **Command-history rotation rep** — quaternion-as-input vs 6D; default 6D for
  consistency with proprio.

## Handoff to Milestone 6 and Milestone 7

- **M6** (eval harness + Phase-1 results) consumes the M5 deliverable: it runs
  the trained `ResidualPolicy` as the "learned assist" mode against the
  human-only (`NoAssist`) baseline under the paired-counterbalanced protocol, and
  produces the measured KPI tables. M5 gives M6 a real-time learned assist behind
  the seam; M6 owns the trial concepts and the numbers.
- **M7** (Phase 2, vision) widens this same architecture with the image-CNN
  branch (+ optional aux head). Because M5 built the locked skeleton minus the
  `e_img` branch, Phase 2 is an additive change and the `Phase2 − Phase1`
  ablation cleanly measures "what vision added."
