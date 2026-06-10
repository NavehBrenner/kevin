# Human-Like Command Generation — The Scripted Noisy Human

Companion docs: [problem-structure.md](problem-structure.md) (notation, the four streams) · [expert-corrections.md](expert-corrections.md) · [policy-model.md](policy-model.md). Refines [`../../project-scope.md`](../../project-scope.md) Component 5.

This document specifies the **scripted noisy human**: the programmatic actor that produces the operator command stream `c_t` during data generation and benchmarking. It is *not* a model of human cognition — it is a controllable, seedable source of *realistically-wrong* coarse commands, so that (a) the expert has something to correct and (b) every configuration in the KPI study can be re-run under identical operator behavior.

## What it must produce, and what it must NOT be

The noisy human emits, every control step, a commanded EE pose `c_t` (position + orientation, + an optional grip signal) heading toward the target hole. The design constraint that drives everything below:

> **The noise must be structured and low-frequency, not per-step white noise.**

Why this matters — the trap to avoid: if we generated commands as `c_t = (true optimal pose at t) + independent Gaussian noise per step`, then the expert's optimal correction would be *exactly the negative of the injected noise*, and the policy's job would collapse to "denoise the command" — a trivial, unphysical task that learns nothing about contact or geometry. We explicitly reject that. Instead the noisy human commits to a **biased, drifting, coarse trajectory** that is internally consistent over time, the way a real operator's hand is wrong in a *correlated* way (a steady misjudgment of where the hole is, plus slow wobble), not in an i.i.d. way. The expert then has to reason about geometry and contact to fix it — which is the skill we actually want cloned.

## The generative model

Per episode, the noisy human's behavior is fully determined by a fixed seed and a small parameter set. Composition of three layers:

### 1. Intent — the coarse goal

At episode init, intent is set to **"navigate to and insert into hole #k"**. The actor computes a coarse target pose `g = (p_hole + bias, R_hole)`:

- `bias ∈ ℝ³` — a **per-episode constant** offset drawn once (e.g. `bias ~ N(0, σ_bias²)`, a few mm–cm). This models a real operator's *systematic* misjudgment of where the hole is. It does **not** resample each step.
- Orientation target is the hole's nominal insertion orientation, with a per-episode constant angular bias.

This biased goal is what the actor *believes* it is aiming at. It is wrong, and it stays wrong in a consistent direction for the whole episode — that consistency is what makes the correction problem non-trivial.

### 2. Trajectory — coarse motion toward the (biased) goal

The actor moves from the current commanded pose toward `g` with a simple coarse policy — e.g. a capped-rate proportional move (a "point and push" approach), not an optimal trajectory. It is deliberately unaware of contact: it will keep pushing the peg against flat wall if its biased goal sits off the hole, exactly the situation the assist layer must rescue. No knowledge of the true hole, no replanning on contact.

### 3. Noise — correlated, low-frequency perturbation

On top of the coarse trajectory, add structured perturbation, **not** white noise:

- **Slow drift**: a low-frequency random process (e.g. an Ornstein–Uhlenbeck process, or a band-limited / smoothed random walk) on position and orientation, so the command wanders coherently over hundreds of ms rather than jittering each step.
- **Optional tremor**: a small higher-frequency component *if* we want to stress smoothness, kept well below the per-step magnitude that would make denoising the whole game.
- **Update rate**: the command target refreshes at a realistic **~5–10 Hz** (the scope's figure) and is held/interpolated between refreshes to ~100 Hz, mimicking a human issuing discrete coarse intents while the controller runs fast. This rate mismatch is itself a realistic source of lag the assist must absorb.

So the full command is:

```
c_t = coarse_move_toward(g)  ⊕  drift_t  ⊕  tremor_t
g   = (p_hole + bias_episode,  R_hole · ΔR_bias_episode)     # fixed per episode
```

with `drift_t`, `tremor_t` correlated in time, and `bias_episode` constant.

## Why correction can't be "just undo the noise"

Because the noise is **not** added independently each step and the actor does **not** track the true optimum:

- The expert never sees the injected `bias`/`drift` directly — it sees the *resulting command* `c_t` and the *true geometry*, and must infer a correction. The correction is a geometric quantity (toward the real hole, respecting contact), not a recorded noise sample.
- Errors are **path-dependent**: where the peg ends up at step `t` depends on every correction applied before `t` (compliance, contact, prior `Δ`). The per-step problem is therefore *not* independent across steps — which is the realistic, harder, and intended situation. (This was the explicit concern raised in discussion: per-step independent correction would be wrong; the structured-noise design makes the steps genuinely coupled.)

## Realism vs. control — the deliberate balance

The noisy human is tuned to be *hard for the unassisted baseline but fixable by a good correction*. Per the scope, the actual noise magnitudes (`σ_bias`, drift time-constant and amplitude, tremor amplitude) are **placeholders to be calibrated against the human-only baseline** — chosen so the human-only configuration fails often enough to leave headroom, without being impossible. We do not freeze these numbers now; we freeze the *form* of the model (biased + drifting + coarse) now.

## Determinism, seeding, paired comparisons

- Every episode is fully reproducible from `(master_seed, episode_index)`: the bias draw, the drift process, the tremor, and the scene randomization all derive from it.
- For the KPI study, the **same seed list** drives both configurations (human-only / +residual). This gives *paired* comparisons: identical operator behavior, only the assist layer changes — the cleanest possible attribution of KPI differences to the assist.
- Grip signal: baseline-closed; the actor may emit an "open / release" event at episode end (the scope's disengage), but does not micro-modulate grip — grip-force micro-adjustment is left for the expert/policy to supply.

## Interfaces and placement

- Implemented as one of the three swappable **input strategies** (Strategy pattern seam), behind the same interface as the MediaPipe-vision and keyboard inputs. To the controller it is indistinguishable from a real operator — it just emits `c_t`.
- Lives on the **input** side of the pipeline; it has no access to privileged state, no knowledge of the expert, and no knowledge of the residual policy. (The expert and policy consume `c_t` as the command stream / command-history input — see [problem-structure.md](problem-structure.md).)
- Per the code conventions, the scripted-human driver is a runnable script under `scripts/` (data-gen driver), and its noise model is config-driven (YAML/Hydra).

## Open / to-calibrate

- Exact drift process (OU vs. smoothed random walk) and its time-constant — pick empirically for visual realism.
- Whether to include the high-frequency tremor at all in the headline runs.
- Noise magnitudes — **deferred to post-baseline calibration**, by design.

---

### Provenance note

Records the structured-noise design agreed in the 2026-05/06 scoping discussions, refining [`../../project-scope.md`](../../project-scope.md) Component 5 (which left "Gaussian vs structured noise, drift dynamics" open). The decisive call captured here: **structured low-frequency biased noise**, explicitly rejecting per-step i.i.d. noise to avoid the trivial "negate-the-noise" expert. Project-internal rationale; no `raw/` source backs it directly.
