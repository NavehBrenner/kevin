# Problem Structure — What We Are Learning

Companion docs: [human-generation.md](human-generation.md) · [expert-corrections.md](expert-corrections.md) · [policy-model.md](policy-model.md) · [evaluation-protocol.md](evaluation-protocol.md). The authoritative high-level scope is [`../../project-scope.md`](../../project-scope.md); this file pins down the *learning problem* those documents assume.

This document fixes notation and states, precisely, the inputs, the ground truth, and the supervised target for the residual policy. Read it first — the other three design docs use the symbols defined here.

## The setup in one paragraph

A human operator drives a Franka Panda toward a chamfered hole using coarse 6-DoF commands. Their command is good at the centimeter scale and bad at the millimeter scale. A learned **residual policy** watches the same situation the robot is in and adds a small bounded correction to the operator's command, every control step, so that the noisy approach still lands the peg. The policy is trained by **behavioral cloning** to imitate an analytical expert that *does* have privileged access to the true geometry. The whole point is that the deployed policy must reproduce the expert's corrections from **non-privileged observation alone**.

## Frames and notation

- **World frame** `W`, origin at the robot base, `z` up. Every pose below is expressed in `W` (one convention, declared once — same rule as the scope doc).
- A **pose** is `(position ∈ ℝ³, orientation ∈ SO(3))`. Orientation is stored as a rotation matrix `R` internally and as a 6D continuous representation when fed to the network (see [policy-model.md](policy-model.md)).
- Discrete control step index `t`, running at ~100 Hz.

Key symbols, used consistently across all four docs:

| Symbol | Meaning |
|---|---|
| `c_t` | operator command at step `t` — a commanded EE pose (+ optional grip). The **noisy human** in training; a real or scripted operator at eval. |
| `o_t` | the policy's **observation** at step `t` (the four sensor streams below). Privileged state is **not** in `o_t`. |
| `s_t` | privileged true state (true peg-tip pose, true hole pose, full arm state). Available to the expert and the logger **only**. |
| `Δ` | a correction: `(Δposition ∈ ℝ³, Δorientation ∈ ℝ³ axis-angle, Δgrip ∈ ℝ¹)`, always clamped. |
| `Δ*_t` | the **expert's** correction at step `t` — the behavioral-cloning target. |
| `π_θ` | the residual policy network, `Δ_raw = π_θ(o_t)`. |
| `command*_t` | augmented command actually sent to the controller: `command*_t = c_t ⊕ clamp(Δ)`. |
| `⊕`, `⊖` | pose composition / difference (position adds; orientation composes via rotation multiply, differences via `log(R_a · R_bᵀ)`). |
| `p_tip`, `p_hole` | true peg-tip position and hole-entry position (components of `s_t`). |
| `n` | unit insertion axis of the target hole (points into the hole). |
| `d` | scalar approach distance, `‖p_tip − p_hole‖`. |
| `g(d)` | the expert's distance gate, smoothstep, `≈0` far from the hole, `→1` at contact. |

## The control equation

Every step, the deployed system computes:

```
Δ_raw      = π_θ(o_t)            # network output (unbounded)
Δ          = clamp(Δ_raw)        # safety clamp, applied BEFORE the controller sees it
command*_t = c_t ⊕ Δ            # operator command + correction
→ impedance controller → MuJoCo step
```

Clamp bounds (from the scope's safety envelope): `Δposition ≤ 3 cm/step` (±2 cm before the LAB-100 recalibration), `Δorientation ≤ 10°/step`, `Δgrip ≤ 5 N/step`. Setting `Δ = 0` recovers the unassisted operator exactly — that is the "human-only" baseline for free.

This is a **residual** formulation: the policy never commands an absolute pose. It only nudges. The stable backbone (impedance control + passive chamfer alignment) does the heavy lifting; the policy supplies the last-millimeter intent the operator's coarse command lacks.

## What the policy observes — `o_t`

Four **independent** sensor streams. "Independent" is load-bearing: none is derivable from the others, which is *why* the policy needs all of them. (This was the central clarification in the design discussion — see the note at the bottom.)

1. **Command history** — the operator's recent commanded poses/deltas (last `H_c ≈ 50` steps, ~0.5 s). This is the channel of **intent**: where the operator is trying to go. A single instantaneous command is a local nudge, not a destination — the *coarse goal* only emerges from watching the trajectory of commands over time. Phase 1 and Phase 2.
2. **Force/torque history** — the 6-axis wrist wrench (3 force + 3 torque), gravity-bias-subtracted, last `H_f ≈ 20` steps. This is the channel of **contact reality**: what the environment is doing back to the peg ("where is it catching?"). It carries *zero* position information in free space. Phase 1 and Phase 2.
3. **Proprioception** — actual EE pose (position + 6D rotation), joint angles (7), joint velocities (7), gripper width (1). This is the channel of **actual state**: where the arm really is *now*. Phase 1 and Phase 2.
4. **Wrist camera image** — current RGB frame (optionally a short stack), e.g. 224×224×3. This is the channel of **where the world is**: it locates the target hole during approach. **Phase 2 only.**

### Why no single stream substitutes for another

- F/T is **not** derivable from the command stream: it is a measured environment reaction. In free space it is ~0 regardless of command; on contact it depends on geometry the command knows nothing about.
- Actual EE pose is **not** derivable from F/T: F/T has no position content at all.
- Actual EE pose is **not** the integral of the command stream either: (a) impedance compliance lets the real pose deflect from the commanded pose under contact, and (b) the command channel excludes the policy's own `Δ`, so integrating commands would miss every correction already applied.

So `(command, F/T, proprioception)` are three genuinely orthogonal views, and the camera adds a fourth (exteroceptive) view in Phase 2. The learning problem is: **fuse these four partial views into the correction the expert computed from full privileged state.**

## What is privileged (and therefore *not* an input) — `s_t`

The expert sees the **true** peg-tip pose and **true** hole pose directly; the deployed policy never does. The gap between "expert with `s_t`" and "policy with `o_t`" is the entire research contribution. The privileged channels:

- True peg-tip pose and true target-hole pose (and thus `p_tip`, `p_hole`, `n`, `d`).
- Full arm state ground truth.

These are logged per step for **offline analysis only**, never fed to the deployed network. (In Phase 1 the policy has no exteroceptive sense at all — the operator's coarse command does the localizing and the F/T residual handles contact-reactive alignment; there is no controller-side hole prior. See the scope's sensing table.)

## The ground truth / supervised target

The training label at each step is the **expert's clamped correction `Δ*_t`**, computed by the analytical privileged-info expert (full algorithm in [expert-corrections.md](expert-corrections.md)). One logged training row is:

```
( o_t ,  Δ*_t ,  s_t [analysis only] ,  success flags )
```

The behavioral-cloning objective is to make the network reproduce the expert's correction from observation alone:

```
minimize  E_t [ L( π_θ(o_t) ,  Δ*_t ) ]
```

with `L` a per-channel weighted MSE/Huber over `(Δposition, Δorientation, Δgrip)`. Full loss, weighting, and data-split details live in [policy-model.md](policy-model.md).

Crucially we clone the expert's **action**, not the true pose. The policy is never asked to regress `p_hole`; it is asked to regress *what to do about it*. This keeps the target in the same bounded `Δ`-space as the deployed output and makes the imitation contract exact (expert and policy share an output signature — the scope's "symmetric output contract").

## Why behavioral cloning is the right frame (and its risk)

BC turns "act well under partial observation" into ordinary supervised regression: free, abundant, perfectly-labeled data because the expert and the noisy human are both scripted (see [human-generation.md](human-generation.md)). The known failure mode is **covariate shift / compounding error**: a policy trained only on the expert's near-optimal state distribution can drift into states the expert never visited, where its corrections degrade and errors snowball. Mitigations we hold in reserve: keep failure episodes in the dataset for state coverage (already the scope's policy), inject small action/observation noise during data-gen so the expert demonstrates *recovery*, and **DAgger** as the escalation if open-loop BC underperforms. RL is explicit anti-scope.

## Two phases, same problem

- **Phase 1 — F/T-only residual (contact-reactive alignment).** `o_t` = streams 1–3 (no image). The operator's coarse command brings the peg to the hole; the policy feels the contact and corrects alignment to seat it. No localization — its value is the last-millimeter insertion. Guaranteed deliverable; isolates the contact-reasoning contribution.
- **Phase 2 — vision-conditioned residual (smart approach).** `o_t` = streams 1–4. The wrist camera lets the policy locate the hole and sharpen the approach before contact; the same contact-alignment ability then carries the peg home. Headline configuration.

The problem structure is identical across phases; only the observation `o_t` widens. That is deliberate — it makes the Phase-1-vs-Phase-2 ablation a clean "what did vision add?" comparison.

---

### Provenance note

This document records design decisions reached in the scoping discussions of 2026-05 / 2026-06, refining [`../../project-scope.md`](../../project-scope.md) (Component 4, previously "deferred"). The key refinement over the scope draft: the explicit treatment of the **four independent observation streams** and the clarification that intent/goal is inferred from **command history**, not supplied by any external pointing device. No `raw/` literature source backs this page directly — it is project-internal design rationale.
