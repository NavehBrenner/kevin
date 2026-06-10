# The Policy — Architecture of the Residual Correction Network

Companion docs: [problem-structure.md](problem-structure.md) (notation, the four streams, the learning objective) · [human-generation.md](human-generation.md) · [expert-corrections.md](expert-corrections.md). Refines [`../../project-scope.md`](../../project-scope.md) Component 4.

This document specifies `π_θ`: the network that, every control step, maps the observation `o_t` to a bounded correction `Δ`. It is the project's headline ML contribution. Symbols are defined in [problem-structure.md](problem-structure.md); read that first.

## Job, in one line

```
Δ_raw = π_θ(o_t)        Δ = clamp(Δ_raw)        command*_t = c_t ⊕ Δ → impedance controller
```

`π_θ` is trained by behavioral cloning to reproduce the expert's `Δ*_t` from non-privileged observation. It outputs a **residual** on the operator's command — never an absolute pose.

## Output

`Δ = (Δposition ∈ ℝ³, Δorientation ∈ ℝ³ axis-angle, Δgrip ∈ ℝ¹)` — 7 numbers, identical signature to the expert. The clamp (`±2 cm / ±10° / ±5 N` per step) is applied **outside** the network, before the controller, so the policy is safe-by-construction even if it emits garbage. The network itself can output unbounded reals; we may add a `tanh`-scaled head to keep raw outputs near the clamp range and ease training, but the hard safety bound is the external clamp, not the activation.

## Inputs — the four streams (recap)

From [problem-structure.md](problem-structure.md), the observation `o_t` is four independent streams, each with its own small encoder:

| Stream | Shape (example, tunable) | Phase | Encoder |
|---|---|---|---|
| Command history | `H_c×7` (pose/delta, last ~50 steps) | 1 & 2 | **GRU** → `e_cmd` |
| F/T history | `H_f×6` (bias-subtracted wrench, last ~20 steps) | 1 & 2 | **GRU** → `e_ft` |
| Proprioception | `~24` (EE pose 3+6D, joints 7, joint vel 7, grip 1) | 1 & 2 | MLP → `e_pro` |
| Wrist image | `128×128×3` (current frame, opt. short stack) | 2 only | **CNN encoder, pretrained-init + fine-tuned end-to-end** → `e_img` |

`H_c`, `H_f`, the image resolution and stack depth are hyperparameters calibrated against validation curves. Histories are zero-padded at episode start (the model must tolerate a partially-filled buffer; that is itself realistic — early in an episode there is little history).

## Overall architecture

A **multi-stream encoder + fusion MLP head**:

```
   command window (H_c×7) ──► GRU ──────────────► e_cmd ┐
   F/T window     (H_f×6) ──► GRU ──────────────► e_ft  │
   proprioception (~24)   ──► MLP ──────────────► e_pro ├─► concat ─► MLP fusion head ─► Δ_raw (7) ─► clamp ─► Δ
   wrist image (128²×3) ──► CNN (fine-tuned) ───► e_img ┘   (e_img present in Phase 2 only;
                              └─► [aux head] ──► all-hole heatmap   aux branch, training only, λ-weighted)
```

- Each stream is encoded **separately** then fused by concatenation — different modalities, different natural encoders, fused late. This keeps the architecture identical between phases: Phase 1 simply drops the `e_img` branch from the concat.
- The fusion head is a small MLP (a few layers, ReLU/GELU) → 7 outputs. This is where cross-modal reasoning happens ("F/T says catching on the +x rim, command says still pushing +x, image says hole is at −x → correct toward −x").
- **Latency budget**: must run inside one ~100 Hz control step (~10 ms). The image encoder is the cost driver (its weights are fine-tuned at *training* time, but *inference* is a plain forward pass — fine-tuning adds no runtime cost over a frozen encoder). The aux head runs only during training and is dropped at deployment. We decimate camera frames if needed (the scope flags rendering throughput as a risk).

## Encoder decisions (locked)

### Decision A — temporal encoder family for the history streams *(locked: GRU, 1D-CNN as fallback)*

How to encode the command/F/T windows:

- **GRU / small RNN (chosen).** Carry a recurrent hidden state across steps; naturally handles variable-length history and the time-correlated structure of the command/F-T streams (intent emerges over hundreds of ms, contact evolves over a contact event). Costs: stateful inference (reset hidden state per episode), slightly trickier to batch. Chosen because a recurrent model is the more natural fit for "integrate an evolving history into an intent/contact estimate."
- **1D-CNN over a fixed zero-padded window (fallback).** Treat the `H×C` history as a 1D sequence, convolve over time. Fixed-size input, fast, stateless, trivially batched, deterministic latency. The documented fallback for the "alternatives considered" slide, and the thing to switch to if the GRU proves fiddly to train or its statefulness causes deployment headaches.
- **Transformer / attention (rejected).** Overkill for ~20–50 step windows; latency and data cost not justified at this scale.

Decision: **GRU primary, 1D-CNN as the viable fallback.** The two share the same input/output contract (history window in, `e_*` embedding out), so swapping A→fallback is a localized change to the encoder modules, not an architecture rework.

### Decision B — the image encoder (Phase 2) *(locked: pretrained-init, fine-tuned end-to-end, with an optional all-holes auxiliary loss)*

The image encoder is **trained jointly with the rest of the network** (not frozen), so it can learn whatever representation actually helps the correction — including target-agnostic features like *all* holes in the frame and their spatial relations — and let the fusion head select the target using the command/F-T intent it alone sees. Two earlier framings were rejected for concrete reasons:

- **Why not freeze + predict the target hole's pose (the original "privileged-aux" plan)?** Two problems. (1) It's an *engineered bottleneck*: constraining the encoder to reproduce a single 2-D hole coordinate is a task a Laplace/blob kernel nearly solves — a poor use of a CNN, and a ceiling on what the encoder can represent. (2) It's *ill-posed under multiple holes*: target identity is determined by intent, which lives in the command/F-T streams that the image encoder does **not** receive. Asking the encoder to localize *the target* forces it to guess which hole is the target — impossible from pixels alone.
- **Why end-to-end fixes this.** Even though the encoder still gets no direct F/T or command input, gradients from the fusion head (which *does* see them) flow back into it. So the encoder is free to learn a target-agnostic scene representation while the head does intent-conditioned selection. That division of labor — *encoder represents the scene, head picks the target using intent* — is exactly what a fixed hole-pose head cannot express.

The locked design, from safest to richest (the implementation spectrum / ablation axis):

1. **Pretrained init, fine-tuned end-to-end (chosen default).** Start the CNN from a small pretrained backbone (e.g. MobileNet/ResNet-class) and let BC gradients fine-tune it jointly. You get the rich, freely-learned representation, but starting from decent generic features rather than noise — directly attacking the vision-BC data-hunger risk at near-zero extra cost.
2. **Optional all-holes detection auxiliary loss (a λ knob, not a commitment).** Add a side branch off the encoder feature map predicting a heatmap/segmentation of **every** hole in the frame (well-posed — no intent needed, no target-guessing), trained with `BC loss + λ·aux`. It *shapes* the encoder early and stabilizes convergence without bottlenecking it: the embedding `e_img` flowing to the fusion head stays high-dimensional and free. Dial `λ→0` to recover pure end-to-end. The aux head is training-only and dropped at inference.
3. **Freeze fallback.** If joint training proves unstable or too slow within the time budget, freeze a pretrained backbone (or an autoencoder latent) as a stateless feature extractor. This is the *fallback*, not the plan — it sacrifices the learned-representation upside for training stability.

Decision: **pretrained-init + fine-tune end-to-end (1)**, with **the all-holes aux loss (2) as a tunable stabilizer**, and **freeze (3) held as the safety fallback.** This keeps the encoder free to learn hole relations and richer scene structure (the goal), while two independent knobs — the aux `λ` and the freeze fallback — bound the schedule risk that from-scratch-style vision training can blow up.

### Schedule note (why the fallbacks exist)

Pure from-scratch end-to-end vision BC is the highest-variance part of the project: it couples learning *perception* and *correction* from the same demonstrations, off a weak bounded-regression signal, and when it fails to converge the cause is ambiguous (data? encoder? loss weights?). For a solo build where vision is already the tighter Phase-2 add (~30–40 h over Phase 1), pretrained-init and the aux-`λ` knob are cheap insurance, and the freeze fallback guarantees Phase 2 still ships even if joint training stalls. **Phase 1 (F/T-only) remains the guaranteed deliverable regardless** — none of this touches it.

## Training — behavioral cloning

- **Objective**: per-channel weighted regression of the policy output onto the expert's clamped correction:
  ```
  L = w_pos · Huber(Δ̂.pos, Δ*.pos) + w_ori · Huber(Δ̂.ori, Δ*.ori) + w_grip · Huber(Δ̂.grip, Δ*.grip)
  ```
  Separate weights because the channels have different units/scales (cm vs rad vs N) and different importance. Huber (smooth-L1) over plain MSE for robustness to the occasional large expert correction; MSE is the simpler fallback.
- **Orientation handling**: regress in a continuous representation (6D rotation or axis-angle) — never raw Euler/quaternion-with-sign-ambiguity — and compute the loss as a proper rotation difference (geodesic / `log(R̂·R*ᵀ)`), not naive component subtraction.
- **Image encoder trained jointly**: all branches — GRUs, proprio MLP, fusion head, *and* the image CNN — are trainable during BC; the CNN starts from a pretrained init and is fine-tuned end-to-end (Decision B). Optional all-holes aux loss adds `+ λ·aux` to the objective; the aux head is dropped at inference. Freeze is the fallback only.
- **Data split by episode**: train/val split at the **episode** level, never the step level — steps within an episode are highly correlated, so a step-level split leaks and inflates validation scores.
- **Volume** (scope targets): Phase 1 ~1,000 episodes (~0.5–1M frames, a few GPU-hours); Phase 2 ~5,000 episodes (~2.5–5M frames, ~10–20 GPU-hours). Calibrate by validation curves; sim throughput supports overnight regeneration.
- **Covariate-shift mitigations** held in reserve: keep failure episodes (state coverage), optionally inject small expert-action noise at data-gen so the expert demonstrates recovery, and escalate to **DAgger** if open-loop rollouts drift. RL is anti-scope.

## Phase 1 vs Phase 2

| | Phase 1 (F/T-only) | Phase 2 (vision-conditioned) |
|---|---|---|
| Streams in `o_t` | command, F/T, proprio | + wrist image |
| Architecture | fusion head over 3 encoders | + fine-tuned image-CNN branch (+ optional aux head) |
| Hole localization | none — operator's coarse command brings the peg to the hole vicinity; the policy only corrects alignment on contact | policy reads the hole from the camera and sharpens the approach before contact |
| Role | guaranteed deliverable; isolates contact-reactive alignment | headline; the ablation `Phase2 − Phase1` measures "what vision added" |

The architecture is intentionally the *same* skeleton across phases — Phase 2 only widens the input — so the comparison is clean.

## Implicit vs explicit goal — the alternative pair (for the design-review writeup)

How the policy obtains the coarse goal it's correcting toward is itself a design fork worth presenting as "alternatives considered":

- **Implicit (chosen).** The goal is **never an explicit input**. The policy infers intent end-to-end from the command-history stream (and, in Phase 2, the image), and emits a correction directly. No external pointing device, no goal estimator. One network, one objective. This is the formulation specified above.
- **Explicit (alternative).** A separate stage estimates a coarse target pose — e.g. once per ~second, take the recent operator pose-deltas, extrapolate the implied trajectory toward the wall to *compute* a coarse target position (replace or moving-average across updates) — and feed that estimated goal as an extra input to the correction network. More interpretable and debuggable (you can inspect the estimated goal), modular; but adds a hand-designed stage, a second failure surface, and a tuning burden, and risks the "we assumed an external estimator" critique.

We chose **implicit** to keep the contribution a single learned mapping from honest observation to correction, and because the explicit goal-estimator edges toward assuming an external system (eye-tracking-like) that we deliberately don't have. The explicit variant is documented as the principled alternative and a natural ablation if the implicit policy struggles to acquire intent.

## What is NOT an input (guarding the framing)

The policy never receives privileged true poses, no external/top-down scene camera, and no externally-supplied goal pose (that's the rejected explicit variant). Feeding any of these would trivialize perception and ruin the project's premise (see [problem-structure.md](problem-structure.md) and the scope's sensing table).

## Open / to-decide (carry into the next discussion)

*(Decisions A and B are now locked — see "Encoder decisions" above.)*

- History lengths `H_c`, `H_f`; image resolution and stack depth.
- Loss specifics: Huber vs MSE; per-channel weights `w_pos/w_ori/w_grip`.
- Aux loss `λ` schedule (constant, or anneal to 0); exact aux target (heatmap vs per-hole keypoints).
- Whether a `tanh`-scaled output head helps training.
- GRU specifics: hidden size, layers, shared vs separate GRU for command and F/T streams.
- Inference-latency measurement once the encoder sizes are fixed.

---

### Provenance note

Records the residual-policy architecture from the 2026-05/06 scoping discussions, refining [`../../project-scope.md`](../../project-scope.md) Component 4 (previously deferred). Settled here: the **multi-stream-encoder + fusion-head** skeleton, BC loss shape (per-channel Huber, rotation-aware, episode-level split), and the **implicit-vs-explicit goal** alternative pair. Encoder decisions **locked** (2026-06-08): **Decision A** — GRU temporal encoders, 1D-CNN as fallback; **Decision B** — image CNN trained jointly (pretrained-init + fine-tuned end-to-end), with an optional all-holes-detection auxiliary loss (`λ` knob) and a freeze fallback. The freeze + predict-target-hole-pose plan was **rejected** (engineered bottleneck; ill-posed target selection under multiple holes without intent). Note: this updates the scope's earlier "pretrained backbone used as-is / no custom-trained vision models" line — Naveh confirmed the scope is open to change and the encoder may be fine-tuned. Project-internal rationale; no `raw/` source backs this page directly.
