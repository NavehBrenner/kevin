# The Policy — Architecture of the Residual Correction Network

Companion docs: [problem-structure.md](problem-structure.md) (notation, the four streams, the learning objective) · [human-generation.md](human-generation.md) · [expert-corrections.md](expert-corrections.md). Refines [`../../project-scope.md`](../../project-scope.md) Component 4.

This document specifies `π_θ`: the network that, every control step, maps the observation `o_t` to a bounded correction `Δ`. It is the project's headline ML contribution. Symbols are defined in [problem-structure.md](problem-structure.md); read that first.

## Job, in one line

```
Δ_raw = π_θ(o_t)        Δ = clamp(Δ_raw)        command*_t = c_t ⊕ Δ → impedance controller
```

`π_θ` is trained by behavioral cloning to reproduce the expert's `Δ*_t` from non-privileged observation. It outputs a **residual** on the operator's command — never an absolute pose.

## Output

`Δ = (Δposition ∈ ℝ³, Δorientation ∈ ℝ³ axis-angle, Δgrip ∈ ℝ¹)` — 7 numbers, identical signature to the expert. The clamp (`±3 cm / ±10° / ±5 N` per step) is applied **outside** the network, before the controller, so the policy is safe-by-construction even if it emits garbage. The network itself can output unbounded reals; we may add a `tanh`-scaled head to keep raw outputs near the clamp range and ease training, but the hard safety bound is the external clamp, not the activation.

## Inputs — the four streams (recap)

From [problem-structure.md](problem-structure.md), the observation `o_t` is four streams. The recurrent core is **stateful**, so there are no history windows: each stream contributes its **current per-step value**, normalized, and all temporal memory is carried in the GRU hidden state.

| Stream | Per-step shape (tunable) | Phase | Pre-core handling |
|---|---|---|---|
| Command | `7` (pose/delta) | 1 & 2 | normalize → concat |
| F/T | `6` (bias-subtracted wrench) | 1 & 2 | normalize → concat |
| Proprioception | `~24` (EE pose 3+6D, joints 7, joint vel 7, grip 1) | 1 & 2 | normalize → concat |
| Wrist image | `224×224×3` (current frame, opt. short stack) | 2 only | **CNN encoder (pretrained-init, fine-tuned)** → `e_img` → concat |

The vector streams get **no learned per-stream encoder** — they are already low-dimensional physical features, and the GRU's input-to-hidden matrix is itself the learned mixing projection (see *Overall architecture*). Only fixed, non-learned transforms are applied to them: per-channel **normalization** (fixed train-set stats, stored for inference) and the quaternion→**6D** map for orientations. The image is the sole exception — pixels need a learned CNN encoder. Image resolution/stack depth and the GRU hidden size/layers are hyperparameters calibrated against validation curves. The hidden state is reset to zero at episode start (the model must tolerate a cold start — itself realistic).

## Overall architecture

A **single stateful recurrent core over an early-fused observation**. Each control step, every stream's current value is normalized, the image (Phase 2) is encoded by the CNN, and all are concatenated into one input vector `x_t` fed to one GRU that carries its hidden state across the whole episode (reset per episode). An MLP head maps the hidden state to the correction:

```
   command  c_t       (normalized) ┐
   F/T      ft_t      (normalized) ├─► concat ─► x_t ─► [ stateful GRU core ] ─► h_t ─► MLP head ─► Δ_raw (7) ─► clamp ─► Δ
   proprio  (3+6D…)   (normalized) │                    (1–2 layers, hidden h,
   wrist image ─► CNN ─► e_img     ┘                     h carried step→step)
                       └─► [aux head] ─► all-hole heatmap        (e_img Phase 2 only; aux training-only, λ-weighted)
```

- **Early fusion into one recurrent core.** Because the core is stateful, there are no per-stream history windows — each stream contributes its current value and all temporal memory lives in `h_t`. The streams interact *inside* the recurrence, every step, and again in the head.
- **No learned per-stream encoders for the vector streams.** Command/F-T/proprio are already low-dimensional physical features; the GRU's input-to-hidden matrix is the learned projection that mixes them, so separate per-stream MLPs would be redundant. They get only fixed transforms — normalization and quaternion→6D. **Only the image gets a learned encoder** (the CNN), because pixels require feature extraction before fusion.
- **Capacity from depth, not a deeper cell.** Add capacity by stacking GRU layers / widening the hidden state, plus the MLP head — never by replacing the cell's gated-affine transition with a deep MLP. Deepening the per-step *recurrent transition* lengthens the through-time gradient path and hurts trainability (the gated near-identity update is what lets gradients survive hundreds of steps); depth *outside* the recurrence — input side, head, stacked layers — does not have this problem.
- **The MLP head** (a few layers, ReLU/GELU → 7 outputs) is the final nonlinear mapping; cross-modal reasoning ("F/T says catching on the +x rim, command says still pushing +x, image says hole at −x → correct toward −x") now happens both in the recurrent state and the head.
- **Phase 1 vs 2 is a clean input-width change**: Phase 1 `x_t = [cmd, ft, proprio]`; Phase 2 widens it with `e_img`. Core and head are otherwise identical, preserving the `Phase2 − Phase1` ablation.
- **Latency budget**: one ~100 Hz control step (~10 ms). Statefulness makes the recurrent path **O(1) per step** (consume one `x_t`, advance `h_t`). The image CNN is the only real cost driver (Phase 2); it runs once per new frame (decimate frames if rendering throughput demands), and the all-holes aux head runs at *training* time only and is dropped at inference.

## Encoder decisions (locked)

### Decision A — temporal architecture *(locked: single stateful GRU core; windowed late-fusion as the documented alternative)*

How to integrate the time-correlated command/F-T history into the correction:

- **Single stateful GRU core, early fusion (chosen).** One GRU carries a hidden state across the whole episode (reset per episode); each step it consumes the concatenated, normalized per-step streams and advances `h_t`. Matches deployment exactly — at run time the policy holds the same `h_t` and does O(1) work per control step, with no window re-encoding — and lets the modalities interact inside the recurrence. Costs: stateful training (truncated BPTT, correlated minibatches, per-episode reset) is more involved than shuffled-window training. Chosen because train/deploy fidelity and O(1) streaming (including future vision) outweigh the training simplicity of windows.
- **Windowed, separately-encoded streams (documented alternative).** The earlier framing: a GRU (or 1D-CNN) per stream over a fixed last-`H` window → per-stream embedding → late fusion; samples are i.i.d. windows shuffled freely (lower-variance SGD, trivial batching). Rejected as primary because windowed training must be paired with window re-encoding at deploy (≈49 redundant steps recomputed per tick; worse with images), and pairing windowed training with stateful deploy is a silent train/deploy distribution mismatch. Retained as the fallback if stateful training proves fiddly, and as the "alternatives considered" contrast.
- **Transformer / attention (rejected).** The reactive horizon is short (~0.2–0.5 s) and the data modest, so attention isn't justified; and for streaming it is *worse* — vanilla self-attention is O(L²) per step, KV-caching is O(L) with a context-growing cache, both heavier than a GRU's fixed-size O(1) state. A state-space model (S4/Mamba) is the modern "stateful + long-range" option but is out of scope here.

Decision: **single stateful GRU core (early fusion); windowed late-fusion held as the documented fallback/alternative.** Capacity is added by stacking GRU layers and widening the hidden state plus the head MLP — never by deepening the cell's internal transition.

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
- **Stateful training (truncated BPTT)**: episodes are processed as ordered sequences in chunks; the hidden state is carried between chunks (detached to truncate gradients) and reset at episode boundaries, so training matches the stateful deployment. A per-step supervised loss applies at every timestep. The exact protocol — including whether a teacher-forcing-style scheme is used — is still under discussion (see *Open / to-decide*).
- **Image encoder trained jointly**: all branches — GRUs, proprio MLP, fusion head, *and* the image CNN — are trainable during BC; the CNN starts from a pretrained init and is fine-tuned end-to-end (Decision B). Optional all-holes aux loss adds `+ λ·aux` to the objective; the aux head is dropped at inference. Freeze is the fallback only.
- **Data split by episode**: train/val split at the **episode** level, never the step level — steps within an episode are highly correlated, so a step-level split leaks and inflates validation scores.
- **Volume** (scope targets): Phase 1 ~1,000 episodes (~0.5–1M frames, a few GPU-hours); Phase 2 ~5,000 episodes (~2.5–5M frames, ~10–20 GPU-hours). Calibrate by validation curves; sim throughput supports overnight regeneration.
- **Training protocol — staged (BC → DART → DAgger).** Phase 1 trains by **stateful BC** on the M4 corpus (offline; the guaranteed deliverable). Covariate-shift fixes are escalated only as closed-loop spot-checks demand:
  1. **BC (offline, default).** Regress onto the logged expert `Δ*` over the expert's state distribution. Keep failure episodes for state coverage.
  2. **DART (offline escalation).** Inject small expert-action noise at *data-gen* so the corpus covers a recovery tube around the expert path (Laskey et al., 2017). Stays fully offline — cheap; try before DAgger.
  3. **DAgger (online escalation).** Roll the policy out in sim, query the **analytical privileged expert** at the policy's *own* visited states, aggregate and retrain (Ross et al., 2011). Feasible here precisely because the expert is closed-form and free to query online — the usual human-in-the-loop cost is absent. Use a **β-decay schedule** (execute the expert with prob. β early, anneal toward the policy) and **always BC-pretrain first** — never pure on-policy from scratch. Note: only the realized *state/contact* is policy-induced; the base operator command stays scripted.

  RL is anti-scope.

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

- TBPTT truncation length (steps per backprop chunk); image resolution and stack depth.
- Loss specifics: Huber vs MSE; per-channel weights `w_pos/w_ori/w_grip`.
- Aux loss `λ` schedule (constant, or anneal to 0); exact aux target (heatmap vs per-hole keypoints).
- Whether a `tanh`-scaled output head helps training.
- GRU core sizing: hidden size and number of stacked layers.
- **Training protocol**: stateful BC via truncated BPTT (reset per episode) is locked; the exact variant — including whether a teacher-forcing-style scheme is used — is still under discussion (see *Training*).
- Inference-latency measurement once the core/CNN sizes are fixed.

---

### Provenance note

Records the residual-policy architecture from the 2026-05/06 scoping discussions, refining [`../../project-scope.md`](../../project-scope.md) Component 4 (previously deferred). Settled here: the **multi-stream-encoder + fusion-head** skeleton, BC loss shape (per-channel Huber, rotation-aware, episode-level split), and the **implicit-vs-explicit goal** alternative pair. Encoder decisions **locked** (2026-06-08): **Decision A** — GRU temporal encoders, 1D-CNN as fallback; **Decision B** — image CNN trained jointly (pretrained-init + fine-tuned end-to-end), with an optional all-holes-detection auxiliary loss (`λ` knob) and a freeze fallback. The freeze + predict-target-hole-pose plan was **rejected** (engineered bottleneck; ill-posed target selection under multiple holes without intent). Note: this updates the scope's earlier "pretrained backbone used as-is / no custom-trained vision models" line — Naveh confirmed the scope is open to change and the encoder may be fine-tuned. **2026-06-17 revision (Decision A re-opened and re-locked):** the temporal architecture changed from *windowed, separately-encoded streams with late fusion* to a **single stateful GRU core over an early-fused, normalized observation** — no learned per-stream encoders for the vector streams (the GRU input matrix is the projection; only the image keeps its CNN), capacity via stacked layers + MLP head rather than a deeper cell transition, and stateful truncated-BPTT training to match the O(1)-per-step streaming deployment. The windowed/late-fusion design is retained as the documented alternative/fallback. Training-protocol details (any teacher-forcing variant) remain under discussion. Project-internal rationale; no `raw/` source backs this page directly.
