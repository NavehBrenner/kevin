# Milestone 7 — Vision-Conditioned Residual — Phase 2

**Goal (as scoped)**: extend the Phase-1 F/T-only residual with a **vision stream** (wrist-camera
frames → image CNN → the residual GRU) so the policy can infer the hole location it *cannot* read
from command/F-T/proprioception in free space — and thereby lift closed-loop insertion success
**outside** the narrow chamfer-contact band that caps Phase 1. DAgger and a stronger analytical
expert were the two fallback levers if plain BC-on-more-data stalled.

> **This spec is retrospective (written 2026-07-23, closes LAB-79).** M7 was the project's
> largest arc and — uniquely — was executed without a spec written first. It is recorded here
> **as it actually happened**, ending in a **documented negative result**: vision, DAgger, and a
> better expert were each explored and **none lifted closed-loop seating success**. The
> deliverable of M7 is the **mechanism** that explains why — a transferable finding about the
> whole imitation-learning family on this task — not a success number.
>
> **Read every rate below at its power (LAB-114).** M7's closed-loop comparisons are single
> checkpoints at **20 eval seeds**; the training-seed noise floor was later measured at **18 pp**
> and a 20-seed arm carries a ±20 pp exact interval. So M7's *directional* margins (35 vs 40; 40
> vs 30) are draws — its honest conclusion is **"no vision benefit was detectable at this
> power"**, and the large margins that survive (40 vs 10) plus the *mechanism* carry the result.
> Full numbers: [`results/kpi-dashboard.md`](results/kpi-dashboard.md) §4/§6. Mechanism:
> wiki `synthesis/imitation-limits-closed-loop`, `concepts/vision-conditioned-policy`,
> `concepts/training-seed-variance`.

## Definition of done (as met)

M7 is complete because the question it asked is **answered with a mechanism**, not because a
target was hit:

- The **vision deploy path exists and works** end-to-end (LAB-83): a `--vision` checkpoint loads
  through the same `--policy tf` / `--vision-checkpoint` path, the wrist camera is enabled
  automatically, and the image→`Observation`→`LearnedResidual` pipeline runs in the loop.
- The vision policy was **trained and evaluated** against F/T-only and human-only in a paired
  ablation (`eval_stageC*`), across the in-band (es0.4) and flat-wall (es1.0) operating points.
- **DAgger** (3 on-policy rounds) and a **better analytical expert** (five slam-prevention knobs)
  were built, run, and measured.
- The negative result is **explained at the mechanism level** and the explanation is
  checkpoint-independent (theory + exact/byte-identical probes), so it survives LAB-114.

## What was in M7 · what was not

**In scope (built):**
- Vision-conditioned `LearnedResidual`: MobileNetV3-small image encoder → 128-d embedding →
  concatenated into the GRU input alongside command/F-T/proprioception (LAB-83).
- The vision **deploy path** (env wrist-camera capture, frame cadence, `PolicyConfig.use_vision`
  as the modality switch) and the 3-way eval (`--ftonly-checkpoint`/`--vision-checkpoint`).
- Small-GPU training economics (LAB-82): frozen-encoder + batch-2 to fit an 8 GB laptop; Stage-C
  VRAM cuts (AMP, gradient-checkpointing, encode-chunking) to fine-tune an unfrozen backbone.
- **DAgger** on-policy relabeling (LAB-105/106) and a **slam-preventing expert** (LAB-108).

**Anti-scope (deferred, deliberately):**
- RL / reward-based fine-tuning — out of scope for a solo, deadline-bound project (needs a reward
  + sim loop + tuning far beyond the budget).
- A contact-recovery state machine and operator-side (contact-aware scripted human) fixes — the
  latter changes the *task*, not the policy. Both are carried into the go-forward decision (D-6),
  not M7.
- MediaPipe Holistic / full-arm tracking (an M8 stretch, never M7).

## What actually happened — the arc

Executed as a sequence of LAB issues, each narrowing the hypothesis:

1. **Vision deploy path + first training (LAB-82/83).** The image stream shipped and ran in the
   loop. On an 8 GB laptop only a **frozen** encoder at batch-2 fit; render throughput (~10 fps)
   capped the practical corpus at ~300 episodes (`dataset_vision`, 300 ep, seed 82).
2. **Offline, vision looked strictly better (the trap).** Held-out BC error: vision **6.94 mm**
   vs F/T **7.63 mm** vs the zero-Δ prior **4.75 mm** — and vision had the best `best_val_loss` of
   the whole arc (`vision_frozen_lab82`, 0.00107). **Both modalities still under-fit the zero-Δ
   prior overall.**
3. **Closed-loop, vision did not separate from F/T.** In the paired ablation
   (`eval_stageC_band04`, es0.4, 20 seeds): human 35% / F/T 40% / **vision 40%** — a tie in-band;
   out-of-band (es1.0) vision 10% vs F/T 20%, inside the noise floor. Unfreezing the encoder
   (Stage C, `vision_stageC`) did not change the verdict.
4. **Fixing the offline metric made closed-loop worse (LAB-104/106).** Two interventions —
   position-loss ×10 + weight-decay, then a `command_ee_delta` **feedback feature** — drove
   offline error to a record **3.46 mm (beating the prior for the first time)** and **collapsed
   closed-loop to 10%** (`ftonly_gate_wpos10_wd`), with *more* force-aborts. A more accurate
   imitator was a worse controller.
5. **DAgger degraded rather than rescued (LAB-105/106).** Three F/T rounds on the ar100 base:
   **40% → 30% → 15%** (rollout success 0.325 → 0.25). The bounded expert cannot demonstrate a
   recovery from the force-abort states the policy actually visits.
6. **A better expert was refuted (LAB-108).** Five slam-prevention knobs were all inert; the
   expert's own ceiling stayed ~73.3%. The binding constraint is operator-originated, pre-contact
   force-abort — which a bounded residual cannot fix.

## The result (the mechanism is the deliverable)

M7's transferable finding, and why it does not depend on any single checkpoint:

- **Identifiability ceiling (LAB-77).** The operator's command already proxies the hole location,
  so vision carries little *marginal* signal; the free-space correction a clone would learn is ≈0
  by construction. Theory + byte-identical parameter sweeps.
- **Far-field gating failure (LAB-106).** Trained GRUs emit a ~5.6 mm correction floor across the
  ~60% of steps that are free-space, where the expert is *exactly* zero — a constant correction
  where a distance-gated one is needed. Exact error-decomposition probe.
- **Offline/closed-loop anti-correlation (LAB-106).** Per-step BC fidelity is *anti-predictive* of
  seating success here; only a closed-loop ablation is a valid signal. This is the single most
  important cross-cutting M7 finding.
- **The bounded-expert / DAgger argument (LAB-105/106).** On-policy relabeling can only teach
  behaviors the expert can *perform*; a bounded, distance-gated analytical expert is not competent
  on the force-abort regime, so DAgger's founding premise is structurally violated.

**What could actually lift success — all OUTSIDE per-step imitation** (feeds the go-forward
decision D-6, not M7): a contact-aware operator, a contact-recovery controller, or RL with a
seating reward. Per-step BC on this task has been shown, mechanistically, to be at its ceiling.

## The LAB-114 recalibration (applied 2026-07-23)

M7's numbers were single checkpoints at 20 seeds. After the training-seed spread was measured
(18 pp), M7's **directional** claims were down-weighted to *"no benefit detectable at this
power"*, while the **mechanism** claims — which rest on theory and exact probes, not point
estimates — stand unchanged. Crucially, **underpowering cannot manufacture a null**: a weak test
failing to find an effect is "not shown", which is close to what M7 already concludes. (The
Phase-1 failure was the opposite and worse — a *positive* manufactured by two lucky draws.) So M7
needed its wording audited, not its experiments re-run. No new M7 compute was spent.

## Files this milestone touched (principal)

- `src/ai_teleop/policy/` — `residual_policy.py` (vision branch), `model.py`, `config.py`
  (`use_vision`, `use_command_ee_delta`), `train.py` (Stage-C VRAM controls).
- `src/ai_teleop/input/` — wrist-camera capture on the deploy path.
- `src/ai_teleop/eval/ablation.py` — `--ftonly-checkpoint` / `--vision-checkpoint` 3-way.
- `scripts/dagger.py` — on-policy DAgger relabel loop.
- `data/dataset_vision*` — the 300-episode vision corpus; `dagger_ft_agg` (aggregated).

## Handoff

M7 closes Phase 2. The go-forward decision (D-6, `docs/review/go-forward.md`) weighs whether to
pursue any of the out-of-imitation levers or to declare the training arc a documented negative and
spend the remaining budget strengthening what already works. The Phase-1 result it strengthens is
itself under revision (LAB-114) — see [`results/kpi-dashboard.md`](results/kpi-dashboard.md).
