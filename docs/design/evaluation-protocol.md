# Evaluation Protocol — KPIs and Validation Design

Companion docs: [problem-structure.md](problem-structure.md) · [human-generation.md](human-generation.md) · [expert-corrections.md](expert-corrections.md) · [policy-model.md](policy-model.md). The authoritative high-level scope is [`../../project-scope.md`](../../project-scope.md); this file pins down *how success is measured and defended*.

This document fixes the evaluation methodology: the configurations compared, the KPIs, and — most importantly — the experimental design that makes the headline claim statistically and methodologically defensible. The deck's "Evaluation plan" slide is the summary; this is the rationale behind it.

## The headline claim

The project exists to prove **one** comparison: **insertion success rate with the residual policy engaged vs. with it off**, on the same task. Everything else (time, force, smoothness) is supporting evidence.

We state the target as an **absolute** bar — assisted reaches ≈100% success — rather than a purely relative "more than without." Rationale: the unassisted baseline is not guaranteed to be low (a careful operator might already reach ~95% on an easy task), so a relative claim could be a tiny, unconvincing delta. An absolute near-100% under a task where the baseline is meaningfully below ceiling is the stronger, cleaner statement.

### Difficulty calibration is what makes the claim meaningful

If the unassisted baseline is already near ceiling, there is no headroom to demonstrate value and the result looks like noise. So **task difficulty is deliberately calibrated**, set so the unassisted human (or scripted human) baseline sits meaningfully below 100%.

Two **orthogonal** geometric knobs do most of the work, because each tightens a *different* error axis:

- **Peg/hole clearance → position-accuracy demand.** Tight clearance = small gap between peg and hole (e.g. a 9.5 mm peg in a 10 mm hole = 0.5 mm clearance). The smaller the gap, the more precisely the peg *tip* must be positioned over the hole to enter. Note: the peg is always *smaller* than the hole — "tight" means *small clearance*, never a larger peg.
- **Chamfer angle → orientation-accuracy demand.** The chamfer is the funnel that passively corrects small misalignments on contact. A shallow/short chamfer tolerates less angular error, so the peg must be more closely *aligned* with the insertion axis to catch. This is the rotational counterpart to clearance.

The two interact at the extremes (a very generous value on one axis makes the other axis's task easier too), but in the working range each independently controls one accuracy measure — clearance the translational tolerance, chamfer the rotational tolerance. Secondary knobs (command-noise magnitude, force cap, timeout) tune difficulty further without touching geometry.

These magnitudes are placeholders until **calibrated against the human-only baseline**, so the task is genuinely hard rather than trivial or impossible.

## Configurations compared

Two runtime configurations over the same trials:

1. **Human-only (off)** — no assistance, Δ=0. Baseline.
2. **Human + residual policy (vision)** — policy on. The headline configuration.

Both run on the same always-on impedance backbone + Δ-clamp / force cap; the *only* thing that changes between them is whether the policy contributes a correction. Plus an **ablation**: the Phase-1 F/T-only residual on the same trials, isolating the contribution of vision.

(An earlier draft included a third "Human + heuristic" config — a hand-coded spiral search. That mode was cut: spiral search added engineering time without strengthening the headline claim, which is purely *assist on vs off*. The impedance backbone and clamp it relied on are retained as the always-on substrate, not as a comparison config.)

## KPIs (per trial)

- **Insertion success** (bool) — the headline metric.
- **Time-to-insert** (s).
- **Peak contact force** (N) — safety proxy.
- **Contact events before success**.
- **Trajectory smoothness** (integrated jerk).

## Experimental design — the part that makes it defensible

Two experiments answer **different questions**; the strongest thesis reports both and is explicit about which is primary.

### Primary: scripted noisy-human, paired seeds (internal validity)

Run the [scripted noisy human](human-generation.md) with **fixed per-episode seeds**, each seed executed once with the policy and once without. Because the command stream is identical across the two runs of a seed, **the only thing that differs is whether the policy is on** — this isolates the policy's effect with zero operator variance and high statistical power. This is the **primary, rigorous, reproducible** result.

### Secondary: live human study (external validity)

A real operator (the author) performs ~100 trials with/without assistance across hole sizes. This proves the assist transfers to a **real teleoperator** — the result that makes this a *teleoperation* contribution, not just a control result. It is noisier and lower-power, so it is the **external-validity bonus**, not the core claim. Time-permitting; if not, the scripted paired-seed result stands on its own.

The live study carries two confounds that the design must neutralize:

- **Learning effect** — the operator improves between trials. Mitigation: counterbalanced order — for each seed, perform one assisted and one unassisted trial, alternating which comes first across seeds, so improved success cannot be attributed to a second attempt.
- **Experimenter bias** — an invested author may (even unintentionally) underperform unassisted to favor the system. Mitigation, strongest form: **blinding** — whether assistance is active is drawn randomly and hidden, one trial per seed, with enough seeds that assisted seeds cannot be claimed to have been easier.

Tension to acknowledge openly: the **paired-counterbalanced** design maximizes power but allows learning; the **blinded single-trial** design removes bias but needs more seeds. The scripted paired-seed experiment sidesteps both problems entirely, which is exactly why it is primary.

## Success criteria (from the deck)

- Working webcam → robot demo, assistance toggleable at runtime.
- Phase 1 beats human-only on success; peak force bounded by construction.
- Phase 2 beats human-only on success **and** peak force, statistically; and beats Phase 1 (the vision ablation).

## Why the safety claim is a guarantee, not a statistic

The peak-force KPI is bounded by design, not by hoping the policy behaves: the residual is **hard-clamped** (±2 cm / ±10° / ±5 N per step) *before* the controller sees it, and the impedance backbone bounds peak force mechanically. Even a 100%-wrong network output cannot exceed the envelope — see [policy-model.md](policy-model.md) and the safety layering in the deck.

---

### Provenance note

Records the evaluation/validation methodology decided in the 2026-06 design-review preparation, refining [`../../project-scope.md`](../../project-scope.md) (KPIs and eval randomization) and the deck's Evaluation slide. Key decisions captured: the **absolute-success** framing and its dependence on **difficulty calibration for headroom**; the two orthogonal difficulty knobs (**clearance → position accuracy, chamfer → orientation accuracy**); the split between the **scripted paired-seed primary experiment (internal validity)** and the **live blinded human study (external validity)**; and the counterbalancing/blinding mitigations for learning effect and experimenter bias. **2026-06-09 revision**: the heuristic/spiral-search assistance mode was cut, collapsing the comparison from three-way to **two-way (assist off vs on)**; the impedance backbone + Δ-clamp it relied on are retained as the always-on substrate. Project-internal rationale; no `raw/` source backs this page.
