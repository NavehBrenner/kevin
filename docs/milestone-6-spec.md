# Milestone 6 — Evaluation Harness + Phase-1 Results

**Goal**: turn M5's qualitative "it seems to seat the peg" into **measured,
defensible numbers** — a reproducible two-way KPI comparison (human-only vs the
Phase-1 F/T-residual) in which the residual *measurably* beats human-only. This is
the **first publishable result** and the close of Phase 1.

M5 delivered a trained `ResidualPolicy` that runs in real time behind the M3
assistance seam and qualitatively improves seating on held-out episodes. M6 builds
the apparatus that *quantifies* that improvement and defends it: a passive-observer
evaluation harness, an ablation runner over paired seeds, difficulty calibration so
the comparison has headroom, and the KPI tables/plots that become the D1 result.

The methodology is **locked** in [`design/evaluation-protocol.md`](design/evaluation-protocol.md)
— the headline claim (absolute success rate, assist on vs off), the KPI set, the
paired-seed primary experiment, and the controller↔harness decoupling. This spec
is that protocol's build-order expansion.

## Definition of done

By the end of M6 we can:

- **Observe** any runtime episode passively — detect trial start/end, classify
  success/failure, and compute the five KPIs — with **no controller→harness
  dependency** (the harness reads the `Observation` stream only; the controller
  stays mode-less and trial-unaware).
- **Calibrate** the deferred eval-randomization magnitudes (peg/hole clearance,
  chamfer angle, command-noise σ, force cap, timeout) so the **human-only baseline
  sits meaningfully below ceiling** — there is headroom to demonstrate value.
- **Run the ablation**: human-only (`NoAssist`) vs F/T-residual (`ResidualPolicy`)
  over ~100 **paired seeds** — same seed ⇒ identical scripted-operator command
  stream, only the assist layer differs — reproducibly.
- **Report**: KPI tables + plots (success rate, time-to-insert, peak force,
  contacts-before-success, smoothness) with the paired-design summary statistics,
  and a short results writeup. The F/T residual beats human-only on success;
  peak force is bounded by construction.

## What's in M6

- **Passive-observer evaluation harness** (LAB-36, `eval/`) — watches a running
  episode via the existing per-tick hook, owns the *trial* concepts (start/end
  detection, success/failure classification, depth/contact bookkeeping), and emits
  a per-trial KPI record. No upstream dependency from the controller.
- **Ablation orchestration + difficulty calibration** (LAB-37, `eval/` +
  `scripts/`) — the paired-seed runner that executes each seed once per
  configuration, plus the calibration sweep that pins the difficulty knobs against
  the human-only baseline.
- **KPI tables + plots + Phase-1 results writeup** (LAB-38, `eval/` + `scripts/` +
  `docs/`) — aggregate the per-trial records into the publishable comparison: the
  headline success-rate result, per-KPI distributions, paired per-seed deltas, and
  the summary statistics.

These three implementation issues **already exist** in the M6 milestone (LAB-36,
LAB-37, LAB-38); this spec is their detailed expansion, one build step each.

## What's not in M6 — explicit anti-scope

- **Vision.** The harness must be config-agnostic, but the *vision-conditioned*
  policy and the `Phase-2 − Phase-1` vision ablation are **M7**. M6 compares
  exactly two configurations: human-only vs Phase-1 F/T-residual.
- **The live-human study.** The real-operator, blinded/counterbalanced runs
  (external validity) are **M8/M9**. M6's measured result is the **scripted
  noisy-human, paired-seed** experiment (internal validity) — the primary, rigorous,
  reproducible claim that stands on its own.
- **Final cross-config runs + demo video.** The full final KPI sweep across all
  configurations, statistical writeup, and the demo montage are **M9**.
- **Re-training the policy.** M6 evaluates whatever checkpoint M5 produced;
  retraining is M5 calibration, not M6 scope.
- **New assist modes.** The cut heuristic/spiral-search config is not revived (see
  `evaluation-protocol.md`); the impedance backbone + Δ-clamp remain the always-on
  substrate, not a comparison config.

## Design — the four pieces

### The headline claim and why calibration is load-bearing

The project exists to prove **one** comparison: insertion success rate with the
residual engaged vs. off, on the same task. The target is stated as an **absolute**
bar (assisted reaches ≈100%) rather than a relative delta, because a careful
operator might already be near ceiling on an easy task and a tiny relative delta
would be unconvincing. That framing only works if the **unassisted baseline has
headroom** — hence difficulty calibration (LAB-37) is not a tuning afterthought but
a precondition for the result being meaningful.

Two **orthogonal** geometric knobs do most of the work, each tightening a different
error axis (see `evaluation-protocol.md`):

- **Peg/hole clearance → position-accuracy demand.** Smaller gap = the peg tip must
  be positioned more precisely to enter. (The peg is always *smaller* than the hole;
  "tight" means *small clearance*.)
- **Chamfer angle → orientation-accuracy demand.** A shallower/shorter chamfer
  funnel tolerates less angular error, so the peg must be more closely aligned with
  the insertion axis to catch.

Secondary knobs (command-noise σ, force cap, timeout) tune difficulty further
without touching geometry.

### KPIs (per trial) — recap from the protocol

| KPI | Type | Role |
|---|---|---|
| **Insertion success** | bool | **Headline metric.** |
| Time-to-insert | s | Supporting. |
| Peak contact force | N | Safety proxy — **bounded by construction**, not by hope. |
| Contact events before success | count | Supporting. |
| Trajectory smoothness | integrated jerk | Supporting. |

The peak-force KPI is a **guarantee, not a statistic**: the residual is hard-clamped
(±2 cm / ±10° / ±5 N per step) *before* the controller sees it, and the impedance
backbone bounds peak force mechanically — even a 100%-wrong network output cannot
exceed the envelope.

### Harness — passive observer (LAB-36)

The harness is a **passive observer**: it watches the `Observation` stream a running
episode produces and never calls into the controller. The decoupling is a
**Dependency-Inversion pillar** from `project-scope.md` — the controller stays
mode-less and knows nothing about trials, success, or KPIs; *trial concepts live
only in `eval/`*.

The seam already exists. `sim.runner.run_episode` exposes a per-tick
`step_callback(step, observation, base_command, delta, command) -> bool` — the same
hook M4's data-gen logger uses — called with the **pre-step** observation each tick,
and a truthy return ends the episode. The harness plugs in there:

```
run_episode(..., step_callback=trial_observer)   # observer reads Observation, never the controller
```

What the observer reads off each `Observation` (all fields already present):

- `peg_pose`, `hole_poses[target_hole_index]` → insertion depth along the hole axis
  and lateral error → **success/failure classification** and **trial end**.
- `wrist_ft` → **peak |F|** and **contact-event** counting (a contact is a
  force-magnitude excursion above a noise floor).
- `ee_pose` over time → **trajectory smoothness** (integrated jerk via finite
  differences) and time-to-insert (`sim_time`).
- `sim_time` resetting toward 0 → **trial start** (same signal the stateful policy
  uses for its per-episode reset).

The observer emits one **KPI record per trial** (a `@dataclass(frozen=True)` /
on-disk schema — success bool, time-to-insert, peak |F|, contact count, jerk
integral, plus the seed and config label for pairing). Success classification is
the headline decision and gets the most careful acceptance: insertion past a depth
threshold along the hole axis with lateral error within clearance, sustained (not a
transient overshoot).

**Acceptance (LAB-36):** on a known-good seated episode the observer reports
`success=True` with sane KPIs; on a known miss it reports `success=False`; depth and
peak-|F| match a hand-computed value on a fixed short trace; the controller is
untouched (no import from `eval/` into `control/`, asserted structurally); runs
headless without a GPU.

### Ablation orchestration + difficulty calibration (LAB-37)

- **Paired-seed runner.** For each of ~100 fixed per-episode seeds, run the episode
  **once with `NoAssist`** and **once with `ResidualPolicy`**, identical seed ⇒
  identical scripted-operator command stream ⇒ the *only* difference is whether the
  policy contributes a Δ. This zero-operator-variance pairing is what gives the
  result its statistical power; it is the **primary** experiment. The runner drives
  `run_episode` with the harness's observer as `step_callback` and collects the two
  KPI records per seed.
- **Difficulty calibration.** A sweep over the geometric knobs (clearance, chamfer)
  and secondary knobs (noise σ, force cap, timeout) that pins the operating point
  where the **human-only baseline is meaningfully below 100%** — enough headroom
  that the assisted result is a clean statement, not noise. Calibration runs
  human-only only; the chosen magnitudes are recorded (they were left as
  placeholders in the design docs precisely pending this).
- **Reproducibility.** Seeds, knob settings, and checkpoint hash are captured with
  the run so the comparison re-runs bit-for-bit.

**Acceptance (LAB-37):** a calibrated difficulty setting where human-only success is
demonstrably sub-ceiling; a paired run over the seed set produces two KPI records per
seed; a fixed seed reproduces identical command streams across the two configs
(verified by identical `base_command` traces); re-running the orchestration
reproduces the aggregate numbers.

### KPI tables + plots + results writeup (LAB-38)

- **Tables**: success rate, time-to-insert, peak force, contacts-before-success,
  smoothness — human-only vs F/T-residual — with **paired-design summary
  statistics** (per-seed deltas, not just marginal means).
- **Plots**: success-rate bars, per-KPI distributions, paired per-seed deltas.
- **Writeup**: a short Phase-1 results note (in `docs/` or a results notebook
  exported to `docs/`) stating the headline number and the safety-by-construction
  force bound — the artifact D1 repackages.

**Acceptance (LAB-38):** tables + plots regenerate from the stored per-trial records
with one command; the headline success-rate comparison and the paired statistics are
present; the writeup states the result and the bounded-force guarantee.

## Build order (estimated effort in parentheses)

Each step is its own branch → PR → CI → merge, in dependency order.

### Step 1 — Passive-observer evaluation harness · LAB-36 (~4–5 h)

Files: `src/ai_teleop/eval/observer.py` (the trial observer + KPI record),
`src/ai_teleop/eval/schema.py` (the per-trial KPI record shape, behavior-free); tests
in `tests/test_eval_observer.py`.

Build the `step_callback`-compatible observer that owns trial start/end detection,
success/failure classification, and KPI computation off the `Observation` stream.
No controller dependency. Per-trial KPI record out.

### Step 2 — Ablation orchestration + difficulty calibration · LAB-37 (~4–5 h)

Files: `src/ai_teleop/eval/ablation.py` (paired-seed runner),
`scripts/evaluate.py` (the CLI driver, registered as a `kvn`/poe entry), a difficulty
config; tests in `tests/test_ablation.py`.

The paired-seed runner over the two configs, plus the calibration sweep that pins the
difficulty knobs against the human-only baseline. Records seeds + knobs + checkpoint
hash for reproducibility.

### Step 3 — KPI tables + plots + Phase-1 results writeup · LAB-38 (~3–4 h)

Files: `src/ai_teleop/eval/report.py` (aggregation + table/plot generation),
`scripts/report_results.py`, results writeup under `docs/`; tests in
`tests/test_eval_report.py`.

Aggregate the per-trial records into the publishable comparison: headline
success-rate, per-KPI distributions, paired per-seed deltas, summary statistics; the
short writeup.

## Acceptance criteria

- `uv run poe check` green, including the new `eval/` tests.
- The harness classifies a known seated episode as success and a known miss as
  failure, and its depth / peak-|F| / smoothness match hand-computed values on a
  fixed trace — with **no `control/`→`eval/` or `eval/`→`control/` dependency**.
- A calibrated difficulty setting exists at which **human-only success is
  meaningfully below 100%** (headroom for the claim).
- A paired-seed ablation over the seed set runs reproducibly and produces two KPI
  records per seed; identical seeds yield identical scripted-operator command streams
  across configs.
- Tables + plots regenerate from stored records; the **F/T residual measurably beats
  human-only on success rate**, and peak force is bounded by construction.
- M5 (policy + seam), M4 (data-gen), M3 (runner), M2 (harness), M1 (smoke) all still
  pass — M6 adds the `eval/` layer and changes no existing contract.

## Total estimated effort

**10–14 hours**, 3–4 sessions, across three PRs. The long pole is **difficulty
calibration** (LAB-37): it is an empirical sweep, and the genuine risk is that the
Phase-1 F/T-only residual's headroom is narrow (without vision the operator's coarse
command must already place the peg near the hole), forcing a careful operating point
where human-only is sub-ceiling *and* the residual still wins. The harness (LAB-36)
and reporting (LAB-38) are mechanical given the locked protocol.

## Files this milestone touches

```
src/ai_teleop/eval/
├── __init__.py     (populate — re-export observer + ablation + report)   LAB-36/37/38
├── schema.py       (new — per-trial KPI record, behavior-free)           LAB-36
├── observer.py     (new — passive trial observer + KPI computation)      LAB-36
├── ablation.py     (new — paired-seed runner + difficulty calibration)   LAB-37
└── report.py       (new — aggregation, tables, plots)                    LAB-38

scripts/
├── evaluate.py        (new — ablation/calibration CLI driver)            LAB-37
└── report_results.py  (new — regenerate tables + plots)                  LAB-38

docs/
└── phase-1-results.md (new — Phase-1 results writeup)                    LAB-38

tests/
├── test_eval_observer.py  (new)                                         LAB-36
├── test_ablation.py       (new)                                         LAB-37
└── test_eval_report.py    (new)                                         LAB-38
```

`src/ai_teleop/{control,sim,domain,expert,input,policy,data}/` are **not** modified —
M6 consumes the M3 runner's `step_callback` hook and the M5 policy through their
existing contracts. The controller↔harness decoupling means no shared-contract change
is expected (contrast M5's optional `reset()` hook).

## Known unknowns / things to figure out during M6

- **Success-classification thresholds.** Exact insertion depth + sustained-duration
  thresholds that cleanly separate a seated peg from a transient overshoot; calibrate
  against known-good and known-miss traces.
- **Contact-event definition.** The force-magnitude floor and debounce that count
  distinct contacts rather than one sustained press as many.
- **Difficulty operating point.** The clearance/chamfer/noise settings that put
  human-only sub-ceiling while leaving the F/T residual a winnable margin — the
  central empirical unknown, and the milestone's main risk if Phase-1 headroom is
  thin.
- **Seed count.** ~100 paired seeds is the scope target; the final n is whatever
  gives the paired statistics adequate power at the chosen operating point.
- **Smoothness normalization.** Whether integrated jerk is reported raw or normalized
  by path length / time so it compares fairly across trials of different duration.

## Handoff to Milestone 7

- **M7** (Phase 2, vision) **reuses this harness unchanged** — it is config-agnostic
  by construction. M7 adds a third configuration (the vision-conditioned residual) and
  runs it over the *same* paired seeds, so the `Phase-2 − Phase-1` comparison cleanly
  isolates **what vision added** on top of the same evaluation apparatus and the same
  difficulty operating point this milestone calibrated.
- The difficulty knobs M6 pins become the fixed task for M7's ablation, keeping the
  three-way story (off / F/T / vision) on one consistent benchmark.
```
