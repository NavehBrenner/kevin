# Self-Evaluation and Reflective Analysis

> **DRAFT — Naveh to review and make his own.**
> The factual spine below is accurate and sourced. Every 🖊️ marker is a place where the
> document needs *your* judgement in first person, not mine — a self-evaluation written by
> someone else is not a self-evaluation. Delete this box before submitting.

Workshop in Autonomous Systems Simulation (OpenU 20973) · solo project · submitted 2026-08-31.

---

## 1. What I set out to build, and what I built

The plan, approved at topic selection, was a shared-autonomy teleoperation system: a human drives
a robot arm toward a peg-in-hole insertion with coarse hand motion, and a learned residual policy
supplies the last-millimetre corrections the human cannot. Nine implementation milestones, two
phases — a force/torque-only residual first (Phase 1), then vision-conditioned (Phase 2).

**All eight implementation milestones were built** (M9 — final evaluation and delivery — is in progress). The simulation, the impedance control backbone, the assistance
seam, the privileged-information expert and data generation, both policy arcs, live stereo-hand
teleoperation from two webcams, and the evaluation harness all exist, are tested, and run from a
clean clone. 61 modules, ~10,500 lines of Python, 239 tests, a green ruff/mypy/pytest gate on
every commit.

**The central hypothesis did not hold.** The residual policy does not lift closed-loop insertion
success above the unassisted human baseline. That is stated as a failure against my own
pre-registered success criteria in the design document (§8), and it is the honest headline of
this project.

🖊️ *One paragraph in your voice: what you actually expected when you started, and how it felt to
end up here. Not a defence — the reader has the numbers. This is the only part of the submission
where you get to say what the experience was like.*

## 2. What worked

**The architecture.** Two `Protocol`s — `InputStrategy` and `AssistProvider` — carry the entire
design in 29 lines. Swapping the operator (scripted ↔ live webcam) or the assistance (none ↔
analytical expert ↔ trained policy) is a one-argument change to a single function. This was not
theoretical: the headline experiment is literally the same code path run twice with a different
argument, which is what makes the paired comparison trustworthy. Nothing in the control loop, the
input layer, or the simulation had to change to add vision to the policy — Phase 2 is a widening
of one input vector.

**Authority bounded by construction rather than by measurement.** Every residual correction is
hard-clamped to ±3 cm / ±10° / ±5 N per step before the controller sees it, and the impedance
backbone can therefore *command* no more than 12.5 N of restoring force. The clamp applies to the
Euclidean *norm* of the position delta, so ‖Δx‖ ≤ 0.025 m and the bound is λ_max·‖Δx‖ =
500 N/m × 0.025 m. A maximally wrong network output changes neither number. This is a bound on the
policy's **authority**, and it holds without reference to any measurement.

It is **not** a bound on the measured contact force, and the results say so: the F/T sensor reads
the contact reaction — impact transients and the full distal load — which reaches 77.86 N, and 58%
of *successful* trials sit above 12.5 N. What holds on the measurement is a weaker, separate
statement: no trial continues past 30 N, because exceeding it is what makes a trial a
`force_abort` ([results](results/mechanisms.md#7-what-still-stands)).
🖊️ *Worth saying whether you designed the authority bound in on purpose from the start, or
realised its value later.*

**Determinism as a research tool.** The scripted operator is open-loop: its command stream
depends only on a seed and a tick, never on what the robot did. Two runs of one seed therefore
differ *only* in the assistance layer. That property is what made a paired-seed comparison
possible at all, and it is why the eventual null result is trustworthy rather than merely
inconclusive.

## 3. What did not work, and why

The negative result is not one failure but a sequence of levers, each tried and each measured
inert. Documented in full in [`docs/results/further-exploration.md`](./results/further-exploration.md).

| Lever | Outcome |
|---|---|
| Plain behavioral cloning (F/T) | mean Δ −4.4 pp over 5 training seeds |
| Vision conditioning | mean Δ −8.3 pp over 3 seeds; never beat F/T-only at any operating point |
| DAgger on-policy relabelling | +2.0 pp (F/T), +1.3 pp (vision) — inside the noise floor |
| Action-rate penalty | worked, on the wrong axis: removed the jerk cost, left success flat |
| A better analytical expert | refuted across five swept knobs |
| Fine-tuning the image encoder | inert; a linear probe showed the frozen features never encoded lateral offset |

**The mechanism, which is the actual contribution.** Per-step imitation learns to reproduce the
expert's *corrections*, and the offline metric for that is anti-correlated with closed-loop
success. A policy can clone the expert's outputs more and more accurately while getting *worse*
at the task, because the failure that matters — the peg jamming against the wall before it is
aligned — is a state the residual cannot recover from at all. It can nudge the next command; it
cannot retract and retry. That diagnosis is what points at contact-recovery control as the lever
worth trying next.

🖊️ *Your call whether to claim this as a real contribution or present it more modestly. I would
claim it — a mechanism is more transferable than a success rate — but it should sound like you.*

## 4. The mistake that cost the most

**Training was unseeded for most of the project, and I did not know.**

`torch.manual_seed` was never called. The `--seed` argument reached only the train/validation
split, so weight initialisation and batch shuffling came from OS entropy — while every run folder
faithfully recorded a seed and a git commit, and therefore *looked* reproducible. Two runs of the
identical command produced different models, and nothing in the artifacts revealed it.

The consequence was a headline result of +33.3 percentage points, reported in July, that could
not be reproduced. Chasing that irreproducibility is what uncovered the root cause. The fix was
one line plus a regression test asserting that two runs of one seed produce byte-identical
weights.

The deeper cost was not the wasted compute. It was that **every single-checkpoint conclusion in
the project up to that point had to be requalified.** Retraining one fixed recipe across seeds
moves closed-loop success by 20–31 percentage points — larger than any effect the project had
been trying to measure. One seed of the F/T recipe reads as a statistically significant
regression (p = 0.0009); another reads as a win. Both are the same recipe.

That is now a standing rule: a single checkpoint is not a measurement, and every recipe is
reported as a distribution over training seeds with its noise floor printed beside it.

🖊️ *Two things only you can write: (a) how you found it — what made you suspect the number rather
than accept it; (b) what you would put in place at the start of a next project so this class of
error is caught in week one rather than month three.*

## 5. What I would do differently

**Measure the noise floor before measuring anything else.** The first experiment of the project
should have been "train this twice and compare", not "train this once and compare it to the
baseline". It costs one extra training run and it establishes the resolution of every subsequent
measurement. Everything that went wrong downstream is a consequence of not knowing that number.

**Pre-register the reporting rule, not just the metric.** The KPIs were defined up front; how
results would be *selected* was not. That gap is what let a single lucky checkpoint become a
headline, and later what let "best DAgger round" look like a legitimate thing to quote. Round-to-
round Δ swings 36 pp within a single training seed with no trend — picking the best is the same
bias in new clothes.

**Separate confounds at design time.** The one secondary claim the project makes beyond the null —
that DAgger tightens the seed distribution — was measured with plain BC at batch size 16 and
DAgger at batch size 2. Two variables, one comparison. The confound was resolved by retraining
plain BC at batch 2 as a control, which showed the tightening is DAgger's and not the batch
size's. It should not have been built that way.

🖊️ *Add anything else — including non-technical: time management, scope, how you'd approach a
solo research project differently.*

## 6. Self-assessment against the grading criteria

🖊️ **This whole section is yours.** Grading yourself is exactly the reflective act being asked
for, and I should not put numbers in your mouth. The rubric and the evidence are below; the
verdicts are not.

| Criterion | Weight | Evidence available | Your assessment |
|---|---|---|---|
| Code Quality | 25% | 61 modules, 239 tests, ruff + mypy + pytest gate on every commit, enforced by CI | 🖊️ |
| SOLID Principles | 15% | two `Protocol` seams; Dependency Inversion exercised for real by the one-argument assist swap; design document §2–§3 | 🖊️ |
| Reliability, Error Handling, Robustness | 15% | three-layer safety envelope; trip-and-lock watchdog; the residual's authority is clamped by construction, so an adversarial policy output cannot raise the commanded force | 🖊️ |
| Project Structure & Design | 15% | dependency DAG with `common/` as leaf; design alternatives compared and justified; per-subsystem rationale documents | 🖊️ |
| Documentation & Developer Experience | 10% | design document, README, architecture tour, policy guide, CLI reference, KPI dashboard; clean-clone setup script; every checkpoint committed and runnable | 🖊️ |
| Meeting KPIs and Performance Goals | 20% | success-rate criteria **not met**, and no contact-force reduction is established either — 11 of 21 checkpoints sit above baseline, a coin flip; the force criterion is met only as an *authority* bound, not as a bound on measured contact; the two effects that are established are costs (slower, rougher). The measurement itself is rigorous and multi-seeded | 🖊️ |

🖊️ *The 20% row is the hard one and the reader will look at it first. My suggestion — take it or
leave it — is to be straightforwardly honest that the performance goal was not met, and to argue
the case for what the work is worth instead, rather than splitting the difference. A candid
self-assessment reads far better than a generous one.*

## 7. What I would tell someone starting this project

🖊️ *Three or four sentences, your voice. This is the closing note and the last thing the reader
sees.*

---

**Supporting documents.** [Design document](./design-document.md) ·
[KPI dashboard](./results/kpi-dashboard.md) ·
[Further exploration](./results/further-exploration.md) ·
[Architecture tour](./guides/architecture-tour.md)
