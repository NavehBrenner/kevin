# Near-miss — how close did the peg get, when it did not seat?

Every other KPI on this board is either conditioned on seating (`time_to_insert_s`) or
indifferent to it (`peak_contact_force`, `jerk_integral`). None of them can tell a policy that
*almost* inserts from one that flails: both score 0 on the headline. So a live possibility
stayed open — **that the residual was better than the success rate gave it credit for**, moving
the peg substantially closer without crossing the seating threshold.

This page closes that possibility. It is measured, not argued.

Reproduce with, from `kevin/`:

```
uv run python scripts/dev/official_kpi/backfill_near_miss.py --glob 'eval_official_*' --write
uv run python scripts/dev/official_kpi/near_miss.py
```

---

## The metric, and why it is this one

**`min_tip_hole_distance`** — the minimum over the trial of ‖hole centre − peg tip‖.

Two choices in that sentence are load-bearing, and both were fixed *before* the sweep ran:

- **Minimum over the trial, not the final value.** The tip travels *through* the hole origin
  and continues past it, so a perfect insertion's *final* distance is non-zero and rising.
  Final distance is non-monotone in goodness; the running minimum is not — a seated peg bottoms
  out near 0, and a failed trial records how near it ever got.
- **A raw geometric distance, not a relaxed success rule.** The tempting construction is
  "distance to the success set" — clamp each violated term of the seating predicate and combine.
  It saturates at exactly 0 for every trial that ever satisfied the predicate (~50 % of them
  here), which in a paired signed-rank test means half the sample becomes ties and contributes
  nothing — in a metric adopted *specifically* to buy statistical power. A raw distance never
  ties, and is independent of the seating thresholds rather than a restatement of them.

Recorded alongside, off the **same** argmin step so all three describe one instant:
`penetration_at_closest` and `lateral_error_at_closest` (the axial/lateral split). Reporting
`max(penetration)` beside `min(distance)` would describe two different moments of the
trajectory.

### Where the numbers come from

No episode was re-run. Every official eval already logged per-tick realized state under
`runs/<run>/traces/`, so the metric was recovered for the whole back-catalogue by replaying
those traces through the same `TrialObserver` that scored the runs live — 4,200 trials across
21 official runs, ~6 minutes.

That replay carries a free correctness gate, and it passed: **21/21 runs reproduced every
already-stored KPI column** (`outcome`, `peak_contact_force`, `jerk_integral`, `n_steps`)
exactly. The offline path is provably the same calculator, so the new column is not a
reinterpretation of the old runs — it is the measurement those runs would have emitted had the
KPI existed at the time.

One ordering bug was fixed to make the measurement honest: the observer derived seating
geometry *after* the force-abort early return, so the aborting step — the moment of impact —
was never scored. On a force abort that is the single most informative pose there is.

### Pre-registered decision rule

Fixed before running, because searching across candidate near-miss formulations until one
shows a favourable Δ would rebuild [the LAB-114 trap](noise-floor.md) with more knobs:

1. Headline = paired Δ, **unconditional** over all matched trials, aggregated across each
   recipe's training seeds as mean + observed range.
2. A Δ counts **only if it exceeds its own recipe family's training-seed spread**. Sign and
   p-value establish nothing on their own.
3. Vision is the hypothesis, F/T is the control — an F/T-only residual has no information about
   hole location until contact, so it has no mechanism to improve free-space approach.
4. Outcome strata are reported as results, not filtered as nuisances.
5. Failure-conditional comparisons are diagnostics only.

§4 (outcome-stable pairs) was **added after seeing §1–§3** and is not part of the
pre-registration. It is reported because it is strictly more conservative than the registered
headline and moves the conclusion *against* the assist — it removes the one recipe that cleared.
A post-hoc analysis that tightens a null is a different animal from one that rescues a result,
but the ordering is on the record either way.

**Disclosed peek:** during design, a draft variant of this metric was run on
`eval_official_ft_s0` — one F/T run, the control arm. Nothing else was inspected before the
rule above was fixed.

---

## 1. The headline: nothing moves

![closest approach to the hole per recipe, box charts over training seeds against the human baseline](phase-1/near_miss_spread.png)

Read the picture first: every recipe's box either straddles the `human_only` line or sits
barely below it, on an axis that spans 12–18 mm. The two plain-BC recipes are the *widest*
boxes on the chart — a spread produced by nothing but the training seed. The two DAgger boxes
are the tight ones, and both sit under the line.

Paired Δ against `human_only`, in millimetres. Negative = the assist got the peg closer.
The human baseline is a closest approach of **14.73 mm**.

| recipe | seeds | human | assist | paired Δ (range) | seed floor | verdict |
|---|---:|---:|---:|---:|---:|---|
| FT plain | 5 | 14.73 | 15.36 | **+0.62** [−0.26, +2.95] | 3.21 | within floor |
| FT plain (batch 2) | 5 | 14.73 | 14.94 | **+0.21** [−1.15, +3.15] | 4.31 | within floor |
| FT DAgger | 5 | 14.73 | 14.04 | **−0.69** [−0.99, −0.35] | 0.64 | clears |
| Vision plain | 3 | 14.73 | 15.50 | **+0.77** [−0.82, +2.32] | 3.14 | within floor |
| Vision DAgger | 3 | 14.73 | 13.95 | **−0.78** [−1.98, −0.16] | 1.82 | within floor |

**No recipe moves closest approach by even 0.8 mm on a 14.7 mm baseline** — under 5 %. Four of
five sit inside the spread their own recipe produces from nothing but a different training
seed.

The one that clears — FT DAgger, −0.69 mm against a 0.64 mm floor — clears by a hair, and
clears partly *because its floor is unusually narrow*. That is consistent with the independent
LAB-114 finding that DAgger **tightens the seed distribution without moving its centre**: a
narrow floor is easier to clear, so "clears" here is a much weaker claim than the same word
would carry for a recipe with a 3 mm floor. **[§4](#4-only-the-walls-the-assist-never-flipped)
removes even this one** — on walls the assist never flipped, FT DAgger falls to −0.40 against a
0.72 mm floor and stops clearing.

What is worth noting is a sign pattern the floor test does not capture. Per training seed
([`official_kpi_tables.md`](phase-1/official_kpi_tables.md)):

| recipe | per-seed Δ (mm) |
|---|---|
| FT plain | −0.26, −0.17, −0.14, **+0.75, +2.95** |
| FT plain (batch 2) | −1.15, −0.80, −0.45, **+0.30, +3.15** |
| FT DAgger | −0.99, −0.88, −0.70, −0.54, −0.35 |
| Vision plain | −0.82, **+0.80, +2.32** |
| Vision DAgger | −1.98, −0.20, −0.16 |

**Both DAgger recipes are negative on every single training seed** (5/5 and 3/3); both
plain-BC recipes straddle zero. Compared *directly* against their plain counterparts rather
than against the human, DAgger is −1.31 mm [−3.30, −0.56] (F/T) and −1.54 mm [−2.78, +0.63]
(vision). Small, consistent, and pointing the same way as DAgger's other measured effects. It
is suggestive, not established.

**The baseline is deterministic, which is the provenance check.** The checkpoint-free
`human_only` arm returns **exactly 14.73 mm in every one of the 21 runs** — same eval walls,
same operator stream, no policy. Every millimetre of spread in the table above therefore comes
from the training draw and nothing else. This is the same signature LAB-114 used to exonerate
the environment, reproduced independently on a metric that did not exist at the time.

## 2. The trap: by outcome, it looks like a large win

Closest approach split by how the trial ended — mean mm (n trials, median steps):

| recipe | outcome | human_only | assist |
|---|---|---|---|
| FT plain | success | 6.84 (n=250, 3664) | 6.79 (n=228, 3666) |
| FT plain | timeout | 68.55 (n=45, 9000) | 65.73 (n=46, 9000) |
| FT plain | force_abort | 12.54 (n=205, 1619) | 13.74 (n=226, 1642) |
| FT plain (batch 2) | success | 6.84 (n=250, 3664) | 6.56 (n=233, 3354) |
| FT plain (batch 2) | timeout | 68.55 (n=45, 9000) | **46.75** (n=72, 9000) |
| FT plain (batch 2) | force_abort | 12.54 (n=205, 1619) | 13.22 (n=195, 1636) |
| FT DAgger | success | 6.84 (n=250, 3664) | 6.28 (n=260, 3360) |
| FT DAgger | timeout | 68.55 (n=45, 9000) | 54.22 (n=58, 9000) |
| FT DAgger | force_abort | 12.54 (n=205, 1619) | 12.33 (n=182, 1616) |
| Vision plain | success | 6.84 (n=150, 3664) | 6.61 (n=125, 3527) |
| Vision plain | timeout | 68.55 (n=27, 9000) | **49.32** (n=42, 9000) |
| Vision plain | force_abort | 12.54 (n=123, 1619) | 13.17 (n=133, 1664) |
| Vision DAgger | success | 6.84 (n=150, 3664) | 6.12 (n=154, 3528) |
| Vision DAgger | timeout | 68.55 (n=27, 9000) | 58.90 (n=31, 9000) |
| Vision DAgger | force_abort | 12.54 (n=123, 1619) | 12.33 (n=115, 1600) |

Read alone, this table says the assist is dramatically better: on timeouts it gets the peg
**22 mm closer** (68.55 → 46.75), a third of the distance gone. Every stratum improves or holds
for most recipes.

**Almost all of that is arithmetic, not accuracy.** The assist does not only change *where*
trials end up, it changes *which stratum* they land in — batch-2 timeouts go 45 → 72 while
successes go 250 → 233 and aborts 205 → 195. The 27 trials newly classified as timeouts arrive
from the success stratum (~6.8 mm) and the abort stratum (~12.5 mm), both far nearer the hole
than the 68.55 mm the timeout stratum averaged. Dropping them into that bucket pulls its mean
down with no change in accuracy whatsoever.

So every stratum's mean can fall while the overall mean **rises** — Simpson's paradox, live in
this table. That is exactly why the pre-registered headline is unconditional.

## 3. Separating accuracy from outcome-mix

Direct standardization: re-weight the assist's per-stratum means by the **human's** outcome
mix, answering "how close would the assist have got, had it produced the same mix of
successes/timeouts/aborts?"

| recipe | human (mm) | assist raw | assist standardized |
|---|---:|---:|---:|
| FT plain | 14.73 | 15.36 | 14.95 |
| FT plain (batch 2) | 14.73 | 14.94 | **12.90** |
| FT DAgger | 14.73 | 14.04 | **13.07** |
| Vision plain | 14.73 | 15.50 | **13.14** |
| Vision DAgger | 14.73 | 13.95 | **13.42** |

Held at a fixed outcome mix, four of five recipes look **1.3–1.8 mm closer** than the human,
which reads as real aim that the raw comparison cancels out.

**It mostly is not there** — §4 takes the mix out properly and finds at most 0.7 mm. The reason
this table overstates is worth keeping: standardization reweights the *stratum means*, but those
means are themselves contaminated. The assist's timeout trials and the human's timeout trials
are **different walls**, so re-weighting them by a common mix still compares two different
populations. Fixing the weights does not fix the membership.

Kept here because it is the natural next thing to try after §2 and it is wrong in an
instructive way.

## 4. Only the walls the assist never flipped

The clean way to remove the migration is not to reweight it but to **drop it**: keep only the
walls where *both* arms ended the same way, and compare those.

| recipe | stable pairs | → gained | → lost | stable Δ (range) | floor | verdict |
|---|---:|---:|---:|---:|---:|---|
| FT plain | 362/500 | 55 | 77 | +0.63 [−0.12, +2.36] | 2.48 | within floor |
| FT plain (batch 2) | 334/500 | 70 | 87 | +0.11 [−0.95, +2.79] | 3.74 | within floor |
| FT DAgger | 356/500 | 71 | 61 | −0.40 [−0.82, −0.11] | 0.72 | within floor |
| Vision plain | 202/300 | 32 | 57 | −0.03 [−0.85, +1.38] | 2.23 | within floor |
| Vision DAgger | 213/300 | 44 | 40 | −0.72 [−1.53, −0.23] | 1.30 | within floor |

**Nothing clears its noise floor — including FT DAgger**, the single recipe that cleared in §1.
Its Δ falls from −0.69 to −0.40 against a 0.72 mm floor, so the one apparent positive in the
headline was itself part migration. And Vision plain's standardized "−1.6 mm of aim" from §3
collapses to **−0.03 mm** here: essentially all of it was composition.

This is the strictest of the four views and the one to quote.

### Why this is a legitimate restriction and §2 was not

Selecting on *one* arm's outcome is a collider — the failures that survive under the assist are
the harder ones, so the comparison flips sign for a policy that improved. Selecting on the
**joint** outcome of both arms is a different object: it defines a *principal stratum*, the set
of walls whose result the assist never flipped. A stratum defined by both potential outcomes is
a fixed property of the wall, not something the treatment moved, so comparing inside it is a
valid causal contrast.

Principal stratification is normally close to unusable, because you never observe both potential
outcomes for the same unit. **Here you do.** The simulation is deterministic and the design is
paired on the eval seed, so both arms of every wall are observed and stratum membership is read
straight off the data instead of modelled. That is a luxury a real randomized trial does not get,
and it is worth spending.

### What it costs

The stratum excludes exactly the walls where the assist did the most — every conversion, both
directions, 27–33 % of trials. So this answers *"on the walls it did not flip, did it aim
better?"*, not *"is it a better policy"*. The migration counts are printed beside it so the
excluded set is never invisible, and they carry their own finding: **plain BC loses more walls
than it gains** (55/77 and 70/87 F/T, 32/57 vision), while **DAgger is roughly balanced** (71/61
and 44/40) — the reliability effect, visible again on a metric that did not exist when it was
first measured.

## Verdict

**The near-miss hypothesis is not supported.** The residual does not get the peg meaningfully
closer to the hole. On the pre-registered unconditional measure no recipe moves the closest
approach by more than 0.8 mm on a 14.7 mm baseline, and four of five are inside their own
training-seed noise. Restricted to the walls the assist never flipped — the strictest and
cleanest view — **no recipe clears its floor at all**, at most 0.72 mm of movement.

The four views agree, and they get tighter in the right direction: raw ≈ 0; by-outcome looks
like a large win and is Simpson's paradox; standardized looks like ~1.5 mm and is still
comparing different walls; outcome-stable is ~0 to −0.7 and clears nothing.

This is a *stronger* negative than the success-rate null on its own, and it is the reason the
metric was worth adding. A binary KPI cannot distinguish "the policy nearly works" from "the
policy does not work"; both read 0 %. A continuous one can, and it says the second. The
"it is better than we give it credit for" explanation for the flat success rate is now closed
off by measurement rather than left open by the absence of one.

Two things survive as real, both small:

- **DAgger is consistently negative across every training seed** (5/5 F/T, 3/3 vision), where
  plain BC straddles zero, and −1.3 to −1.5 mm measured head-to-head against its plain
  counterpart. It survives the outcome-stable restriction in sign (−0.40, −0.72) without
  clearing its floor. Consistent with DAgger tightening the seed distribution
  ([noise floor](noise-floor.md)) rather than moving the mean.
- **Plain BC loses more walls than it gains and DAgger does not** — 55/77 and 70/87 (F/T),
  32/57 (vision) against DAgger's balanced 71/61 and 44/40. The reliability effect, recovered
  independently on a metric that did not exist when it was first measured.

Neither changes the [mechanisms](mechanisms.md) picture: the expert ceiling, the identifiability
limit, and the force-authority bound all stand unchanged. This adds one closed door.
