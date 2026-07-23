# Spec — the checkpoint-divergence investigation (LAB-42 follow-on)

> **Status (2026-07-23, S5): investigation complete.** G1 done (training seeded + test).
> G2: the recipe's spread is **18 pp**, centred ~−4 pp, and does **not** contain the +33 pp
> headline. Phase 4: **H-B unanswerable** (the headline corpus was overwritten in place),
> **H-C null** (device is a rounding-level perturbation). G3 done: `docs/phase-1-results.md`
> now leads with the distribution. All four hypotheses closed — see *Verdict* below. What
> remains is D-4/D-6 mirroring this, and merging PR #86. Results under *Findings*; the
> original plan is unchanged beneath it.

## Findings (S5, 2026-07-22)

### G1 — training is reproducible (commit `07629ed`)

`torch.manual_seed(seed)` in `train_policy` now seeds weight init, batch shuffling and
worker seeds from the same `--seed` that already picked the train/val split; the metadata
key is `seed`, not `split_seed`. `tests/test_train_policy.py::test_train_policy_is_reproducible_at_a_fixed_seed`
trains twice at seed 0 and asserts bit-identical weights, and once at seed 1 to prove the
seed does something. `write_run_artifacts` also records a `checkpoint_sha256`, and the
retention policy is written down in `docs/results/phase-1/checkpoints/README.md`: after G1 a
checkpoint is regenerable from corpus + seed + commit, so only the two pre-G1 checkpoints
behind published numbers are committed.

### G2 — the spread is 18 pp, and 70.0% is outside it (commit `9b9e2d1`)

Five seeds, one recipe (`dataset_10`, hyperparameters byte-identical to `lab101_ft_ar0_ds10`),
each against `human_only` on the same 100 paired eval seeds. Records:
`docs/results/phase-1/lab114/`.

| train seed | best_val_loss | epochs | human_only | residual | Δ pp | b/c | McNemar p | n | residual on the 30 headline seeds |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00144 | 22 | 50.0% | 48.0% | −2.0 | 13/15 | 0.8506 | 100 | 53.3% (n=30) |
| 1 | 0.00170 | 25 | 50.0% | 47.0% | −3.0 | 13/16 | 0.7111 | 100 | 46.7% (n=30) |
| 2 | 0.00182 | 13 | 50.0% | 35.0% | −15.0 | 7/22 | 0.0081 | 100 | 26.7% (n=30) |
| 3 | 0.00197 | 15 | 50.0% | 47.0% | −3.0 | 12/15 | 0.7011 | 100 | 46.7% (n=30) |
| 4 | 0.00117 | 26 | 50.0% | 53.0% | +3.0 | 19/16 | 0.7359 | 100 | 53.3% (n=30) |

- **Paired Δ: mean −4.0 pp, range [−15.0, +3.0], spread 18 pp.** Residual success 35.0–53.0%.
- **`human_only` returned exactly 50.0% in all five runs** (and exactly 36.7% on the 30-seed
  subset, matching the 2026-07-07 record). The harness has now been shown bit-stable across
  three weeks and eight runs; the spread above is training variance alone.
- One seed (2) is a genuine outlier: it early-stopped at 13 epochs and is the only run whose
  regression is significant on its own (p=0.008). A single checkpoint *can* land there —
  which is the whole point.

**Verdict on H-A: half-confirmed, and not sufficient.** The spread is easily wide enough to
swallow every single-checkpoint comparison in M5–M7 — the ≤2-episode M7 margins are noise at
this power. But it does **not** contain the headline: re-scored on the exact 30 seeds that
produced 70.0%, the five checkpoints span **26.7–53.3%**, leaving 70.0% **16.7 pp above the
best of five**. Whatever produced the 2026-07-07 checkpoint, this recipe on this corpus does
not reproduce it. So **H-B (corpus) and H-C (device) are now live, and Phase 4 is required**.

*(The counter-observation that `ar0` and `ar100` both landed on 14/30 was coincidence at the
count level, as the spec warned it might be. n=2 could not tell; n=5 can.)*

**Secondary question, answered:** `best_val_loss` vs closed-loop success across seeds gives
Spearman ρ = **−0.82** (p=0.089, n=5) — lower val loss, higher success. Across *seeds of one
recipe* the offline metric is directionally predictive, the opposite of LAB-106's
anti-predictive result across *interventions*. At n=5 that is a direction, not a measurement,
but it means checkpoint selection by validation loss is not actively harmful within a recipe.
Plot: `docs/results/phase-1/lab114_val_loss_vs_success.png`.

### H-B — unanswerable: `data/dataset_9`'s episode files are `dataset_10`'s (commit `fcd91d5`)

The H-B arm was launched (5 seeds on `dataset_9`) and stopped after three, because its first
three checkpoints came back with **`checkpoint_sha256` identical to the `dataset_10` runs**,
seed for seed, and their evals returned identical success counts. `scripts/dev/lab114_corpus_identity.py`
found the cause:

- All 200 episode files are **content-identical** across the two directories — distinct files,
  no symlinks or hard links, byte-identical arrays.
- `dataset_10`'s manifest matches those arrays on all 200 episodes. **`dataset_9`'s own
  manifest disagrees with them on 35.**

So `data/dataset_9/` holds the 2026-07-22 regeneration under a 2026-07-06 manifest: the corpus
that trained the headline **no longer exists on disk**. H-B cannot be tested — not refuted,
*unanswerable*. **This is H-8 repeating one layer down**: the checkpoint behind the headline
was lost because `outputs/` is gitignored; the corpus behind it was overwritten in place, and
only its committed manifest survived to prove the overwrite happened.

What that surviving manifest still buys us, and it is worth having: regenerating from the same
committed config under 2026-07-22 code **did** change the trajectories — the concrete size of
the G-4 hole (a config hash cannot see code drift):

| | |
|---|---|
| episodes whose `n_steps` changed | 35 of 200 |
| median \|Δ n_steps\| | **1 step** (34 of the 35 are ≤ 100) |
| the one large change | episode 32: 8061 → 3978 steps |
| baseline outcome flips | **1** (episode 114) |
| corpus baseline success | 22.5% → 23.0% |

So corpus drift is **real but tiny** — one flipped outcome in 200. It remains a poor candidate
for a 20+ pp shift in a trained policy, which is the prior the spec assigned it. The honest
statement is that H-B is untestable *and* implausible, not that it was ruled out.

### H-C — null: device is a rounding-level perturbation (commits `490f008`, records `lab114/hc/`)

CPU vs GPU training was the only recorded difference between the original run and every retrain
that could still be varied. It moves nothing that matters.

- **Offline:** the same seed on CPU and GPU trains for the same 22 epochs to the same
  `best_val_loss` to six decimals. `scripts/dev/lab114_weight_distance.py` scales the weight
  difference: **‖Δw‖/‖w‖ = 5.0e-04 for device vs 1.4e+00 for a seed change** — device perturbs
  the weights ~2900× less than the seed does. It is float-accumulation order, not a different
  model.
- **Closed-loop:** the CPU twin of `seed0` scores **49.0%** against the GPU twin's 48.0% (Δ vs
  `human_only`: −1.0 pp vs −2.0 pp), and the two disagree on **exactly 1 of 100 eval seeds**.
  The rounding-level perturbation stays contained through the contact loop; no chaotic
  amplification.

CPU training cost 36.5 min vs 45.5 s on GPU (47×), which is why H-C was run last and at n=1.
One twin is enough: the effect is a rounding difference, and a rounding difference does not
have a wide success distribution hiding behind it.

## Verdict — all three hypotheses closed

| | | |
|---|---|---|
| **H-A** seed variance | half-confirmed, insufficient | 18 pp spread swallows every M5–M7 single-checkpoint margin, but 70.0% is 16.7 pp above the best of five |
| **H-B** corpus drift | **unanswerable** | the 2026-07-06 corpus was overwritten in place; today's drift is 1 flipped outcome in 200 |
| **H-C** device | **null** | CPU vs GPU is ‖Δw‖/‖w‖ = 5e-04, one eval-seed of difference |
| **H-D** wrong arm | refuted (earlier) | the published arm carries the learned policy's jerk signature |

**No recorded combination of seed, corpus and device reproduces the 2026-07-07 checkpoint's
70.0%.** Two of the three artifacts behind it — the checkpoint and the training corpus — no
longer exist (H-8, and H-B here), so its provenance cannot be reconstructed and is stated as
**unknown**, not disputed. What the recipe *does* produce, measured five ways, is a
distribution centred near **−4 pp** (residual ≈46%, human ≈50%) with an 18 pp spread. That is
the honest Phase-1 result, and G3 writes it.

---

## Why this exists

On 2026-07-22 the Phase-1 headline (`36.7% → 70.0%`, +33.3 pp, 30 seeds) **failed to
reproduce**. Two retrained residuals measured no significant lift over 100 paired seeds
(−4.0 pp and −9.0 pp). The environment was exonerated conclusively — `human_only` uses no
checkpoint and returned **36.7% seed-for-seed on all 30 shared seeds in all three runs** — so
the only variable is the checkpoint.

The root cause is identified but **not yet measured**: `grep -rn "manual_seed" src scripts`
returns nothing. `--seed` reaches only the train/val split; weight init and batch shuffling
come from OS entropy. Two runs of the same command produce different models.

That explains *why* checkpoints differ. It does not tell us **how much they differ**, and
that number decides what the project can claim. Every M5–M7 conclusion rests on one
checkpoint per condition.

## Goals, in priority order

1. **G1 — Make training reproducible.** Same command + same corpus ⇒ bit-identical
   checkpoint. Non-negotiable; everything else is unfalsifiable without it.
2. **G2 — Measure the recipe's spread.** How much does closed-loop success vary across
   training seeds, holding corpus and hyperparameters fixed? This is the number that decides
   whether *any* single-checkpoint result in this project is meaningful.
3. **G3 — Decide the honest Phase-1 claim** from G2, and rewrite D-4/D-6 around it.
4. **G4 (only if G2 leaves it open)** — attribute the residual gap between the 2026-07-07
   checkpoint and the 2026-07-22 ones to corpus drift or device.

## Hypotheses, with what would confirm or kill each

**H-A (primary): training-run variance is large enough to contain both results.**
The recipe's success rate at es0.4 has a spread wide enough that 70.0% and 46.7% are both
plausible draws.
· *Confirms:* G2's spread across ≥5 seeds spans ≳20 pp, or its range covers both values.
· *Kills:* the spread is tight (say ≤5 pp) around ~46%, making 70.0% an outlier the recipe
does not produce.
· *Prior:* **high, but do not over-anchor** — see the counter-observation below. It requires
no additional mechanism, since unseeded init is already proven.

> **Counter-observation, and the reason G2 must actually be run rather than assumed.** The two
> 2026-07-22 checkpoints (`ar0` and `ar100`) are *different* policies — different loss config,
> different unseeded init, and they disagree on *which* seeds flip (`ar100`: {4,8,11,13,21,22,27};
> `ar0`: {0,1,8,13,18,19,24↑,26,27}) — yet they land on the **identical aggregate, 14/30 =
> 46.7%**. Two independent draws agreeing exactly is weak evidence that the recipe's spread on
> this corpus is *tight*, with a mean near 46%, which would make 70.0% a poor fit for H-A and
> raise H-B/H-C instead. n=2 cannot distinguish "tight spread" from "coincidence at the count
> level", which is exactly why ≥5 seeds is the minimum and why the *range* matters more than
> the mean. **Enter G2 without a favourite.**

**H-B: corpus drift moved the training distribution.**
`dataset_10` differs from `dataset_9` in 35 of 200 trajectories (H-9).
· *Confirms:* checkpoints trained on the two corpora separate cleanly **after** G2 establishes
the within-corpus spread — i.e. the between-corpus difference exceeds the within-corpus one.
· *Kills:* the two corpora's distributions overlap within the seed spread.
· *Prior:* **low.** 33 of the 35 differ by ≤6 steps and no episode outcome flipped.
· *Caveat:* `dataset_9` on disk is now a 185-original/15-regenerated hybrid, so it can only
support a *weak* test. Say so in the write-up rather than pretending otherwise.

**H-C: CPU vs GPU training changes the result systematically.**
· *Confirms:* CPU-trained checkpoints at fixed seed separate from GPU-trained ones by more
than the seed spread.
· *Kills:* they overlap.
· *Prior:* **low**, but it is the only difference that was present in the original run and
absent in both retrains, so it cannot be dismissed without measurement.
· *Note:* cheap to test but slow to run (the original was CPU-trained for a reason — no GPU
at the time). Do this last.

**H-D: the 70.0% arm was not what it claims to be.**
· **Already refuted this session.** `eval/ablation.py:104` has only `HUMAN_ONLY` as a built-in
config and `residual` is constructed exclusively from `--residual-checkpoint`; the published
arm carries the learned policy's jerk signature (149.06 vs human 31.15, reproduced at 153.57
by the retrain). Recorded so nobody re-opens it. **Do not re-test.**

## Work plan

### Phase 1 — G1, reproducibility (~1 h, no compute)

- Seed torch from the existing `--seed` in `policy/train.py`: `torch.manual_seed(seed)`, a
  seeded `torch.Generator` on the train DataLoader, and `torch.cuda.manual_seed_all(seed)`.
- Rename the metadata key `split_seed` → `seed` — it currently advertises a narrower guarantee
  than any reader assumes (H-10).
- **Acceptance:** a test that trains twice at 1 epoch on a tiny corpus and asserts identical
  weights. Without that test this regresses silently, exactly as it did originally.
- Note in the run folder whether cuDNN determinism was requested; full bitwise GPU determinism
  may need `torch.use_deterministic_algorithms(True)`. If that proves too costly, **say so in
  the metadata** rather than claiming determinism you don't have.

### Phase 2 — G2, the spread (~1 h compute)

- Train **≥5** checkpoints on one corpus (`dataset_10`), identical hyperparameters, seeds
  `0..4`. ~40 s each on GPU.
- Evaluate each with the **same** paired ablation: `--seeds 100 --error-scale 0.4`. ~14 min
  each. Budget ~90 min wall-clock for 5.
- Report: per-seed success rate, the **mean and range** of the paired delta, and each run's
  offline `best_val_loss` beside its closed-loop result.
- **Secondary question, free with this data:** is `best_val_loss` predictive of closed-loop
  success across seeds? LAB-106 found offline metrics *anti*-predictive across
  interventions; across training seeds of one recipe it is unmeasured, and the answer changes
  whether checkpoint selection by validation loss is defensible at all.

### Phase 3 — G3, the claim

Write the Phase-1 result as a **distribution over the recipe**, not a point estimate:
*"F/T residual, mean Δ = X pp over N training seeds × 100 paired eval seeds, range [a, b]."*
Keep the 2026-07-07 record in the ledger as an unreproducible historical point.

If the mean is ≈0 or negative, the honest conclusion is that **Phase 1 shows no closed-loop
success lift**, and the project's positive result becomes the *bounded-force guarantee* plus
the mechanism findings — not a success-rate improvement. Decide that deliberately; do not let
it be decided by which number is quoted first.

### Phase 4 — G4, **now required** (G2's spread did not account for the gap)

Both are one-variable extensions of the Phase-2 harness, read against the same 100 eval
seeds and the same 30-seed subset.

- **H-B (running, S5):** 5 seeds on `dataset_9` — the corpus the headline was trained on —
  GPU, otherwise identical. Confirmed if its distribution separates from `dataset_10`'s
  by more than the 18 pp within-corpus spread. Remember the caveat: `dataset_9` on disk is
  now a 185-original/15-regenerated hybrid, so a null here is weak evidence, not proof.
- **H-C (queued):** CPU training at fixed seeds, the one condition present in the original
  run and absent from every retrain. Slow to run; do it only if H-B comes back null.

If **both** come back null, the 2026-07-07 checkpoint is not reproducible by any recorded
combination of corpus, device and seed — and the honest conclusion is that its provenance is
unknown, not that it was fraudulent. Say exactly that, and let the distribution be the claim.

## G2 also adjudicates the M7 vision negative — read this before scoping it

The M7 conclusion ("the vision residual never beats F/T-only") rests on the **same n=1
per-condition design**, at **20 eval seeds** rather than 100. Its quantitative comparisons are
2–4 episodes wide:

| Comparison | F/T-only | vision | margin |
|---|---|---|---|
| LAB-104, es1.0 | 20% (4/20) | 20% (4/20) | 0 episodes |
| LAB-106 Stage C, es1.0 | 20% (4/20) | 10% (2/20) | 2 episodes |
| LAB-106 Stage C, es0.4 | 40% (8/20) | 40% (8/20) | 0 episodes |

**What survives regardless of G2:** the *mechanism*. LAB-77's identifiability argument
(the operator command already proxies the hole, so vision carries little marginal signal) is
theory plus byte-identical parameter sweeps, and does not depend on any checkpoint. Likewise
the LAB-105 DAgger structural explanation (a bounded expert cannot demonstrate recovery from
the visited force-abort states).

**What G2 puts at risk:** the *directional* claims. "Vision harms out-of-band, ties in-band,
never beats F/T" is a statement about sign, drawn from single checkpoints at margins of 0–2
episodes. If G2 finds a wide spread, those margins are inside the noise floor and the honest
claim weakens from *"vision does not help"* to *"no vision benefit was detectable at this
power"* — a materially different sentence in a report, and the safer one to have checked.

Note the asymmetry that makes this less alarming than the Phase-1 case: **underpowering
cannot manufacture a null.** A weak test failing to find an effect is "not shown", which is
close to what M7 already claims. The Phase-1 failure was the opposite and worse — a *positive*
manufactured by two lucky draws. So M7 needs its wording audited; it does not need re-running.

**Concretely:** once G2's spread is known, re-read `synthesis/imitation-limits-closed-loop`
and `concepts/vision-conditioned-policy` in the wiki and qualify any sign-claim whose margin
is smaller than the measured spread. **No new M7 compute** — this is a wording audit against a
number G2 produces anyway.

## Constraints and guardrails

- **Do not re-run anything that already has a committed record.** The three existing 100-seed
  record sets are in `docs/results/phase-1/`.
- **Commit every checkpoint's metadata**, and decide explicitly whether checkpoints themselves
  get committed or hashed. H-8 happened because `outputs/` is gitignored and a headline
  checkpoint vanished. A hash in the run metadata is the cheap middle path.
- **Never compare an `expert_success_rate` to a `residual` success rate** (H-11): different
  actor, different success rule, different difficulty.
- **Report each row's `n`** — the paired table now does this (H-1); don't hand-write tables
  that drop it.
- One eval-harness caveat that is *already settled*: the harness is bit-stable across this
  whole period (proven by the 30/30 `human_only` agreement). Do not spend time re-validating it.

## Definition of done

- [x] `torch.manual_seed` wired from `--seed`; a same-seed-twice test asserts identical weights.
- [x] ≥5 seeds trained and evaluated at 100 paired seeds; spread reported with mean and range.
- [x] `best_val_loss` vs closed-loop success plotted across those seeds.
- [x] The Phase-1 claim rewritten as a distribution, in `docs/phase-1-results.md` (top box).
      D-4 (in `PROJECT-REVIEW.md`) still to mirror it.
- [x] M7's sign-claims audited against the measured spread (wording only, no new compute).
      Done S5, wiki-side only — `docs/` carries no M7 numbers (1B: `phase-1-results.md` is the
      only doc with a measured outcome). `concepts/vision-conditioned-policy` now reads *"no
      vision benefit was detectable at any operating point tested"*; the margins that survive
      (40 vs 10, 40 vs 15) are separated from the draws (35 vs 40, 40 vs 30) in
      `synthesis/imitation-limits-closed-loop`; the noise floor itself is
      `concepts/training-seed-variance`.
- [ ] LAB-42 / LAB-101 updated with the outcome; this file deleted or folded into D-4.

## Before you start — two things left in a fragile state (2026-07-22)

1. **The two checkpoints behind this session's published numbers are gitignored.**
   `outputs/policy/runs/lab101_ft_{ar0,ar100}_ds10/` are **768 KB each** and `outputs/` is in
   `.gitignore:55`. Their results are now quoted in `docs/phase-1-results.md`. This is
   **H-8 repeating** — the exact failure that made the original headline unarbitrable. Fix it
   in Phase 1 of this investigation, before training anything new: either commit small
   checkpoints, or record a SHA-256 of each in its `metadata.json`. Decide the policy once.
2. **The branch is unpushed** (12 commits on `feat/lab-42-project-state-review`) and has no PR.
   All of this session's findings exist on one machine.
