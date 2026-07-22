# Spec — the checkpoint-divergence investigation (LAB-42 follow-on)

> **Status:** not started. Written 2026-07-22, at the end of the session that found the
> problem. Self-contained: a session picking this up needs no other context than this file
> and `docs/review/code-audit.md` §H.

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
· *Prior:* **high.** It requires no additional mechanism — unseeded init is already proven.

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

### Phase 4 — G4, only if needed

Run H-B / H-C only if G2's spread fails to account for the gap. Both are one-variable
extensions of the Phase-2 harness.

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

- [ ] `torch.manual_seed` wired from `--seed`; a same-seed-twice test asserts identical weights.
- [ ] ≥5 seeds trained and evaluated at 100 paired seeds; spread reported with mean and range.
- [ ] `best_val_loss` vs closed-loop success plotted across those seeds.
- [ ] The Phase-1 claim rewritten as a distribution, in `docs/phase-1-results.md` and D-4.
- [ ] LAB-42 / LAB-101 updated with the outcome; this file deleted or folded into D-4.
