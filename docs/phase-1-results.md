# Phase-1 Results — F/T Residual vs Human-Only (LAB-38 / M6)

The first publishable result and the close of **Phase 1**: a reproducible, paired-seed
KPI comparison of the F/T-only residual policy against the unassisted human-only baseline,
on the same task under matched operator conditions. This is the artifact the **D1 Design
Review** repackages.

**Headline.** In the calibrated chamfer-contact band, the F/T residual lifts insertion
success from **36.7% → 70.0%** — a **+33.3 pp** paired improvement (McNemar exact
*p* = 0.006; 11 seeds won by the residual vs 1 by human-only). Peak contact force trends
lower (−5.4 N) and is **bounded by construction**. The residual's one cost is
**trajectory smoothness** (jerk rises ~5×) — an honest negative discussed below.

> **Status — preliminary slice.** These numbers come from a residual trained on
> `data/dataset_9` on a **CPU-only** box (early-stopped at epoch 22, best val 0.00140,
> checkpoint git `255714d`) and a **30-seed paired ablation**. They exercise the full
> pipeline end-to-end on real data and stand as an honest preliminary result; the **final
> D1 figure scales the same run to ~100 paired seeds on GPU** — the reporting and ablation
> code are unchanged, only the seed count and the training budget grow. Regenerate with
> the commands in [Reproducing this result](#reproducing-this-result).

## What is compared

Two runtime configurations over the **same trials**, on the always-on impedance backbone
+ Δ-clamp / force-cap substrate — the *only* difference is whether the policy contributes
a correction:

1. **`human_only` (assist off)** — Δ = 0. The baseline.
2. **`residual` (assist on)** — the Phase-1 F/T-only `LearnedResidual` (GRU over
   command + F/T history, MLP over proprioception; **no vision**), clamped to
   ±2 cm / ±10° / ±5 N per step before the controller sees it.

Vision conditioning is **M7**, not compared here. The live-human study (external validity)
is **M8/M9**; this is the **scripted noisy-human, paired-seed** experiment — the primary,
rigorous, reproducible claim.

### Why the pairing gives the result its power

Each seed is run once per config. A seed fixes the procedural wall **and** the scripted
operator's entire command stream (the operator is open-loop — its stream depends only on
its seed, never on the realized state), so between the two runs of a seed **only the
assist layer changes**. The per-seed *delta* therefore isolates the policy's effect with
zero operator variance — the McNemar discordant-pair split for success, the Wilcoxon
signed-rank test for the continuous KPIs.

## The task operating point

The ablation runs at the **deployment (teleop) config** the corpus was generated under
(LAB-96/98/100), so eval samples the same contact dynamics and operator distribution the
policy trained on: `max_dpos`/`joint_damping` at the data-gen defaults, a 9000-step budget
(~18 s @ 500 Hz, LAB-100), and the force cap matched between the controller watchdog and
the observer's FORCE_ABORT threshold (LAB-94 — otherwise the controller freezes the arm at
its lower threshold first and FORCE_ABORT silently never fires).

The one difficulty knob swept is the **operator lateral-error scale** — a multiplier on the
scripted operator's error σ's off the training distribution. A human-only sweep locates the
regime:

| error-scale | 0.2 | 0.3 | **0.4** | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|
| human-only success | 35% | 40% | **35%** | 40% | 35% | 20% |

Human-only sits **sub-ceiling with headroom (~35–40%) across 0.2–0.7**, where contact lands
on the **chamfer** (the residual has a lateral lever), and drops to 20% at scale 1.0 where
contact lands on the **flat wall** (no lever — the Phase-1 identifiability ceiling below).
The headline result is reported at **scale 0.4** (mid-band); scale 1.0 is reported as the
flat-wall ceiling check.

## KPIs

| KPI | Type | Role |
|---|---|---|
| **Insertion success** | bool | **Headline.** |
| Time to insert | s | Supporting (successes only). |
| Peak contact force | N | Safety proxy — **bounded by construction**. |
| Contact events | count | Supporting. |
| Trajectory jerk (∫\|jerk\|) | — | Smoothness. |

**Peak force is a guarantee, not a statistic.** The residual is hard-clamped and the
impedance backbone bounds contact force mechanically, so even a 100%-wrong network output
cannot exceed the envelope — the peak-force column reports a bound the design enforces.

### Read success against the Phase-1 identifiability ceiling

A Phase-1 (no-vision) residual **cannot** improve *success rate* outside the narrow
**chamfer-contact band**: the hole location is not inferable from command/F-T/proprioception
in free space, so the cloned free-space correction is ≈0 by construction, and flat-wall
contact gives the policy no lateral signal (full argument:
`project-wiki/concepts/privileged-learning.md`). So a **structurally-flat success delta in
the flat-wall regime is a result, not a failure**; where success moves, it moves inside the
calibrated band. Vision (M7) is what lifts the success ceiling into the free-space regime.

## Results

### In the chamfer-contact band (error-scale 0.4) — the headline

| KPI | human_only | residual | Δ (paired) | p |
|---|---|---|---|---|
| **Success rate** | 36.7% | **70.0%** | **+33.3 pp** | **0.006** |
| Time to insert (s) | 7.53 | 7.58 | +0.04 s | 0.557 |
| Peak contact force (N) | 29.61 | 24.22 | −5.40 N | 0.092 |
| Contact events | 1.00 | 1.00 | +0.00 | — |
| Trajectory jerk (∫\|jerk\|) | 31.15 | 149.06 | +117.91 | <0.001 |

Paired over 30 matched seeds — discordant success pairs: **11 won by residual, 1 by
human_only** (both 10, neither 8). Records: `results/phase-1/band_scale0.4_trials.csv`.

![Insertion success, assist off vs on](results/phase-1/success_rates.png)

![Per-KPI distributions by config](results/phase-1/kpi_distributions.png)

![Paired per-seed KPI deltas](results/phase-1/paired_deltas.png)

The headline success lift is large and significant. Peak force trends ~5 N lower (and is
capped by construction regardless). Time-to-insert is unchanged. The **residual increases
trajectory jerk ~5×** (31 → 149) — see the caveat below.

### At the training σ's (error-scale 1.0) — the flat-wall ceiling check

| KPI | human_only | residual | Δ (paired) | p |
|---|---|---|---|---|
| **Success rate** | 20.0% | 20.0% | +0.0 pp | 1.000 |
| Time to insert (s) | 9.14 | 8.94 | −0.19 s | 0.625 |
| Peak contact force (N) | 31.73 | 31.67 | −0.07 N | 0.903 |
| Contact events | 1.00 | 1.00 | +0.00 | — |
| Trajectory jerk (∫\|jerk\|) | 40.38 | 155.77 | +115.39 | <0.001 |

Paired over 30 matched seeds (2 discordant success pairs each way → net zero). Records:
`results/phase-1/flatwall_scale1.0_trials.csv`. Exactly as the ceiling predicts: on the
flat wall the residual has **no lateral lever**, so success is structurally flat — it
neither helps nor hurts. This is the control that makes the in-band lift interpretable.

### Honest caveat — the residual costs smoothness

In **both** regimes the residual raises integrated jerk ~5× (highly significant). The
clamped per-step correction is injecting high-frequency motion the human-only baseline
doesn't have — plausibly amplified by this preliminary checkpoint being **CPU-trained and
early-stopped** (22 epochs), so its corrections are noisier than a fully-converged policy's.
This is the smoothness-normalization / action-regularization *known-unknown* flagged in the
M6 spec, and the first thing to revisit for the GPU-scale run: an action-rate penalty in the
BC loss, or reporting jerk normalized by path length. It does **not** touch the safety
guarantee (peak force stays bounded), but it is a real cost the headline should not bury.

## Reproducing this result

Every number is a pure function of the stored per-trial records — no episode is re-run to
report, and re-running the aggregation over the committed CSVs reproduces the tables
bit-for-bit. From `kevin/`:

```bash
# 1. Train the F/T residual on the deployment corpus (CPU or GPU).
uv run python scripts/train_policy.py data/dataset_9 --name lab38_ft_residual

# 2a. Locate the chamfer-contact band (human-only sweep over the operator-error knob).
uv run python scripts/evaluate.py sweep --seeds 20 --error-scale 0.2,0.3,0.4,0.5,0.7

# 2b. Paired ablation in-band: human-only vs residual over matched seeds → trials.csv.
uv run python scripts/evaluate.py pair --seeds 30 --error-scale 0.4 \
    --residual-checkpoint outputs/policy/runs/lab38_ft_residual/checkpoint.pt \
    --out-dir runs/eval-lab38-band

# 3. Aggregate → KPI tables (markdown) + plots + paired statistics.
uv run python scripts/report_results.py --trials runs/eval-lab38-band/trials.csv
```

The committed `results/phase-1/*.csv` are the exact records behind the tables above; the
plots regenerate from them. To reach the final D1 figure, raise `--seeds` to ~100 and train
on GPU (and consider the action-rate penalty for the jerk regression).
