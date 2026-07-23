# Policy guide — train, deploy, evaluate

The operator's manual for the policy subsystem: how to **train** a residual policy, **deploy**
one in an episode, and **evaluate** one with a paired ablation — three runnable recipes — plus a
**checkpoint inventory** of what every run in `outputs/policy/runs/` is and which are superseded.

For *what the experiments found*, see the
[KPI dashboard](results/kpi-dashboard.md); for *where the code lives*, the
[architecture tour](architecture-tour.md). This guide is the *how-to-run*.

> **The policy is a residual.** A base command source (a scripted noisy operator, or a live
> human in M8/M9) drives the arm; the trained network adds a clamped micro-correction
> (±2 cm / ±10° / ±5 N per step) on top of the always-on impedance backbone. "Train a policy"
> means clone the analytical expert's corrections by behavioral cloning; "deploy" means load
> that checkpoint as the correction layer in a live episode.

> **Read every success rate at the noise floor (LAB-114).** A single checkpoint is one draw
> from an **18 pp**-wide distribution over training seeds. These recipes produce and run
> *individual* checkpoints; a rate from one of them is not a result about the recipe. See the
> [KPI dashboard](results/kpi-dashboard.md) top box and
> [§4](#4-recipe-c--evaluate-a-policy-paired-ablation).

---

## 0. Prerequisites — a corpus

Training reads a behavioral-cloning corpus (an M4 dataset dir holding `metadata.json` + episode
`.npz` files). Generate one, or point at an existing `data/dataset_*`:

```bash
# N episodes of the scripted operator + analytical expert → data/my_corpus/
uv run kvn gen --episodes 200 --out-dir data/my_corpus
# add --record all (wrist frames) if you will train a vision policy — see Recipe A.
```

`kvn gen --help` lists the operating-point knobs (error scale, step budget, hole geometry). The
committed corpora and their lineage are in the [dashboard §2.1](results/kpi-dashboard.md#21-corpus-lineage-datadataset_metadatajson).

---

## 1. Recipe A — train a policy

```bash
# F/T-only residual (the Phase-1 headline recipe), seeded and reproducible.
uv run kvn train data/my_corpus --name my_ft_residual --seed 0
```

That writes a self-documenting run folder `outputs/policy/runs/my_ft_residual/`:

| File | What it is |
|---|---|
| `metadata.json` | corpus (dir + fingerprint), full model/loss/train config, `seed`, `git_commit`, `device`, `best_val_loss`, `best_epoch`, and a **`checkpoint_sha256`** |
| `history.json` | per-epoch train/val loss |
| `history.png` | the loss curve |
| `checkpoint.pt` | the deployable weights + serialized `PolicyConfig` |

**Reproducibility (LAB-114).** `--seed` now seeds weight init, batch shuffling *and* the
train/val split, so **corpus + `--seed` + git commit fully pin the checkpoint** — verify by
re-training and comparing `checkpoint_sha256` ([§5](#5-regenerate-and-verify-a-checkpoint)).
Always train on GPU (the `cuda` default); `--device cpu` is ~47× slower and only used to prove
device-invariance.

**The flags that matter** (everything else is a default; `--help` is exhaustive):

| Flag | Effect | When |
|---|---|---|
| `--action-rate-weight 100` | smoothness penalty — cuts the ~5× jerk regression (LAB-104) | any deployable F/T policy; no success cost (dashboard §6) |
| `--vision` | add the image-CNN stream (needs a corpus recorded with `--record all`) | Phase-2 vision policy |
| `--freeze-image-encoder` | pretrained backbone as a fixed extractor (train only the projection) | vision on a small GPU / fast |
| `--checkpoint-image-encoder` + `--image-encode-chunk 32` `--amp` | VRAM cuts that let an unfrozen backbone fine-tune on 8 GB (Stage C) | vision fine-tune |
| `--num-workers 4` | parallel wrist-frame decode | **always with `--vision`** |
| `--weight-position 10 --weight-decay 1e-4` | force the lateral-correction fit (LAB-106) | offline-metric experiments — but see the anti-correlation warning below |

> **Do not tune a policy by offline validation loss across recipes.** On this task per-step BC
> fidelity is *anti*-correlated with closed-loop success — the `--command-ee-delta` /
> `--weight-position 10` fixes drove offline error to a record low and collapsed closed-loop to
> 10% (dashboard §6). Only a closed-loop ablation (Recipe C) is a valid signal. Within a single
> recipe's seeds, val loss *is* directionally predictive (ρ = −0.82), so best-val checkpoint
> selection *of one recipe* is fine.

---

## 2. Recipe B — deploy a policy (one episode)

```bash
# Run one episode with the trained residual as the correction layer.
uv run kvn episode --policy tf \
    --checkpoint outputs/policy/runs/my_ft_residual/checkpoint.pt \
    --seed 7 --headless --max-steps 9000
```

`--policy` selects the correction layer: `noassist` (raw operator, the baseline), `expert`
(analytical residual), `tf` (trained residual — needs `--checkpoint`). **A vision checkpoint
loads through the same `--policy tf` path** — the serialized `PolicyConfig.use_vision` selects
the modality and the wrist camera is enabled automatically; there is no separate `--policy
vision`. Drop `--headless` for the viewer. This is the single-episode smoke path; for a *measured*
comparison use Recipe C.

---

## 3. Recipe C — evaluate a policy (paired ablation)

The rigorous claim is a **paired-seed** comparison: each seed fixes the wall *and* the operator's
whole command stream, so between the two runs of a seed only the assist layer changes (dashboard
§5 explains the power). Two steps — locate the difficulty band, then run the ablation:

```bash
# 3a. Human-only difficulty sweep → find the chamfer-contact band (where the residual has a lever).
uv run kvn evaluate sweep --seeds 20 --error-scale 0.2,0.3,0.4,0.5,0.7

# 3b. Paired ablation in-band → per-trial CSV.
uv run kvn evaluate pair --seeds 100 --error-scale 0.4 \
    --residual-checkpoint outputs/policy/runs/my_ft_residual/checkpoint.pt \
    --out-dir runs/eval-my-ft

# 3c. Aggregate the raw per-trial rows → KPI tables (markdown) + plots + paired stats.
uv run python scripts/report_results.py --trials runs/eval-my-ft/trials.csv
```

**The M7 three-way** (human / F/T / vision) uses the dedicated checkpoint flags and a multi-way
report:

```bash
uv run kvn evaluate pair --seeds 20 --error-scale 0.4 \
    --ftonly-checkpoint outputs/policy/runs/ftonly_ar100/checkpoint.pt \
    --vision-checkpoint outputs/policy/runs/vision_stageC/checkpoint.pt \
    --out-dir runs/eval-3way
uv run python scripts/report_results.py --trials runs/eval-3way/trials.csv \
    --baseline human_only --treatment vision \
    --also-compare human_only:ftonly --also-compare ftonly:vision
```

**Operating-point knobs** — the numbers are only comparable within one setting (dashboard §2):

| Knob | Meaning |
|---|---|
| `--error-scale` | operator lateral-error difficulty; `1.0` = training σ's (flat wall, no lever), `0.4` = chamfer-contact band |
| `--seeds` | paired seed count. **≥100 for a claim** — a 20-seed arm carries a ±20 pp exact interval |
| `--max-steps` | per-episode budget (default matches the data-gen corpus; a mismatch was the LAB-107 bug) |
| `--device cpu` | force CPU inference (vision needs GPU for real-time) |

> **One checkpoint is one draw.** A single ablation gives that checkpoint's rate ± the eval
> interval; the *recipe's* honest number is a distribution over ≥5 training seeds (dashboard §5).
> Report a mean and range, not a point.

---

## 4. The checkpoint inventory

Every run in `outputs/policy/runs/`, by lineage. **`outputs/` is gitignored** — these live on
the training box; a run is recoverable only if it is either committed (below) or regenerable
(post-LAB-114: corpus + `seed` + `commit`, all in its `metadata.json`). Results columns are in
the [dashboard §3–§4](results/kpi-dashboard.md#3-training-runs-m5m7); this table is the
*operational* view.

**M5 — first behavioral clone**

| Run | Corpus · seed · commit | What it is | Status |
|---|---|---|---|
| `lab34_baseline` | `dataset_1` · 0 · `7f72e4c` | first BC, schema-1.0 task geometry | **superseded** — old corpus, not comparable |

**M6 / M7 — F/T-only residual** (all `dataset_vision`, seed 0)

| Run | Commit | Config | Status |
|---|---|---|---|
| `ftonly_baseline_lab82` | `dcea204` | no action-rate penalty | baseline; superseded by `ftonly_ar100` for deploy |
| `ftonly_ar30` | `9023527` | action-rate ×30 | jerk sweep point |
| `ftonly_ar100` | `9023527` | action-rate ×100 | **the deployable F/T policy** (smooth, no success cost) |
| `ftonly_wpos10_wd` | `8d533ce` | pos-loss ×10 + weight-decay | LAB-106 offline-fix (1/2) |
| `ftonly_gate_wpos10_wd` | `8d533ce` | ↑ + `command_ee_delta` feature | **negative artifact** — collapsed closed-loop to 10% (dashboard §6) |

**M7 — vision** (all `dataset_vision`, seed 0)

| Run | Commit | Config | Status |
|---|---|---|---|
| `probe_b2` | `dcea204` | 10-episode batch-2 fits-on-8GB smoke | probe only |
| `vision_frozen_lab82` | `dcea204` | frozen encoder, no action-rate | best offline val (0.00107), closed-loop non-improver — the offline-val trap |
| `vision_frozen_ar100` | `9023527` | frozen encoder + action-rate ×100 | frozen-vision candidate |
| `vision_stageC` | `365f770` | **unfrozen** encoder (Stage C) | the vision ablation arm — ties F/T (NULL, dashboard §6) |

**M7 — DAgger** (`dagger_ft_agg`, grows per round, seed 0, commit `8d533ce`)

| Run | Aggregated corpus | Status |
|---|---|---|
| `dagger_round0` / `1` / `2` | 340 / 380 / 420 ep | **negative** — 40% → 30% → 15%; the bounded expert can't demonstrate recovery (dashboard §6) |

**Phase-1 reproduction + the seed-variance study** (`dataset_10` unless noted, LAB-101/114)

| Run(s) | Seed · commit | What it is | Status |
|---|---|---|---|
| `lab101_ft_ar0_ds10` | 0 · `4137060` | headline recipe, GPU repro (ar0) → −4 pp | **committed** under `docs/results/phase-1/checkpoints/` |
| `lab101_ft_ar100_ds10` | 0 · `e899914` | ↑ + action-rate ×100 → −9 pp | **committed** (same reason) |
| `lab114_seed{0..4}` | 0–4 · `07629ed` | the recipe's spread (the 18 pp floor) | regenerable (post-G1 sha) |
| `lab114_ds9_seed{0..3}` | 0–3 · `9b9e2d1`+ | H-B corpus arm — **byte-identical to `lab114_seed{0..3}`** | proves `dataset_9`==`dataset_10` on disk (dashboard §2.1) |
| `lab114_cpu_seed0` | 0 · `0fd28d0` | H-C device arm (CPU) | proves device is a rounding-level perturbation |

The two `lab101_*` checkpoints are committed because they back published numbers and are
**pre-G1 (unseeded) — no command reproduces them**; retention policy in
[`docs/results/phase-1/checkpoints/README.md`](results/phase-1/checkpoints/README.md). The
2026-07-07 headline checkpoint is **gone** (gitignored, never committed — finding H-8), which is
why its 70.0% can no longer be arbitrated.

---

## 5. Regenerate and verify a checkpoint

Post-LAB-114, a run is a pure function of its recorded inputs:

```bash
# Re-train from the same corpus + seed + commit, then compare the hash to the original metadata.
git checkout 07629ed  # the run's git_commit
uv run kvn train data/dataset_10 --name repro_check --seed 0
# → outputs/policy/runs/repro_check/metadata.json:checkpoint_sha256 must equal the original's.
```

A same-seed-twice test (`tests/test_train_policy.py::test_train_policy_is_reproducible_at_a_fixed_seed`)
keeps this from regressing. **Pre-G1 runs (everything before `07629ed`) have no
`checkpoint_sha256` and cannot be reproduced** — their weight init came from OS entropy; that is
the whole reason the two published `lab101_*` checkpoints are committed rather than regenerated.

---

## Note — commands not yet in the `kvn` front door

`scripts/dagger.py` (on-policy DAgger relabel) and `scripts/report_results.py` (KPI aggregation)
are run as raw `python scripts/…` above — they are **not** in `APP_COMMANDS`, so `kvn dagger` /
`kvn report` do not exist yet. Exposing them (and refreshing the stale `docs/cli.md`, which lists
neither `train` nor `evaluate`) is a **stage-3C** stale-doc fix, tracked there — not changed here.
