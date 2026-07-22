# Code audit — LAB-42 project state review

**Status: round 1 findings (2026-07-21), most verdicts applied.** The audit itself was
produced read-only; the fixes landed afterwards under LAB-110 (see *Outcomes* at the end).

Scope: `kevin/` only — `src/ai_teleop/` (9,689 LOC), `scripts/` (2,749), `scripts/dev/`
(6,952), `tests/` (5,021). `stereohand/` is out of scope.

Each finding has an ID, `file:line` evidence, and a **verdict**: `FIX` / `DELETE` /
`KEEP — because`. Phase 2 executes only the verdicts Naveh approves.

---

## Gate baseline (recorded before any change)

Run on `feat/lab-42-project-state-review` @ `178ce81`, 2026-07-21:

| Check | Result |
|---|---|
| `ruff check --fix src tests scripts` | **clean** |
| `python -m mypy` | **clean** — 61 source files |
| `python -m pytest -q` | **226 passed**, 23.07 s |
| Whole `poe check` | 39.5 s wall |

Nothing was failing before this review started. Any test that fails after Phase 2 is a
regression Phase 2 caused.

---

## Headline

**This is a well-engineered codebase, and the audit should not pretend otherwise.** The
things that usually rot in a solo research repo are in good shape here: the dependency DAG
is real and enforced (`domain/` imports nothing; `common/` imports nothing from `sim/`),
every non-obvious constant carries the experiment that set it, the deliberately-duplicated
code path is *tested for equality* rather than hoped about, and there are no dead exports
(checked mechanically: every name in every `__all__` is imported somewhere).

The findings below are therefore mostly **accreted speculative flexibility** and **doc
drift**, not defects. Two exceptions worth taking seriously: `generate_dataset()`'s 21-
parameter signature (C-1) and the `--policy vision` lie (A-1).

---

## A. Open ends — paths that exist but don't work

### A-1 · `--policy vision` claims to be unimplemented, but the feature shipped · **FIX**

`scripts/run_episode.py:169-171`

```python
# ponytail: vision (Phase-2 vision-conditioned residual) isn't trained yet; fail
raise SystemExit(f"--policy {policy} is not implemented yet.")
```

…and `scripts/run_episode.py:720` documents the flag as "not implemented" in its help text.

But the vision deploy path **shipped in LAB-83** (PR #80): `LearnedResidual` reads
`PolicyConfig.use_vision` off the checkpoint and selects the image branch itself
(`policy/residual_policy.py:_image_tensor`, `policy/config.py:38-52`), and there are three
trained vision checkpoints on disk (`outputs/policy/runs/vision_frozen_lab82`,
`vision_frozen_ar100`, `vision_stageC`).

So the single most interesting thing the project built cannot be run from its own CLI. The
comment is a stale ponytail note from before the feature existed.

**Verdict: FIX.** `--policy learned --checkpoint <vision.pt>` most likely already works
(the checkpoint self-describes); either make `vision` an alias that asserts
`config.use_vision`, or delete the flag value and document that vision is selected by the
checkpoint. Decide in Phase 2 — but the current state is actively misleading.

### A-2 · `keyboard` input promised in the README, never implemented · **DELETE the promise**

`README.md` ("Input strategies") lists **keyboard** — "*developer fallback, deferred (not
yet implemented)*". `docs/milestones.md` M8 scope also lists `KeyboardInput`.

There is no `KeyboardInput` class anywhere in `src/`.

**Verdict: DELETE the promise.** The two real strategies (`scripted`, `vision`) cover data
generation and the live demo. A keyboard fallback earns its keep only if a demo needs it,
and nothing does. Remove the line from `README.md` and mark it dropped in `milestones.md`
rather than carrying a permanent "coming soon" in the public README.

### A-3 · `PolicyConfig.use_tanh_head` — declared, never read · **DELETE (carefully)**

`policy/config.py:59`

```python
use_tanh_head: bool = False
```

Mechanically verified: **one occurrence in the entire repo** — the declaration. No model
code branches on it. It is a squatting knob for a bounded output head that was never built.

**Verdict: DELETE**, with one caution: `PolicyConfig` is serialized into every checkpoint
via `asdict(config)` and rebuilt with `PolicyConfig(**payload["config"])`
(`policy/residual_policy.py:100`). Dropping the field makes **every existing checkpoint
fail to load** with `unexpected keyword argument`. Phase 2 must add a
drop-unknown-keys shim in `load_checkpoint` in the same change — which is a good idea
regardless, since it also unblocks future field removals.

---

## B. Single-implementation abstractions

### B-1 · The hole-shape abstraction has five shapes and one implementation · **DELETE**

The scenegen package is built around `HoleShape = Literal["circle","rect","slot","keyhole","polygon"]`
(`sim/scenegen/config.py:26`). What actually exists:

```
sim/scenegen/sampler.py:32     IMPLEMENTED_SHAPES = ("circle",)
sim/scenegen/solid.py:38-41    _SHAPE_DISPATCH = {"circle": _drill_circle}   # 1 entry
```

The cost of the other four:

| Site | What it is |
|---|---|
| `config.py:118-127` | 5-branch bounding-radius `if/elif` chain |
| `shapes2d.py:29` | `NotImplementedError(f"hole shape {hole.shape!r} not implemented yet")` |
| `sampler.py:132` | same |
| `sampler.py:143` | `NotImplementedError(f"size sampling for {shape!r} not implemented yet")` |
| `solid.py:49` | same |
| `config.py:62-67` | 3 dead sampling-range fields — `slot_width_frac`, `polygon_radius`, `polygon_sides`, each with **1 occurrence in the repo** (the declaration) |

Four raise-sites and three dead config fields to support shapes nobody has asked for, in a
project whose task is *round peg into round hole*. This is the clearest over-engineering in
the repo.

**Verdict: DELETE.** Collapse to circle. Keep `HoleShape` only if `rect` is genuinely wanted
(`rect_side` is read, unlike the other three) — otherwise drop the `Literal` entirely and
delete all four raise-sites plus the dead ranges. Estimated: −80 lines, zero behaviour
change (`IMPLEMENTED_SHAPES` already gates the sampler to circle).

### B-2 · `PolicyConfig.use_command_ee_delta` — a knob for a measured dead end · **KEEP, but shrink**

`policy/config.py:16-24` (9 lines of comment), mirrored in
`policy/residual_policy.py:213-216` and in the training CLI (`--command-ee-delta`).

The wiki's own verdict (`synthesis/imitation-limits-closed-loop.md`): *"a **documented
dead-end** — keep only as a negative-result artifact."* It made the offline metric better
(7.63 → 3.46 mm) and closed-loop **worse** (collapse to 10% on the gate config).

**Verdict: KEEP — because** it is load-bearing evidence for the project's headline negative
result, and deleting it would make the M7 story unreproducible. **But shrink**: the 9-line
config comment can be two lines plus a pointer, and it should be flagged as
`# negative result — do not enable` so a future reader doesn't try it as an improvement.

### B-3 · `domain/interfaces.py` — two Protocols, ~2 implementations each · **KEEP — because**

`InputStrategy` (2 impls: `ScriptedNoisyHuman`, `VisionInput`) and `AssistProvider`
(3: `NoAssist`, `Expert`, `LearnedResidual`).

Normally a Protocol at that ratio is a smell. Here it is **the project's stated design
thesis** — dependency inversion at the assistance seam is what M3 exists to deliver, it is
argued in `project-scope.md`, and it is what let the learned policy drop in with *no edit
to the runner, controller, or input* (`policy/residual_policy.py:5-9`).

**Verdict: KEEP — because the seam is the contribution.** Recorded explicitly so this audit
doesn't read as inconsistent when it deletes B-1 for a similar-looking ratio. The
difference: B-1's variants were never built and nothing asked for them; B-3's variants all
exist and the swap is exercised in the ablation.

---

## C. Complexity concentrations

Measured, not eyeballed — every function in `src/` + `scripts/` with ≥60 lines or ≥10
parameters — 38 total, via a throwaway AST pass. The ones that matter:

| Location | Lines | Params | Note |
|---|---:|---:|---|
| `data/generate.py:266 generate_dataset()` | **240** | **21** | worst on both axes |
| `scripts/dev_harness_controller.py:164 main()` | 246 | 0 | a dev harness; acceptable |
| `sim/runner.py:73 run_episode()` | 135 | 10 | |
| `dagger.py:263 run_dagger()` | 133 | 15 | |
| `eval/ablation.py:110 run_trial()` | 108 | 13 | |
| `input/scripted_noisy_human.py:202 __init__()` | 76 | 15 | |
| `control/backbone.py:108 __init__()` | 69 | 12 | physics gains — arguably fine |
| `eval/ablation.py:221 run_paired()` | 45 | 13 | |

### C-1 · `generate_dataset()` takes 21 parameters · **FIX**

`data/generate.py:266`. Twenty-one positional/keyword parameters threaded through 240 lines,
and the same knobs then re-declared in `scripts/generate_dataset.py`'s argparse (150-line
`main()`) and again in `_episode_fingerprint()` (11 params, `generate.py:154`).

This is the one finding with a real correctness cost, not just an aesthetic one: the corpus
**fingerprint** must include every generation-affecting knob or a regenerated dataset
silently differs from its metadata. With 21 loose parameters, keeping the fingerprint in
sync with the signature is manual and unchecked.

**Verdict: FIX** — collapse into a frozen `GenerationConfig` dataclass (the codebase already
has this pattern: `SimConfig`, `PolicyConfig`, `TrainConfig`, `SamplingRanges`). The
fingerprint then derives from `asdict(config)` and cannot drift. This is the highest-value
refactor in the audit, and it is mostly mechanical.

**Caveat for Phase 2:** dataset metadata is committed and `regenerate_from_metadata()`
(`generate.py:599`) must keep reading old manifests byte-identically. Wrap, don't rewrite.

*Applied note:* deriving the hash from `asdict(config)` verbatim turned out to be wrong — see
C-1a. The payload keeps its legacy conditional structure; what makes drift impossible is a
test that perturbs **every** field and asserts the hash flips.

### C-1a · the fingerprint has two real holes · **DOCUMENT, do not fix**

Surfaced by `scripts/dev/lab42_fingerprint_audit.py`, written as the before/after guard for
C-1. It recomputes each committed manifest's fingerprint the way `regenerate_from_metadata()`
does and compares it to the stored value. **Two of ten manifests already disagree:**

```
DRIFT data/dataset_0  committed=0cb4240c72f75635  recomputed=00d01ffe4b68395f
DRIFT data/dataset_1  committed=290f175018e5b9d3  recomputed=01c463abd0fb84c2
```

Both are `schema_version` 1.0 (2026-06-16), written before `generated_walls` entered the hash
payload. So the exact failure mode C-1 warns about has *already happened once*: a knob was
added to generation and the two oldest manifests silently stopped reproducing their own hash.
They are superseded corpora (the live operating points are `dataset_9` and `dataset_vision`,
both `ok`), so rehashing them buys nothing — but it is the concrete evidence that the guard
was needed.

**Hole 2:** the three termination thresholds — `success_depth`, `lateral_tolerance`,
`force_cap` — are **not** in the payload, yet they *do* determine a trajectory: `EpisodeLogger`
returns truthy when one trips and `sim/runner.py:160-163` breaks the loop on it. Two corpora
differing only in `lateral_tolerance` therefore collide on fingerprint, and the cache would
happily reuse the wrong episodes. Committed history is split across both settings
(`dataset_2/3/4` at 0.006, `dataset_6`→`dataset_vision` at 0.010), so folding them in now
invalidates one group or the other whichever legacy default is chosen.

**Verdict: DOCUMENT.** The three names are listed in `_UNFINGERPRINTED` (`generate.py`) with
the reasoning inline, and `test_fingerprint_covers_every_config_field` asserts that set is
exactly the set of unhashed fields — so the hole cannot silently grow. Closing it properly
means a fingerprint **version** prefix (v2 payload includes the thresholds; v1 manifests keep
matching), which is a real change to a committed contract and belongs to whoever next needs to
vary a threshold — not to a cleanup pass.

### C-2 · `run_episode.py` is 891 lines but *not* a mess · **KEEP — correction to the plan**

The planning note flagged this file on raw LOC. Reading it, the size is **organized**:
`main()` is only 116 lines and the bulk is four argparse-builder functions
(`add_run_args` 79, `add_input_args` 64, plus two more) that LAB-87 already extracted.

**Verdict: KEEP.** The plan's "split CLI construction from the run loop" item is already
done. Dropping it from Phase 2.

### C-3 · `run_trial` / `run_paired` share 13 parameters · **FIX (low priority)**

`eval/ablation.py:110` and `:221`. LAB-107 was *caused* by exactly this: `max_steps`
defaulted differently in two places, and the DAgger path silently under-budgeted seating by
5000-vs-9000 steps. The fix added a signature-parity regression test
(`tests/test_ablation.py`) — a test that only exists because the signature is too wide to
eyeball.

**Verdict: FIX (low priority)** — a shared `TrialConfig` would make the parity test
unnecessary. Worth doing only if C-1's refactor establishes the pattern anyway.

*Applied note:* fixed by **deletion**, not by a config object — `run_paired` never *used* the
13 parameters, it only forwarded them, so `**trial_kwargs` removes the duplicate defaults
outright. A config type here would have added a third home for the same values (alongside
`run_trial`'s signature and the `Config` ablation arm). C-1's pattern was the right call for
`generate_dataset`, which genuinely *owns* its knobs; it is the wrong call for a forwarder.

---

## D. DRY

### D-1 · Two rotation-helper modules, arbitrarily split · **FIX**

| Module | Exports | Used by |
|---|---|---|
| `common/geometry.py` | `mat3_to_quat`, `quat_mul`, `quat_conjugate` | `sim/scene.py`, `input/vision_input.py`, `input/hand_tracker.py` |
| `common/utils/rotations.py` | `quat_to_matrix`, `axis_from_quat`, `quat_to_6d` | `common/seating.py`, `expert/expert.py`, `data/dataset.py`, `policy/residual_policy.py`, 8 dev scripts, `tests/test_expert.py` |

Six thin MuJoCo wrappers, same purpose, two files, two import paths, in the same package.
Only `geometry`'s `mat3_to_quat` is re-exported from `common/__init__.py`; the `utils`
module is reached by deep import everywhere. `common/utils/rotations.py` is also the only
module in `src/` with **no docstring**, against an otherwise rigorous convention.

**Verdict: FIX** — merge into `common/geometry.py`, export all six from `common/__init__`,
delete `common/utils/`. ~20 import sites, mechanical, ruff will catch any miss.

### D-2 · The dataset↔deploy stream duplication is **correctly** handled · **KEEP — because**

`data/dataset.py:143 extract_training_episode()` (batch, `(T, …)`) and
`policy/residual_policy.py:189 _assemble_streams()` (single step) build the same three input
vectors independently. That is a textbook silent-covariate-shift trap.

It is **guarded**: `tests/test_residual_policy.py:98` — *"The wrapper's per-step streams must
equal `extract_training_episode`'s"* — asserts equality directly, and both sites carry
cross-referencing comments (`residual_policy.py:196`, `:213`).

**Verdict: KEEP — because** the duplication is intrinsic (batch vs. O(1) per-tick) and the
equality is *tested*, not assumed. This is the right way to handle unavoidable duplication
and should be cited as such rather than "fixed".

### D-3 · `PEG_HALF_LENGTH` re-hardcoded in dev scripts · **DELETE (with the scripts)**

`common/seating.py:29` defines `PEG_HALF_LENGTH = 0.030` and its docstring explains it
exists precisely so the constant isn't duplicated. Then:

- `scripts/dev/sweep_krot_multiseed.py:55` — `0.030 * axis_from_quat(...)`
- `scripts/dev/sweep_rotational_stiffness.py:62` — same

Both are in the delete pool below, so this resolves itself.

---

## E. Hygiene

| ID | Finding | Verdict |
|---|---|---|
| E-1 | `mypy` covers `files = ["src"]` only (`pyproject.toml`). `scripts/` + `tests/` — 14.7k LOC, more than `src/` — are unchecked. | **FIX** — extend to `scripts` + `tests`, or scope it deliberately and say why in `pyproject.toml`. Expect an initial batch of errors in `scripts/dev/`; that pool is shrinking anyway (F-1). |
| E-2 | `stereohand` pinned by bare git URL, no tag/SHA (`pyproject.toml`, `stereo-input` extra). | **FIX** — pin to a tag. D2's acceptance is "clean clone → run"; an unpinned git dep can break that silently. |
| E-3 | ~2.4 GB of untracked `data/` dirs (`lab108_*sweep` ×4, `dagger_*_agg` ×2, `dataset_42`, `_gpu_render_probe`, `dataset_vision_probe`). Kept out of commits only incidentally by the `runs/` + `*.npz` rules — their `metadata.json` files are **not** ignored and will be picked up by a careless `git add`. | **FIX** — `.gitignore` the sweep/probe dirs by pattern. |
| E-4 | `outputs/policy/kpi_report/kpi_comparison.json` still reads `"vision_residual": "PENDING — needs vision deploy path … LAB-83"`. LAB-83 closed 2026-07-08. | **FIX in Phase 3** (it's a results artifact, not code). |
| E-5 | LAB-107/108 (`178ce81`) sits unmerged on `feat/lab-108-expert-slam-prevention` with no PR. `master` lacks the eval-path fix that makes cross-path numbers comparable. | **Naveh's call** — open PR #85 before or after this review. |

---

## F. `scripts/dev/` triage

67 files, 6,952 LOC — **more code than `scripts/` and 72% the size of `src/`** — in a repo
whose README calls itself a public showcase.

Mechanical signal: **40 of 67 are referenced by no doc, no wiki page, and no `.claude`
rule.** Those are the deletion pool. The other 27 are cited somewhere and need judgment.

| Bucket | Count | Verdict |
|---|---:|---|
| **Delete** — one-off lab probes whose conclusion is already in the wiki | ~40 | `DELETE`. The finding is the artifact; the script is scaffolding. Includes `lab77_*`, `lab78_*`, `lab81_vision_bringup_report`, `lab104_*`, `lab105_stagec_ablation_summary`, `lab106_ft_checkpoint_sweep`, `debug_*` (5), `sweep_k_rot`/`sweep_krot_multiseed`/`sweep_rotational_stiffness`/`sweep_joint_damping`, `probe_*` (5), `verify_*` (4), `spotcheck_*` (3), `manual_test_*` (2), `demo_jacobian_transpose`, `demo_null_space`, `aggregate_eval` (its own header says *"throwaway dev aggregator"*). |
| **Keep** — reusable instruments, still cited | ~20 | `record_comparison_grid.py` (LAB-42's own demo tool), `policy_latency.py`, `render_cost_probe.py`, `profile_render_vs_sim.py`, `loop_rate_probe.py`, `poll_rate_probe.py`, `compare_human_vs_scripted.py`, `lab95_*`, `lab98_expert_recalibration_sweep`, `lab105_perception_probe`, `lab106_{delta_target_audit,error_decomp}`, `lab108_{brake,align}_sweep`, the stereo/camera probes. |
| **Promote** to `scripts/` | ~3 | `record_comparison_grid.py` is the project's demo-video tool and belongs on the `kvn` CLI, not in `dev/`. |

**Verdict: DELETE the ~40.** Every one is recoverable from git history, and each deleted
file's conclusion is already durable in `project-wiki/`. Exact list to be confirmed
one-by-one in Phase 2 — the counts above are the mechanical first pass, not the final call.

---

## What is explicitly good (do not "fix" these in Phase 2)

Recorded so a later cleanup pass doesn't mistake deliberate design for accident:

1. **The dependency DAG is real.** `domain/` imports nothing from the package; `common/` is
   sim-free; `common/geometry.py` and `common/seating.py` both declare themselves DAG leaves
   and are. This is what makes the seam swap actually work.
2. **Constants carry their experiments.** `_MAX_DELTA_POSITION` (`domain/delta.py:28-34`)
   explains why 2 cm became 3 cm, which issue changed it, and how legacy corpora stay
   byte-identical. `SamplingRanges.chamfer` (`scenegen/config.py:69-77`) embeds the whole
   LAB-77 sweep. Nobody has to re-derive these.
3. **Back-compat is deliberate.** Every new `PolicyConfig` field is defaulted specifically so
   old checkpoints deserialize (`config.py:38-40`); corpora fingerprint their own Δ-clamp so
   legacy datasets regenerate identically (`domain/delta.py:32`).
4. **The MuJoCo footguns are documented at the call site.** `impedance.py:120-125` explains
   that `mju_subQuat` returns a *body-frame* axis-angle so no `R.T` is needed — the kind of
   comment that prevents a wrong "fix".
5. **No dead exports.** Mechanically checked: every name in every `__all__` is imported
   somewhere in `src`/`scripts`/`tests`.
6. **Ponytail markers are used correctly** — as named ceilings with upgrade paths
   (`residual_policy.py:233`, `runner.py:193`), not as apologies.

---

## G. Round-2 findings — Naveh's read (2026-07-22)

Two findings from Naveh reading the post-Phase-2 tree. Both verified with `file:line` and, for
G-2, against episodes on disk. **Both fixed 2026-07-22** — outcomes at the end of this section.

### G-1 · Training lives in `scripts/`, so `dagger.py` shells out to it · **FIX**

`dagger.py:291-372` imports `subprocess` and `sys` inside `run_dagger()`, resolves
`scripts/train_policy.py` by path (`parents[2] / "scripts" / "train_policy.py"`), builds a
14-element argv, and runs it with `check=True` — once per DAgger round.

The subprocess is a **symptom, not the cause**. `dagger.py` is in `src/ai_teleop/`; `scripts/`
is not a package and is not importable from installed code, so shelling out is the only option
*given where training lives*. The actual finding is that **training is core functionality
sitting in a script**: `scripts/train_policy.py` is 504 lines and holds `train()` (the epoch
loop, early stopping, best-weight restore) plus `main()`, while a slice of the same
concern — `policy/run_artifacts.py`, 150 lines of run-folder/history/provenance writing — is
already in the package. The seam between them is arbitrary.

This is the one place the repo breaks its own stated pattern. `scripts/generate_dataset.py`
opens by declaring it: *"The generation pipeline is core functionality and lives in the package
(`ai_teleop.data.generate`); this script is just its command-line front door."* Data generation
follows that. Training does not — and `dagger.py` pays for it.

Concrete costs, not stylistic ones:

- **No type checking across the call.** The 14-element argv is stringly-typed; mypy checks
  neither the flag names nor the value types. A renamed `--action-rate-weight` fails at
  runtime, mid-round, after the rollouts have already been simulated.
- **Errors arrive as an exit code.** `check=True` raises `CalledProcessError` — no exception
  type, no traceback into the training loop, no structured result. The return value that
  matters (the checkpoint path) is *reconstructed by string convention*
  (`runs_root / run_name / "checkpoint.pt"`, `dagger.py:373`) rather than returned.
- **Process-per-round overhead** — a fresh interpreter and a fresh `import torch` every round.

**Verdict: FIX** — move the training pipeline into `ai_teleop.policy.train` (mirroring
`ai_teleop.data.generate`), leave `scripts/train_policy.py` as the thin argparse front door it
claims to be, and have `dagger.py` call `train_policy(...)` directly. That is the same
front-door-over-package refactor the repo already applies everywhere else, and it makes the
subprocess disappear on its own.

**Also asked: does anything else shell out where an import would do?** Swept `src/`,
`scripts/`, `tests/` — five `subprocess` sites, and only one more is this pattern:

| Site | Verdict |
|---|---|
| `dagger.py:352` → `scripts/train_policy.py` | **the finding above** |
| `scripts/dev/record_then_render.py:66,80` → `scripts/run_episode.py` (×2) | same shape, dev script; low stakes, but it inherits the fix if `run_episode`'s core ever moves into the package |
| `cli.py:117` → the `kvn` target script / poe task | **correct.** A launcher's job is to launch; sub-process isolation is the point. |
| `policy/run_artifacts.py:52` → `git rev-parse` | **correct.** External binary. |
| `scripts/dev/lab42_fingerprint_audit.py:24` → `git ls-files` | **correct.** External binary. |

### G-2 · The metadata schema exists and the writers ignore it · **FIX**

`data/schema.py:85-140` defines `EpisodeMetadata` — a thorough `TypedDict` (17 required keys +
a documented `total=False` tail for the replay spec and the LAB-96/98/100 knobs). So the single
place that defines the contract *does* exist. It is enforced on exactly one side:

| Side | Typed? | Where |
|---|---|---|
| **Readers** | ✅ `EpisodeMetadata` | `trajectory.py:162` `load_episode`, `dataset.py:144`, `run_episode.py:174,233,270` |
| **Writers** | ❌ `dict[str, object]` | `generate.py:496` and `dagger.py:192`; `EpisodeRecorder.save(metadata: dict[str, object])` (`trajectory.py:149`) passes it straight through |

`load_episode` *annotates* the result of `json.loads` as `EpisodeMetadata` (`trajectory.py:165`)
— a declaration on trust. Nothing verifies the blob ever matched. So the contract is asserted
where it is consumed and unchecked where it is produced, which is exactly backwards.

**It has already drifted.** Two hand-rolled writer dicts exist, and they disagree — `dagger.py`
omits `expert_d_far`, `speed_lognormal_median`, `speed_lognormal_sigma`, `expert_brake_gain`,
`expert_brake_lead_floor` and `delta_clamp`. `expert_d_far` is **required** in
`_EpisodeMetadataBase`, so every DAgger episode on disk violates the declared schema:

```
$ data/dagger_agg1/runs/episode_1000021/episode.npz
missing required keys: ['expert_d_far']
```

No consumer breaks *today* — `expert_d_far` happens to be read from the dataset-level config
(`dagger.py:104`), never from the episode blob — so this is a latent violation, not a live bug.
That is precisely why it survived: mypy cannot see it, and no test asserts a written blob
against the schema. The same untyped-writer pattern covers the other two on-disk shapes:
`_episode_summary` (`generate.py:558`) and `_write_dataset_metadata` (`generate.py:619`) both
name their `EpisodeSummary` / `ResBCDatasetMetadata` shape **in a docstring** instead of
annotating it. C-1 has just made the inconsistency visible — `GenerationConfig.to_dataset_config()
-> DatasetConfig` is typed, and it sits ten lines from two dicts that are not.

**Verdict: FIX** — annotate the writers (`-> EpisodeMetadata`, `-> EpisodeSummary`,
`config: DatasetConfig`), tighten `EpisodeRecorder.save` to take `EpisodeMetadata`, and let
mypy close the loop. Expect it to *fail first* on `dagger.py`'s missing keys — that failure is
the finding proving itself, and the fix is to stamp the corpus knobs DAgger rollouts actually
ran under. Where DAgger genuinely has no value for a key, move that key out of the required
base rather than writing a fake one.

---

## Coverage — what this round actually read

Honest scope, so round 2 knows where to look:

- **Read closely:** `domain/*`, `common/*` (all), `control/*` (all), `expert/expert.py`,
  `policy/{__init__,config,residual_policy}.py`, `cli.py`, `scenegen/config.py`.
- **Surveyed structurally** (AST metrics, grep, import graph, usage counts): everything else
  in `src/` + `scripts/`.
- **Not yet read line-by-line:** `data/generate.py`, `data/dataset.py`, `data/step_callbacks.py`,
  `sim/scene.py`, `sim/runner.py`, `eval/{observer,report,trace}.py`, `dagger.py`,
  `input/{vision_input,scripted_noisy_human,hand_tracker}.py`, the test suite.

Round 2 priority: `data/generate.py` (C-1's target) and `eval/report.py` — the latter feeds
the KPI dashboard, so a bug there would corrupt Phase 3.

---

## Phase 2 shortlist, ranked

| # | Finding | Effort | Why it's ranked here |
|---|---|---|---|
| 1 | **A-1** `--policy vision` | S | The repo lies about its own headline feature |
| 2 | **C-1** `generate_dataset()` → config object | M | Only finding with a correctness cost (fingerprint drift) |
| 3 | **B-1** collapse hole shapes to circle | S | −80 lines, zero behaviour change |
| 4 | **F** delete ~40 dev scripts | S | −4k LOC off a public showcase repo |
| 5 | **D-1** merge the two rotation modules | S | Mechanical, ruff-verified |
| 6 | **A-3** drop `use_tanh_head` + add checkpoint key shim | S | Shim is worth having regardless |
| 7 | **E-1/E-2/E-3** mypy scope, pin stereohand, gitignore | S | Hygiene; E-2 protects D2 acceptance |
| 8 | **A-2** delete the keyboard promise | XS | One line of README |
| 9 | **B-2** shrink the `command_ee_delta` comment | XS | |
| 10 | **C-3** shared `TrialConfig` | M | Only if C-1 establishes the pattern |

---

## Outcomes (LAB-110, 2026-07-21)

Gate after every change: **ruff clean, mypy clean, 227 tests pass** (was 226 — one added).

| # | Finding | Status | Note |
|---|---|---|---|
| 1 | A-1 `--policy vision` | ✅ done | Flag value removed; the **checkpoint** now selects modality (`use_vision`), and `main` auto-enables wrist capture, duck-typed exactly like `eval.ablation.run_trial`. Proven end-to-end: `kvn episode --policy tf --checkpoint .../vision_frozen_lab82/checkpoint.pt` runs, logs *"vision checkpoint — enabling wrist capture every 20 ticks"*, and completes. Also de-duplicated `DEFAULT_WRIST_RENDER_EVERY` into `sim/scene.py` as the one source of truth. |
| 2 | C-1 `generate_dataset()` | ✅ done | 21 params → 8; the 14 corpus knobs became a frozen `GenerationConfig` that owns `fingerprint()`, `to_dataset_config()` and `from_metadata()`. The last two kill a duplicated legacy-defaults block in `regenerate_from_metadata()`, which shrank 60 lines → 12. The payload construction is byte-identical, verified two ways: `scripts/dev/lab42_fingerprint_audit.py` recomputes all 10 committed manifests' fingerprints before/after (identical output), and the pinned dataset_6 / dataset_8 golden hashes still assert. New guard: `test_fingerprint_covers_every_config_field` perturbs every field and asserts the hash flips — an unhashed new knob now fails the suite. CLI flag defaults now read off `GenerationConfig()` instead of nine imported `DEFAULT_*` constants. |
| 3 | B-1 hole shapes | ✅ done | −79 lines. `HoleShape` narrowed to `Literal["circle"]`; 4 `NotImplementedError` sites, the 1-entry dispatch dict, the 5-branch bounding chain and 3 dead `SamplingRanges` fields all gone. **Verified byte-identical geometry** by generating wall seed 17 under a git worktree at the pre-change commit: every hole position, diameter, chamfer, orientation and all 9 mesh hashes match. Only the header's provenance block loses the 4 dead keys. |
| 4 | F dev-script prune | ✅ done | 67 → 37 files (**−31**, ~3.5k LOC). Kept every script *cited* by `src/`, `docs/`, or the wiki — including brace-form citations like `lab106_{delta_target_audit,error_decomp,ft_checkpoint_sweep}.py`, which a naive filename grep misses. Deleting a script that a source comment names as a constant's provenance would break the "constants carry their experiments" property this audit praised. |
| 5 | D-1 rotation modules | ✅ done | `common/utils/` deleted; all six helpers now live in `common/geometry.py` and are exported from `common/__init__`. ~20 import sites rewritten. |
| 6 | A-3 `use_tanh_head` | ✅ done | Removed, **with** a drop-unknown-keys shim in `load_checkpoint` that logs what it drops. Verified against all 13 on-disk checkpoints — every one carried `use_tanh_head`, so without the shim the removal would have stranded the project's entire trained-model history. Regression test: `test_checkpoint_with_retired_config_key_still_loads`. |
| 7 | E-2 / E-3 | ✅ done | `stereohand` pinned to `@v0.1.0`; `data/` sweep + probe dirs gitignored by pattern (named `dataset_<n>/` dirs deliberately still visible, since their `metadata.json` is committed on purpose). |
| 7 | E-1 mypy scope | 🟡 **tests done 2026-07-22; `scripts/` → LAB-113** | Turning on `scripts` + `tests` surfaces **41 errors in 18 files** — five of them likely real bugs (a TypedDict mismatch in recorded metadata, a success-rate generator typed `object`, a Pillow-10 break in the demo-grid tool). Too large to fold in here without masking regressions. One freebie taken: `_fast_exit` is now `-> NoReturn`, which was causing a false *Missing return statement* on `main()`. |
| 8 | A-2 keyboard promise | ✅ done | Removed from `README.md`; struck through in `docs/milestones.md` with the reason. |
| 9 | B-2 `command_ee_delta` | ✅ done | Comment cut 7 lines → 5 and re-headed **`NEGATIVE RESULT — do not enable`**. Knob kept: it is load-bearing evidence for the M7 result. |
| 10 | C-3 `TrialConfig` | ✅ done — **not** as specified | No new config type. `run_paired` was a pure forwarder that re-declared 10 of `run_trial`'s defaults; it now takes `**trial_kwargs` and forwards. −25 lines, no new abstraction, and the defect class is *gone* rather than re-housed: there is exactly one definition of each default left. All four call sites already passed keywords, so none changed. The LAB-107 parity test now asserts `run_trial`'s default **and** that `run_paired` no longer has a second copy. (A shared `TrialConfig` would have been a third place for the same values to live, next to the existing `Config` ablation-arm type — worse, not better.) |

### Correction to finding F

One file was lost that should not have been: **`scripts/dev/lab104_residual_magnitude.py`**. It
was untracked (never committed), so the delete is unrecoverable from git. It *is* cited by the
wiki (`concepts/vision-conditioned-policy.md:310` and `log.md:1154`) as the source of the
deployed-vs-expert Δ-magnitude numbers (vision 0.0068, F/T 0.0095 vs expert 0.0052 m/step).

The **finding survives** — those numbers are recorded in the wiki — but the citation now dangles.
Phase 3 must either reconstruct the script or annotate the citation. It has not been silently
rewritten, because a reconstruction that produced different numbers would be worse than none.

### Also deferred

`record_comparison_grid.py` was **not** promoted to `scripts/` + the `kvn` CLI as planned:
mypy flags `Image.NEAREST` at line 99, removed in Pillow 10, so the project's demo-video tool
may currently be broken. Promoting a broken tool is worse than leaving it. Tracked in LAB-113.

### G-3 · `run_episode` types its controller nominally, unlike its two other collaborators · **KEEP — because**

`sim/runner.py:73-77`. The runner takes `input_strategy: InputStrategy` and
`assist: AssistProvider` — both Protocols from `domain/interfaces.py` — but
`controller: Controller`, the concrete class. It only ever touches `controller.compute()`
(`:165`) and `controller.status` (`:204`), so the nominal type is wider than the dependency.

Surfaced by turning mypy on for `tests/`: `tests/test_episode_e2e.py`'s `_RecordingController`
is a structural stand-in that wraps a real `Controller` to record commands, and it needs an
`arg-type` ignore at each call site despite satisfying everything the runner uses.

**Verdict: KEEP — because**, by this audit's own rule (see B-1 vs B-3). A Protocol earns its
keep when *the seam is the contribution* — that is true of the assistance seam (`InputStrategy`,
`AssistProvider`), which is what M3 exists to deliver and what let the learned policy drop in
untouched. The controller is the seam's downstream *consumer*, not the seam itself, and it has
exactly one implementation. Adding a `ControlLaw` Protocol here would be the same
one-implementation abstraction B-1 deleted. Two `type: ignore`s in one test file is the cheaper
side of that trade — recorded here so a future reader doesn't "fix" it and re-introduce the
inconsistency in the other direction.

### G outcomes (2026-07-22)

Gate green after both: ruff clean, mypy clean (**60 files**, was 59), **230 tests** (was 229).

| # | Status | What landed |
|---|---|---|
| G-1 | ✅ done | Training moved to `ai_teleop/policy/train.py`; `scripts/train_policy.py` is now a 200-line argparse front door over it (was 504 lines of pipeline). New `train_policy()` returns a frozen `TrainedRun` carrying `checkpoint_path`. `dagger.py` calls it directly — the `subprocess`/`sys` imports, the resolved script path and the 14-element argv are gone, as is the `runs_root / name / "checkpoint.pt"` string convention. |
| G-2 | ✅ done | `EpisodeMetadata` split into `EpisodeSpec` (what a writer supplies) + `EpisodeMetadata` (adds the two keys `EpisodeRecorder.save` stamps). Writers annotated: `save()`, both `episode_metadata` blobs, `_episode_summary`, `_summary_from_cache`, `_write_dataset_metadata`, `seed_aggregate`, `append_summaries`, `rollout_and_relabel`. mypy now checks the write side. |

**G-1 paid for itself immediately.** Moving 400 lines from `scripts/` into `src/` put them under
mypy for the first time (`scripts/` is still out of scope — LAB-113), which surfaced **5 real
type errors** in code that had been running for months: `per_step_image_embedding` called with
`Tensor | None` on both arguments, and three `len(loader.dataset)` calls on a `Dataset` that
isn't `Sized`. Fixed with an assert documenting the invariant (`train_policy` ties
`load_images` to `config.use_vision`, so a vision batch always has frames) and one `_n_episodes`
helper. Two `PolicyConfig`-vs-argparse desync paths closed as a side effect: `load_images` and
`command_ee_delta` now derive from the config object rather than being passed separately.

**G-2 failed first exactly where predicted.** With the writers annotated, mypy flagged
`dagger.py` and nothing else — three errors, all the missing-keys drift. The fix stamps the
values the rollout genuinely ran under, read from the same `config` that builds the relabeling
expert (`expert_from_config`) and seeds the operator: `expert_d_far`, `expert_brake_gain`,
`expert_brake_lead_floor`, `delta_clamp`, `speed_lognormal_median`, `speed_lognormal_sigma`.
Nothing was invented and no key was demoted out of the required base. Also removed a bare `0.03`
magic number in `expert_from_config` (now `DEFAULT_DELTA_CLAMP`) and de-duplicated the
speed-draw config reads, which had been done twice in one function.

The bug class is now caught **twice**, verified by mutation — deleting the `expert_d_far` line
gives:

```
mypy:    dagger.py:211: error: Missing key "expert_d_far" for TypedDict "EpisodeSpec"
pytest:  AssertionError: missing required keys: ['expert_d_far']
```

The runtime half is `tests/test_dagger.py`, asserting a *written* episode against
`EpisodeMetadata.__required_keys__` — static typing alone can't prove what actually reached
disk, which is how the original drift survived.

**Not done, deliberately:** the existing `data/dagger_*_agg/` episodes on disk still lack the
key. They are untracked scratch corpora from the LAB-105 rounds, already superseded, and
rewriting their metadata would change artifacts whose provenance is the point. The fix applies
to everything written from here.

### E-1 revisited — `tests/` is now type-checked (2026-07-22)

E-1 (mypy scope) was split to LAB-113 in Phase 2 on the strength of "41 errors in 18 files".
Re-measured after C-1/C-3/F/G-1 landed, that number had collapsed: **`tests/` was down to 2
errors** — most of the original debt was in the 31 dev scripts finding F deleted and in the
400 lines G-1 moved into `src/`.

But 2 was a trap. `mypy` **skips the bodies of unannotated functions by default**, and pytest
test functions are unannotated (**218 of 229** here), so adding `tests` to `files` alone would
have checked almost nothing while reporting a green gate — type-checking theatre. With
`check_untyped_defs = true` the true figure was **20 errors in 7 files**, all fixed:

| Kind | Count | What it was |
|---|---:|---|
| Optional never narrowed | 9 | `provider._ft_bias` / `_hidden` (`ndarray \| None`, `Tensor \| None`), `meta["wall_seed"]` (`int \| None`) used directly. A `None` there fails as an obscure `AttributeError`/`TypeError` mid-assertion instead of at an explicit `assert`. Fixed by narrowing. |
| Deliberately partial fixtures | 6 | Tests that write minimal/legacy metadata blobs on purpose (the loader's back-compat cases, the recorder round-trip, the empty-save rejection). Now carry a narrow `typeddict-item` ignore **and a comment saying the sparseness is the point** — the annotation documents intent instead of hiding drift. |
| Stub/nominal-typing warts | 5 | `DataLoader(list)` and `len(Dataset)` (the torch stubs under-type both), and G-3's `_RecordingController`. Consolidated behind one `_episode_loader` helper and narrow ignores. |

`src/` was already clean under `check_untyped_defs`, so the flag is set globally rather than
per-module. Gate coverage went **60 → 86 files**. `scripts/` (19 errors in 8 files, including
the Pillow-10 break in the demo-grid tool) stays out — that remains LAB-113.
