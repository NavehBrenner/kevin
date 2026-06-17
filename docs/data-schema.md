# Trajectory Schema — the M4→M5 Data Contract

The data-generation driver (`scripts/generate_dataset.py`, LAB-28) writes **one
file per episode**; M5's dataset loader (LAB-32) reads them. This schema is the
*only* thing M5 depends on — it is versioned (`SCHEMA_VERSION`) and frozen by
meaning. Everything else about data generation (noise magnitudes, gate
constants, scene layout) may change without breaking M5.

Authoritative definition + writer/reader: `src/ai_teleop/data/trajectory.py`.

## Dataset layout

One directory per master seed (LAB-47); **one sub-directory per episode**
(`SCHEMA_VERSION` 2.0):

```
data/dataset_<seed>/
    metadata.json              # dataset-level statistics (see below)
    runs/
        episode_00000/
            episode.npz        # per-episode trajectory — the BC corpus
            imgs/              # per-step wrist-cam frames; empty unless
                               # generated with --render-images (vision is M7)
        episode_00001/
            episode.npz
            imgs/
        ...
```

The per-episode folder (rather than a flat `episode_NNNNN.npz`) lets a rendered
frame stream sit beside its trajectory without a second index — the M5 corpus is
F/T-only so `imgs/` stays empty, but the contract is ready for M7 vision. Build
these paths with the `episode_dir` / `episode_npz_path` / `episode_imgs_dir`
helpers in `trajectory.py`.

`generate_dataset.py` defaults `--out` to `data/dataset_<seed>`; pass `--out` to
override. The episodes are the contract M5 trains against; `metadata.json` is a
human/tooling-facing summary (not consumed by the loader).

**What is and isn't committed.** The episode trajectories (`runs/**/episode.npz`,
plus any `imgs/`) are git-ignored — large and fully regenerable. `metadata.json` **is** committed: it
records every trajectory-determining input, so it both documents a dataset and
can rebuild it. Regenerate the byte-identical episodes from a committed metadata
file with:

```
python scripts/generate_dataset.py --from-metadata data/dataset_<seed>/metadata.json
```

This reads the seed + config back out, re-runs generation next to the metadata
(or to `--out`), and verifies the regenerated `fingerprint` matches — a mismatch
warns that code or config has drifted from when the dataset was authored. (The
refreshed `metadata.json` is identical except its `generated_at` timestamp.)

## Format

**NPZ** (`numpy.savez_compressed`), one per episode, at
`runs/episode_<index:05d>/episode.npz`. Chosen over Parquet because it needs no
extra dependency (pyarrow/pandas live only in the `ml` extra, absent from CI) and
the M5 loader accepts NPZ. Each per-step column is a stacked `(T, …)` array;
episode metadata is a JSON string under the `metadata` key.

Load with `ai_teleop.data.load_episode(path) -> (columns, metadata)`.

## Per-step columns

All world-frame; SI units (metres, radians via quaternion, newtons); quaternions
`(w, x, y, z)`. `T` = number of control steps in the episode.

| column | shape | meaning |
|---|---|---|
| `step` | `(T,)` | control-step index (0-based) |
| `sim_time` | `(T,)` | seconds since reset |
| `wrist_ft` | `(T, 6)` | wrist wrench `(Fx,Fy,Fz,Mx,My,Mz)`, **bias-subtracted** |
| `joint_positions` | `(T, 7)` | arm joint angles |
| `joint_velocities` | `(T, 7)` | arm joint velocities |
| `ee_pose` | `(T, 7)` | TCP pose |
| `gripper_width` | `(T,)` | finger opening (m) |
| `cmd_position` | `(T, 3)` | operator command position (pre-Δ) |
| `cmd_quaternion` | `(T, 4)` | operator command orientation |
| `cmd_grip` | `(T,)` | operator command Δgrip force |
| `delta_position` | `(T, 3)` | **expert Δ position — BC target** |
| `delta_orientation` | `(T, 3)` | **expert Δ orientation (axis-angle) — BC target** |
| `delta_grip` | `(T,)` | **expert Δ grip force — BC target** |
| `peg_pose` | `(T, 7)` | privileged true peg body pose |
| `target_hole_pose` | `(T, 7)` | privileged true target-hole pose |
| `distance` | `(T,)` | privileged tip→hole distance `d` |
| `step_success` | `(T,)` | bool — peg inserted at this step |

The training row M5 assembles is `(observation streams from the first columns,
expert Δ as the target)`. The **privileged** columns (`peg_pose`,
`target_hole_pose`, `distance`) are for offline analysis only — never an input to
a deployed policy.

### F/T bias subtraction

`wrist_ft` is **bias-subtracted**: the driver tares against the wrist wrench at
reset (the static gravity load of the grasped peg in free space) and subtracts it
from every row, so the logged channel is contact-only — what a real F/T sensor
gives after taring. `Observation.wrist_ft` itself stays **raw**.

### Windowing is M5's job

These rows are **flat per-step**. Assembling the windowed streams the policy
consumes (`H_c×7` command history, `H_f×6` F/T history, …) is the M5 dataset
loader's responsibility, not M4's.

## Per-episode metadata (the `metadata` JSON)

| key | meaning |
|---|---|
| `schema_version` | this schema's version (`"1.0"`) |
| `n_steps` | episode length `T` |
| `master_seed`, `episode_index` | reproducibility key — regenerates the episode exactly |
| `scene_seed` | `[master_seed, episode_index]` — entropy fed to the scene/"wall" RNG (target hole + joint start offset) |
| `human_seed` | concrete int seeding the scripted operator (derived from the key) |
| `fingerprint` | hash of all trajectory-determining inputs (used for the regen cache) |
| `max_dpos`, `expert_d_far` | controller clamp / expert engagement distance used |
| `target_hole_index` | which hole was the active target |
| `terminal_reason` | `success` \| `force_abort` \| `timeout` (combined human **+** expert) |
| `episode_success` | bool (`terminal_reason == success`) |
| `success_depth`, `lateral_tolerance`, `force_cap` | the terminal-condition thresholds used |
| `baseline_terminal_reason` | *(if baseline ran)* same episode with the expert replaced by `NoAssist` |
| `baseline_success` | *(if baseline ran)* bool — did the **human alone** seat the peg? |

## Terminal conditions (privileged, in the driver)

The driver — not the controller, which stays mode-less — classifies each episode
from privileged geometry:

- **success** — insertion depth past `success_depth` along the bore with lateral
  error under `lateral_tolerance`.
- **force_abort** — wrist force magnitude exceeds `force_cap`.
- **timeout** — step budget reached without success.

`terminal_reason` scores the **combined** system actually rolled out: noisy human
command + expert residual. It is *not* the expert alone or the human alone.

**All episodes are kept** (failures included): diverse state coverage helps BC.

## Human-only baseline (LAB-47)

Unless disabled with `--no-baseline`, each episode is re-run on the **same scene
and the same operator command stream** with the expert replaced by `NoAssist` (no
residual), scored with the identical termination logic but **not** saved as a
trajectory. This measures what the noisy human achieves *without* corrections, so
the expert's lift is legible rather than implicit. The per-episode outcome is
stamped into the trajectory metadata (`baseline_*` above) and aggregated into
`metadata.json`.

## Dataset `metadata.json`

Written once per dataset by the driver (not by the schema layer; not read by the
M5 loader). Aggregates the run:

| key | meaning |
|---|---|
| `schema_version`, `master_seed`, `n_episodes` | dataset identity |
| `generated_at` | UTC ISO-8601 timestamp |
| `fingerprint` | shared trajectory fingerprint for this config |
| `config` | `max_steps`, `max_dpos`, `expert_d_far`, the thresholds, and `scene` |
| `expert` | `{counts: {reason: n}, success_rate}` — the combined human+expert system |
| `baseline_no_assist` | *(if baseline ran)* same shape — the human-only rate |
| `expert_lift` | *(if baseline ran)* `expert.success_rate − baseline.success_rate` |
| `episodes` | per-episode list: index, `file`, `n_steps`, hole, reasons, successes |

A `success_rate` is `null` if any episode is missing that outcome (e.g. a cached
dataset generated before the baseline existed, re-summarized with baseline on).

## Anti-scope

- **Wrist-camera frames** are not in the schema — image rendering + decimation
  into the corpus is **M7** (Phase 2). Phase-1 training uses F/T + proprioception
  + command only.
