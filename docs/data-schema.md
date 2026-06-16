# Trajectory Schema — the M4→M5 Data Contract

The data-generation driver (`scripts/generate_dataset.py`, LAB-28) writes **one
file per episode**; M5's dataset loader (LAB-32) reads them. This schema is the
*only* thing M5 depends on — it is versioned (`SCHEMA_VERSION`) and frozen by
meaning. Everything else about data generation (noise magnitudes, gate
constants, scene layout) may change without breaking M5.

Authoritative definition + writer/reader: `src/ai_teleop/data/trajectory.py`.

## Format

**NPZ** (`numpy.savez_compressed`), one per episode, named
`episode_<index:05d>.npz`. Chosen over Parquet because it needs no extra
dependency (pyarrow/pandas live only in the `ml` extra, absent from CI) and the
M5 loader accepts NPZ. Each per-step column is a stacked `(T, …)` array; episode
metadata is a JSON string under the `metadata` key.

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
| `target_hole_index` | which hole was the active target |
| `terminal_reason` | `success` \| `force_abort` \| `timeout` |
| `episode_success` | bool (`terminal_reason == success`) |
| `success_depth`, `lateral_tolerance`, `force_cap` | the terminal-condition thresholds used |

## Terminal conditions (privileged, in the driver)

The driver — not the controller, which stays mode-less — classifies each episode
from privileged geometry:

- **success** — insertion depth past `success_depth` along the bore with lateral
  error under `lateral_tolerance`.
- **force_abort** — wrist force magnitude exceeds `force_cap`.
- **timeout** — step budget reached without success.

**All episodes are kept** (failures included): diverse state coverage helps BC.

## Anti-scope

- **Wrist-camera frames** are not in the schema — image rendering + decimation
  into the corpus is **M7** (Phase 2). Phase-1 training uses F/T + proprioception
  + command only.
