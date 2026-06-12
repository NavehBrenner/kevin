# Milestone 2 — Backbone Controller Online

**Goal**: turn an EE-pose command into compliant arm motion. The arm tracks an externally-supplied 6-DoF pose target each control tick, behaves like a spring on contact (direction-dependent stiffness), and self-protects via a force-cap watchdog plus two lock states. **No assistance logic, no inputs** — just a backbone that any future Δ source (no-assist, expert, learned residual) and command stream (scripted noisy-human, keyboard, vision) can drive.

This milestone takes the static MuJoCo scene produced in M1 and adds the layer that makes the arm *move on command*. By the time we hand off to M3, the only question still open should be "what produces the command stream", not "does the controller track the command".

## Definition of done

By the end of M2 we can:

- Build a `Controller` against a `SimEnv` and feed it a stream of EE-pose commands (position + quaternion + optional grip-force scalar) at the control rate.
- Watch the arm smoothly track a sequence of pose waypoints in the interactive viewer.
- Push the peg into the wall by commanding a target *inside* the wall and observe: the arm moves up to the wall, makes contact, and the peg/wall normal force stays bounded — the arm *gives* laterally and rotationally because of direction-dependent impedance, not because of explicit force clipping.
- Trip the force-cap watchdog by ramping commanded depth and observe an automatic transition to **hold lock** (arm freezes, commands ignored, watchdog state visible to the harness).
- Trigger **park lock** programmatically and watch the arm autonomously return to the M1 home pose, then idle there.
- Run all of the above headless (no viewer) for CI / regression purposes.

## What's in M2

- A `Command` dataclass (target EE pose + optional Δgrip-force, all in world frame).
- Operational-space **differential IK** solver: Jacobian-based, damped least squares, with a null-space posture cost that biases the redundant 7th DoF toward a nominal elbow configuration.
- **Direction-dependent impedance controller** producing joint torques (or joint-velocity / joint-position setpoints, depending on which Panda actuator group we adopt — see *Known unknowns*). Stiffness is high along the local insertion axis (peg long axis) and low laterally and on pitch/roll.
- **Force-cap watchdog** monitoring wrist F/T magnitude with a configurable threshold; trip → `HoldLock`.
- **Lock state machine** inside the controller: `Active` / `HoldLock` / `ParkLock`, with deterministic transitions and a `ParkLock`-to-`Active` path once the home pose is reached.
- A `Controller` facade class wiring the above against a `SimEnv` — exposes `compute(obs, command)` which writes to `data.ctrl` as a side effect, plus `request_hold_lock()`, `request_park_lock()`, `release_lock()`.
- A **dev harness script** that exercises every code path above.

## What's not in M2 — explicit anti-scope

- **The assistance seam** (the Δ-source interface) → M3.
- **Any input strategy** (scripted noisy-human, keyboard, vision) → M3 / M8. M2's harness uses hardcoded waypoints; the production command source comes later.
- **Expert, policy, training, evaluation, data logging** → M4+. Per-step trajectory logging exists only in the dev harness for tuning, not in the controller itself.
- **Trial-level concepts** (success, failure, timeout). The controller is mode-less in the autonomy sense — see `project-scope.md` *Runtime state — two modes only*. Trial bookkeeping lives in the (future) eval harness.
- **Gripper control beyond a baseline grip force**. M2 sets the gripper to a fixed closing force at reset and exposes a `Δgrip-force` channel in `Command`, but the channel is plumbed and clamped only — it is not exercised in the harness. The expert / residual policy will exercise it in M4+.
- **Adaptive / model-predictive impedance**, force-control modes, hybrid position-force schemes. The impedance law is static and direction-dependent — that's enough.
- **Configuration system**. Controller gains, force cap, stiffness profile etc. are constants in code (or constructor args). Hydra / YAML comes in M4+ when there are many configs to manage.

## Build order (estimated effort in parentheses)

### Step 1 — Decide actuator group, baseline grip (~30 min)

The vendored `franka_emika_panda` model from Menagerie exposes multiple actuator groups (typically a torque group and a position group; the gripper has its own actuator). Decide which group M2 drives and verify it works on the M1 scene.

- Default plan: drive the 7 arm joints via the torque (motor) group. This makes impedance control natural (impedance → joint torques directly).
- Fallback: drive joint position targets and tune the per-joint stiffness / damping in the MJCF. Simpler but less faithful to the impedance story.
- Set a baseline gripper closing force at scene reset so the peg stays held throughout M2 trials (this happens automatically via the M1 keyframe today; just confirm).

Output: a short note in `project-wiki/entities/franka-panda.md` recording which actuator group we picked and why.

### Step 2 — `Command` dataclass + control conventions (~30 min)

File: `src/ai_teleop/common/command.py`.

```python
@dataclass(frozen=True)
class Command:
    target_position: np.ndarray   # (3,) world frame, metres
    target_quaternion: np.ndarray # (4,) world frame, (w, x, y, z), unit quat
    delta_grip_force: float = 0.0 # N, additive on top of baseline grip
```

Document in the docstring: same world-frame and quaternion convention as `Observation`. Clamping (Δposition ≤ 2 cm/step, Δorientation ≤ 10°/step, Δgrip-force ≤ 5 N/step) is enforced *inside* the controller — see `project-scope.md` *Residual policy interface*.

### Step 3 — Differential IK with null-space posture cost (~2–3 h)

File: folded into `src/ai_teleop/control/impedance.py`. The DLS pseudoinverse and null-space projector are computed inline there (see Step 4); the standalone `diff_ik.py` was removed as redundant (LAB-19).

Pure-function module — no controller state. Takes:

- Current joint positions (7,)
- Current EE pose (from observation)
- Target EE pose (from command)
- Geometric Jacobian (6 × 7) at the TCP site — computed via `mujoco.mj_jacSite`.
- Damping constant λ.
- Nominal posture `q_nominal` (7,) — typically the M1 home pose.
- Posture gain `k_posture`.

Returns: a target joint-velocity vector `qdot_des` (7,) such that:

1. `qdot_des` moves the EE toward the target pose in one timestep (small-displacement linearisation).
2. Excess null-space DOF is spent driving `q` toward `q_nominal`.

Implementation:

- Compute pose error `e ∈ R^6`: position diff + orientation diff (axis-angle of `q_target · q_current^-1`, MuJoCo's `mju_subQuat` does this).
- DLS solve: `qdot_task = J^T (J J^T + λ^2 I)^-1 e / dt`.
- Null-space projector: `N = I - J^+ J`.
- Posture term: `qdot_post = N · k_posture · (q_nominal - q)`.
- Return `qdot_task + qdot_post`.

Unit test: with a stationary target equal to current pose and `q != q_nominal`, `qdot_des` should be purely in the null space (no EE motion); with non-zero pose error, EE drift should match the error direction. Tolerances generous — this is regression detection, not numerical certification.

### Step 4 — Direction-dependent impedance controller (~2–3 h)

File: `src/ai_teleop/control/impedance.py`.

The impedance law produces a Cartesian wrench from the pose error, then projects that into joint torques via `J^T`:

```
F_des = K · (x_target - x_current) - D · ẋ_current
τ    = J^T · F_des  +  τ_gravity
```

with `K`, `D ∈ R^{6×6}` diagonal in the **TCP frame**, not the world frame — so "stiff along insertion axis" rotates with the gripper. Suggested starting values (tune in Step 8):

| Axis (TCP) | K (N/m or N·m/rad) | D (critical-damped) |
|---|---|---|
| z (insertion / "out of gripper") | 800 | 2·√(M·K) |
| x, y (lateral) | 200 | 2·√(M·K) |
| pitch, roll | 5 | … |
| yaw | 5 | … |

`τ_gravity` is the bias-compensation torque from MuJoCo's `data.qfrc_bias` (already includes gravity + Coriolis), so the impedance is "feels weightless except for the spring".

The module exposes a single function `impedance_torque(model, data, target_pose, K_diag_tcp, D_diag_tcp, q_nominal, posture_gain) -> τ (7,)`. It composes diff-IK (for the posture term and joint-velocity reference) with the Cartesian wrench. Decide during implementation whether to keep diff-IK separate (impedance uses it for posture) or fold posture directly into the impedance null-space — both are reasonable.

Test (manual, in viewer): with a fixed target equal to home pose and no external forces, the arm should hold still. Nudge it by hand in the viewer (push a joint); it should spring back.

### Step 5 — Lock state machine + force-cap watchdog (~1–2 h)

File: `src/ai_teleop/control/lock.py`.

States and transitions:

| State | What it does | Transitions out |
|---|---|---|
| `Active` | Pass through external `Command` to the impedance controller | `request_hold_lock()` → `HoldLock`; `request_park_lock()` → `ParkLock`; force-cap trip → `HoldLock` |
| `HoldLock` | Override the command with "hold current pose" (target = pose at lock time) | `release_lock()` → `Active` |
| `ParkLock` | Override the command with "go to home pose"; once within tolerance, transition to `HoldLock` automatically | Auto: home reached → `HoldLock`; `release_lock()` → `Active` |

The force-cap watchdog runs every controller tick:

```python
if np.linalg.norm(obs.wrist_ft[:3]) > force_cap_n:
    self.transition_to(HoldLock, reason="force_cap_trip")
```

Exposes a small `LockStatus` struct (current state + last-trip reason + sim-time of last transition) so the future eval harness can read it without depending on the controller's internals.

### Step 6 — `Controller` facade (~2 h)

File: `src/ai_teleop/control/backbone.py`.

```python
class Controller:
    def __init__(self, env: SimEnv, *, force_cap_n: float = 30.0,
                 stiffness_tcp: np.ndarray = ..., damping_tcp: np.ndarray = ...,
                 home_pose: np.ndarray | None = None): ...

    def compute(self, obs: Observation, command: Command) -> None:
        """Resolve lock state, clamp the command, run impedance, write data.ctrl.
        Does NOT step the sim — caller invokes env.step()."""

    def request_hold_lock(self) -> None: ...
    def request_park_lock(self) -> None: ...
    def release_lock(self) -> None: ...

    @property
    def status(self) -> LockStatus: ...
```

Notes:

- Constructor caches actuator IDs (analogous to how `SimEnv` caches sensor / joint IDs in M1) so `compute()` does no name lookups in the hot loop.
- Clamping: hard-clip `target_position` to the current EE position ± 2 cm and `target_quaternion` to within 10° of current EE orientation (axis-angle) — see `project-scope.md` *Residual policy interface*. The same clamps will protect M5's learned residual.
- The `Controller` owns the home pose (defaulted from `SimEnv`'s home keyframe via a one-time `mj_resetDataKeyframe` introspection at construction).

### Step 7 — Dev harness script (~1–2 h)

File: `scripts/dev_harness_controller.py`.

Exercises every visible behaviour. Mirrors the structure of `scripts/smoke_test_sim.py`:

1. Build a `SimEnv` (viewer mode) and a `Controller`.
2. **Waypoint phase**: drive the EE through a small box path in space (e.g., 4 corners of a 10 cm square in front of the wall, each held for 1 s). Visual confirmation only — operator watches the viewer.
3. **Compliance phase**: command a target *inside* the wall (e.g., 5 cm behind the wall surface). Expect the arm to make contact, the peg to seat against the wall, and the contact force to plateau well below the force cap because the lateral/rotational axes give.
4. **Force-trip phase**: ramp commanded depth deeper (or stiffen the lateral impedance) until the watchdog trips. Print the `LockStatus` transition; viewer shows the arm freezing.
5. **Park phase**: `release_lock()` → `request_park_lock()` → watch the arm return to the home pose, then auto-transition to `HoldLock`.
6. Run the same script with `render_mode="headless"` (driven by a `--headless` flag) and an assertion checklist — for CI / regression.

The headless variant also saves a tiny CSV of (sim_time, ee_pose, wrist_ft, lock_state) over the run to `outputs/m2_harness_trace.csv` for tuning plots. This logging lives only in the harness — the controller itself does not log.

### Step 8 — Tuning pass (~2–3 h)

The starting gains in Step 4 will be wrong. Plan to:

- Plot `outputs/m2_harness_trace.csv` (matplotlib) — pose-tracking error and contact force over time.
- Adjust stiffness profile until: free-space tracking is crisp (< 2 mm steady-state error), wall contact is gentle (peak force < 15 N at the suggested 5 cm intrusion), and the force cap trips cleanly at the configured 30 N when deliberately overdriven.
- Adjust damping until there's no visible oscillation when the arm hits a waypoint.
- Adjust DLS λ until the arm passes near (but does not lock at) the workspace boundary without exploding joint velocities.

Tuning is iterative and the per-axis numbers will change; the *shape* of the stiffness profile (stiff-along-insertion, soft-laterally) is the design contract and should not change. If it does, that's a design-doc update — go modify `project-scope.md`.

## Acceptance criteria

- `python scripts/dev_harness_controller.py` (viewer mode) plays through all five phases without manual intervention; an operator visually confirms the behaviour described in Step 7.
- `python scripts/dev_harness_controller.py --headless` returns exit 0 and asserts:
  - Steady-state position error at each waypoint < 5 mm (generous; real target is ~2 mm post-tuning).
  - Peak wrist force during the *compliance phase* < `force_cap_n`.
  - Force-cap watchdog trips exactly once during the *force-trip phase*; controller is in `HoldLock` afterward.
  - After `release_lock()` + `request_park_lock()`, controller reaches `HoldLock` at the home pose within a configurable timeout (default 5 s sim time).
- `from ai_teleop.control.backbone import Controller` imports clean (no circular imports against M1 modules).
- `outputs/m2_harness_trace.csv` exists after a headless run.
- Pre-existing M1 smoke test (`scripts/smoke_test_sim.py`) still passes — controller code does not perturb the `SimEnv` contract.

## Total estimated effort

**12–16 hours** including tuning. Spread across 2–3 sessions. Tuning is the long pole and is highly sensitive to MuJoCo's contact-solver settings — budget extra if Phase 3 (compliance) misbehaves.

## Files this milestone touches

```
src/ai_teleop/control/
├── __init__.py             (already exists; populate)
├── impedance.py            (new — also holds the DLS IK + null-space posture term)
├── lock.py                 (new)
└── backbone.py             (new — Controller facade)

src/ai_teleop/common/
└── command.py              (new — Command dataclass)

scripts/
└── dev_harness_controller.py  (new)

tests/
└── test_backbone_smoke.py  (new — headless CI regression for impedance + lock invariants, LAB-21)

outputs/
└── m2_harness_trace.csv    (generated by harness in headless mode; gitignored)
```

`src/ai_teleop/sim/scene.py` is **not** modified — the M1 API contract is sufficient. The controller writes to `env.data.ctrl` directly between `env.get_observation()` and `env.step()`.

## Known unknowns / things to figure out during M2

- **Actuator group on the Menagerie Panda model.** Torque (motor) actuators are the natural fit for impedance, but the vendored model may default to position actuators with per-joint PD baked in. Resolve in Step 1; record the choice in `project-wiki/entities/franka-panda.md`.
- **Whether `mujoco.mj_jacSite` returns the Jacobian at the wrist site or the TCP site we defined.** Either is fine, just need to be consistent. Pick whichever has a single-call API and stick with it.
- **Gravity/Coriolis compensation source**: `data.qfrc_bias` includes both, but verify it's well-defined for the chosen actuator group (it's not always the right offset for position actuators).
- **Direction-of-insertion definition**. The TCP's z-axis (out of gripper) is the natural insertion axis for our peg orientation, but confirm against the M1 keyframe — the home pose's TCP frame matters. If the stiff axis doesn't align with the physical peg axis, the "stiff along insertion" design intent is broken.
- **DLS λ in collision**. The DLS damping that keeps free-space motion stable may be too low when the arm is in contact — the Jacobian effectively rank-drops. May need contact-aware λ; defer to Step 8 tuning if it surfaces.
- **Headless-mode timing**. Without a viewer to throttle, the headless harness runs at sim-speed. Assertion timeouts are in *sim time*, not wall time — be explicit in the code.

## Handoff to Milestone 3

M3 takes the `Controller` produced here and adds:

- A `ScriptedNoisyHuman` stub input strategy that supplies the `Command` stream `Controller.compute` currently gets from a hardcoded waypoint list.
- The assistance seam (Strategy pattern): one interface through which the Δ source plugs in. The same `Controller` is driven either by raw input (no-assist mode, zero Δ) or input + a Δ source, and later input + learned residual (M5). Swapping the Δ source touches nothing upstream or downstream.

M3 does **not** require the M2 impedance gains to be final — only the API contract above (`Controller.compute(obs, command)`, lock requests, `LockStatus`) needs to be stable. Gains can keep being tuned through M3 and M4 as we see real contact patterns.
