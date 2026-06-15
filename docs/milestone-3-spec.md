# Milestone 3 — Assistance Seam + Scripted Input Online

**Goal**: close the loop end-to-end *without a human in it*. M2 left the
`Controller` driven by a hardcoded waypoint list; M3 replaces that with two new
layers behind clean interfaces — a **command source** (a stub `ScriptedNoisyHuman`
input strategy) and the **assistance seam** (the one interface through which any
correction-Δ source plugs in). With both wired, the full stack runs a complete
episode on its own: scripted input → seam → `Controller` → `SimEnv`.

This milestone is deliberately about *structure*, not behaviour. The arm does not
yet insert anything reliably — the scripted human is dumb and the only Δ source we
ship is the zero-Δ `NoAssist`. What M3 buys is the **dependency-inversion seam**
(an architectural pillar from `project-scope.md`): by the time we hand off to M4,
dropping in the analytical expert — and later M5's learned residual — must touch
*nothing* upstream (the input) or downstream (the controller). M3 proves that
property with a dummy Δ source before any real one exists.

## Definition of done

By the end of M3 we can:

- Drive the M2 `Controller` from a `ScriptedNoisyHuman` instead of a hardcoded
  waypoint list — the arm tracks a noisy "go to the hole vicinity" command stream.
- Run a full episode-length loop **end-to-end in no-assist mode** (the `NoAssist`
  provider supplies zero Δ) through the seam, headless and in the viewer, with no
  manual intervention.
- **Swap the Δ source for a dummy provider** that returns a fixed non-zero Δ and
  observe the combined command reach the controller — *without editing the input
  strategy or the controller*. This is the seam's acceptance test.
- See the whole composition (`InputStrategy → AssistProvider → apply_delta →
  Controller.compute → env.step`) live in one runner, reusable as a library
  function for M4's data-generation rollout.

## What's in M3

- **`domain/` interfaces** (`Protocol`s — the abstractions every concrete layer
  depends on, per Dependency Inversion):
  - `InputStrategy` — produces the base per-tick `Command` from an `Observation`.
  - `AssistProvider` — *the seam*. Given the `Observation` and the base `Command`,
    returns a correction `Delta`.
- **`Delta` dataclass** + a pure **`apply_delta(command, delta) -> Command`**
  combine function (and the per-step Δ bounds the residual interface promises).
- **`NoAssist`** — the default zero-Δ `AssistProvider`; recovers no-assist mode
  for free.
- **`ScriptedNoisyHuman`** — a *stub* `InputStrategy`: command the target pose
  plus simple additive noise. Just enough to drive the system.
- **An end-to-end runner** (`scripts/run_episode.py`) composing the stack, plus a
  reusable `run_episode(...)` function M4 will call for data generation.
- **A seam regression test** proving (a) no-assist runs end-to-end and (b) an
  injected dummy Δ reaches the controller with zero upstream/downstream change.

## What's not in M3 — explicit anti-scope

- **A realistic noise model.** `ScriptedNoisyHuman` here is a stub — per-axis
  Gaussian jitter on a fixed target. The realistic model (drift, intent phases,
  release/withdraw, per-episode noise patterns) is refined in **M4**.
- **The expert.** M3 ships only `NoAssist` and a throwaway dummy Δ source. The
  analytical privileged-info expert is **M4**; it will implement the *same*
  `AssistProvider` interface defined here.
- **The learned policy** (`AssistProvider` via a BC-trained network) → **M5+**.
- **Real input devices** (MediaPipe vision, keyboard) → **M8**. The
  `InputStrategy` interface is designed to admit them, but M3 implements only the
  scripted stub.
- **Data logging / trajectory files.** M3's runner steps the loop and may print,
  but structured per-step logging to disk is **M4**. (The runner is *structured*
  so M4 can bolt logging on without reshaping it.)
- **Trial-level concepts** (success, failure, timeout, KPIs). Still out — the
  controller stays mode-less; trial bookkeeping lives in the future eval harness
  (M6). M3's "episode terminates" is just a step budget.
- **Configuration system** (Hydra/YAML). Targets, noise σ, seed, step budget are
  constructor args / CLI flags. Config consolidation comes in M4+.

## The assistance seam — design

The seam is a **Strategy pattern** realized through two `Protocol`s in `domain/`,
plus a pure combine function. Nothing in `domain/` imports `ai_teleop.sim` — the
interfaces depend only on the `Command` / `Observation` dataclasses in `common/`,
which is what keeps every layer swappable.

### `InputStrategy` (command source)

```python
class InputStrategy(Protocol):
    def get_command(self, observation: Observation) -> Command:
        """The coarse per-tick EE-pose setpoint, before any assistance."""
```

Concrete implementations live in `input/`: `ScriptedNoisyHuman` (M3),
`VisionInput` / `KeyboardInput` (M8).

### `AssistProvider` (the seam)

```python
class AssistProvider(Protocol):
    def get_delta(self, observation: Observation, command: Command) -> Delta:
        """The correction to add on top of the input's command. Zero ⇒ no assist."""
```

Concrete implementations: `NoAssist` (M3, in `domain/`), the analytical expert
(M4, in `expert/`), the learned residual (M5, in `policy/`). All three share this
exact signature — that symmetry is what makes behavioral cloning clean (the policy
mimics the expert's `Delta` output) and what lets the runner stay source-agnostic.

### `Delta` + `apply_delta`

```python
@dataclass(frozen=True)
class Delta:
    delta_position: np.ndarray     # (3,) world frame, metres
    delta_orientation: np.ndarray  # (3,) axis-angle, radians
    delta_grip_force: float = 0.0  # newtons

ZERO_DELTA: Delta = Delta(np.zeros(3), np.zeros(3), 0.0)
```

The combine step, applied each tick between the input and the controller:

```
base_command = input_strategy.get_command(observation)
delta         = assist.get_delta(observation, base_command)
command       = apply_delta(base_command, delta)   # → Controller.compute
```

`apply_delta` semantics:

- `target_position` ← `base.target_position + delta.delta_position`.
- `target_quaternion` ← `delta.delta_orientation` (as a rotation) composed onto
  `base.target_quaternion`, renormalized. Use MuJoCo's quaternion helpers
  (`mju_axisAngle2Quat` + `mju_mulQuat`) for consistency with the rest of the
  stack — no hand-rolled quaternion math.
- `delta_grip_force` ← `base.delta_grip_force + delta.delta_grip_force`.

### Two clamps, two purposes

`project-scope.md` *Residual policy interface* promises the Δ source is
**safe-by-construction**. We honour that with a per-step Δ clamp *inside the seam*,
distinct from the M2 controller's command clamp:

1. **Δ clamp (seam).** `apply_delta` clamps the incoming `Delta` to the
   residual-interface bounds before applying it: `|Δposition| ≤ 2 cm`,
   `|Δorientation| ≤ 10°`, `|Δgrip| ≤ 5 N` per step. A misbehaving provider can
   never inject more than one bounded nudge — this is the contract M5's network
   relies on.
2. **Command clamp (controller, M2, unchanged).** `Controller.compute` then clamps
   the *combined* command to within 2 cm / 10° of the **current EE pose**. This
   bounds total per-step EE motion regardless of how input and Δ stack up.

The two are independent and both stay in force; M3 adds (1) and leaves (2) exactly
as M2 shipped it.

## Build order (estimated effort in parentheses)

### Step 1 — Domain seam: interfaces + `Delta` + `apply_delta` + `NoAssist` (~2–3 h)

Files: `src/ai_teleop/domain/interfaces.py`, `src/ai_teleop/domain/delta.py`
(re-exported from `domain/__init__.py`).

- Define `InputStrategy` and `AssistProvider` as `typing.Protocol`s
  (`@runtime_checkable` so the dummy-source test can assert conformance).
- Define `Delta` (frozen dataclass) + `ZERO_DELTA`.
- Implement `clamp_delta(delta) -> Delta` (the per-step bounds) and
  `apply_delta(command, delta) -> Command` (clamp, then add/compose).
- Implement `NoAssist` (`get_delta` returns `ZERO_DELTA`).

Unit tests (`tests/test_seam.py`):
- `apply_delta(command, ZERO_DELTA) == command` (component-wise allclose).
- A Δ exceeding the bounds comes back clamped to exactly the bound.
- A small Δorientation composes onto the base quaternion and the result stays unit
  norm.
- `NoAssist()` satisfies `isinstance(..., AssistProvider)` (runtime-checkable).

### Step 2 — `ScriptedNoisyHuman` stub input strategy (~1.5–2 h)

File: `src/ai_teleop/input/scripted_noisy_human.py`.

```python
class ScriptedNoisyHuman:
    def __init__(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        *,
        position_noise_std: float = 0.005,   # 5 mm
        orientation_noise_std: float = 0.02,  # ~1.1°
        seed: int = 0,
    ) -> None: ...

    def get_command(self, observation: Observation) -> Command: ...
```

- Each tick, command the *target* pose perturbed by zero-mean Gaussian noise
  (per-axis on position; a small random axis-angle on orientation), drawn from a
  seeded `np.random.Generator`.
- It commands the full target every tick; the M2 controller's 2 cm/10° command
  clamp turns that into a smooth bounded approach — so the "coarse human" need not
  do any trajectory shaping. This is the intended division of labour.
- No drift, no phases, no release behaviour — those are M4.

Unit tests: with `seed` fixed the command stream is reproducible; with noise std 0
the command equals the target exactly; mean over many draws ≈ target.

### Step 3 — End-to-end runner (~1.5–2 h)

File: `scripts/run_episode.py`, exposing a reusable
`run_episode(environment, controller, input_strategy, assist, *, max_steps, render)`
that M4 will import for data generation.

```
observation = environment.reset()
for _ in range(max_steps):
    base_command = input_strategy.get_command(observation)
    delta         = assist.get_delta(observation, base_command)
    command       = apply_delta(base_command, delta)
    controller.compute(observation, command)
    environment.step()
    observation = environment.get_observation()
```

- The script's `__main__` builds a `SimEnv` (viewer or `--headless`), a
  `Controller`, a `ScriptedNoisyHuman` aimed at the trial's target hole (read the
  target hole pose from `observation` — `target_hole_index` + `hole_poses`), and a
  `NoAssist` provider, then calls `run_episode`.
- CLI flags mirror the M2 harness: `--headless`, `--seed`, `--max-steps`.
- Keep `run_episode` free of logging/printing side effects beyond a return value
  (e.g. final observation + lock status) so M4 can wrap it; put any console output
  in `__main__`.

### Step 4 — Seam injection test + acceptance (~1.5–2 h)

The M3 acceptance lives here (`tests/test_seam.py` continued, + a headless
end-to-end test sibling to `test_backbone_smoke.py`).

- **No-assist e2e (headless, fast):** run `run_episode` for a few hundred steps
  with `NoAssist`; assert it completes, the controller stays in a sane lock state,
  and the EE moves toward the target hole (looser than any insertion claim — this
  is plumbing, not performance).
- **Dummy-Δ injection (the key test):** define a throwaway
  `class _FixedDelta: def get_delta(self, obs, cmd): return Delta(...)` returning a
  fixed non-zero Δ. Run the *same* `run_episode` with it swapped in for `NoAssist`
  — no other change — and assert the controller received `apply_delta(base, Δ)`
  (capture the command handed to `Controller.compute`, e.g. via a spy/wrapper, and
  check it equals the combined-then-clamped value). This demonstrates the seam:
  the input strategy and controller are untouched across the swap.

## Acceptance criteria

- `uv run python scripts/run_episode.py` (viewer) runs a full episode with the
  scripted human + `NoAssist`; the arm tracks the noisy target toward the hole and
  the run ends cleanly at the step budget.
- `uv run python scripts/run_episode.py --headless` exits 0.
- `uv run poe check` is green, including `tests/test_seam.py` and the new headless
  end-to-end test:
  - `apply_delta(command, ZERO_DELTA)` round-trips the command.
  - Δ exceeding bounds is clamped to the bound.
  - The dummy-Δ source swaps in for `NoAssist` with **no edit** to
    `ScriptedNoisyHuman` or `Controller`, and the combined Δ reaches the controller.
- `from ai_teleop.domain import InputStrategy, AssistProvider, Delta, apply_delta, NoAssist`
  imports clean; `domain/` has no `ai_teleop.sim` import (enforce by inspection /
  an import test).
- The M2 dev harness (`scripts/dev_harness_controller.py --headless`) and the M1
  smoke test still pass — M3 adds layers on top and modifies neither contract.

## Total estimated effort

**6–10 hours**, 1–2 sessions. Mostly interface design + a little quaternion
plumbing in `apply_delta`. No tuning long-pole (unlike M2) — the scripted human is
intentionally crude and the seam is pure logic.

## Files this milestone touches

```
src/ai_teleop/domain/
├── __init__.py                 (populate — re-export the interfaces + Delta/NoAssist)
├── interfaces.py               (new — InputStrategy, AssistProvider Protocols)
└── delta.py                    (new — Delta, ZERO_DELTA, clamp_delta, apply_delta, NoAssist)

src/ai_teleop/input/
├── __init__.py                 (already exists; populate)
└── scripted_noisy_human.py     (new — stub InputStrategy)

scripts/
└── run_episode.py              (new — composition runner + reusable run_episode())

tests/
└── test_seam.py                (new — seam unit tests + dummy-Δ injection + headless e2e)
```

`src/ai_teleop/control/` and `src/ai_teleop/sim/` are **not** modified — M2's
`Controller.compute(observation, command)` contract and M1's `SimEnv` contract are
both sufficient. The seam composes *around* the controller, not inside it.

## Known unknowns / things to figure out during M3

- **Where the seam composes.** Decision: in the runner (`run_episode`), not inside
  `Controller`. The controller stays a pure command-tracker; the seam is a layer
  above it. This keeps the controller↔harness decoupling intact (see
  `project-scope.md` *Runtime state*). Confirm nothing about `Controller` needs to
  know a Δ source exists.
- **Quaternion composition order in `apply_delta`.** Δorientation is a *body-frame*
  vs *world-frame* nudge — pick world-frame (left-multiply) for consistency with
  how `Command.target_quaternion` is expressed, and unit-test that a known small
  rotation lands where expected. Get this wrong and the expert's orientation
  corrections (M4) will fight the controller.
- **Whether `ScriptedNoisyHuman` should ramp toward the target or command it
  outright.** Plan: command it outright and lean on the controller's per-step
  clamp (above). If the approach looks too aggressive in the viewer, add a simple
  first-order target filter — but only if needed; the clamp likely suffices.
- **Runtime-checkable Protocols + numpy.** `@runtime_checkable` only checks method
  presence, not signatures — fine for the conformance assertion, but don't lean on
  it for correctness. mypy structural typing is the real check.

## Handoff to Milestone 4

M4 takes the seam and runner produced here and adds:

- The **analytical privileged-info expert** as an `AssistProvider` (in `expert/`),
  consuming the privileged true peg/hole poses already present in `Observation`
  and emitting the corrective `Delta` — slotting into the exact seam M3 defined,
  no runner change beyond passing it in place of `NoAssist`.
- A **data-generation rollout** that wraps `run_episode` to log structured per-step
  rows (sensors, base command, expert Δ, privileged ground truth, success flags).
- The **realistic** `ScriptedNoisyHuman` noise model + coverage randomization
  (target/distractor holes, initial peg offset, per-episode noise pattern).

M4 does **not** require M3's scripted human to be realistic — only the
`InputStrategy` / `AssistProvider` / `Delta` contracts and `run_episode`'s shape
need to be stable. Those are the deliverable of this milestone.
