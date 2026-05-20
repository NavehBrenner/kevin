# Milestone 1 — Sim Environment Online

**Goal**: stand up a working MuJoCo scene with all the physical pieces wired in (Panda + chamfered wall with holes + pre-grasped peg + wrist camera + wrist light + F/T sensor), exposed through a clean Python API. **No control logic, no AI** — just a static scene plus a confirmation that all sensors and rendering work.

This milestone is intentionally narrow. By the time we hand off to Milestone 2, the only question we should still be solving is "what does the controller do with this scene", not "does the scene work at all."

## Definition of done

By the end of M1 we can:

- Load the scene in either interactive viewer or headless mode.
- Move the arm by directly setting joint angles (smoke testing only — no controller yet).
- Get a wrist-camera frame as a NumPy array of shape (H, W, 3) uint8.
- Get a wrist force/torque reading as a 6-vector.
- Get full state (joint pos/vel, EE pose, peg pose, all hole poses).
- Reset the scene to a clean state, with the peg pre-grasped at the gripper at a known initial pose.

## What's in M1

- Scene assets (MJCF files for Panda + wall + holes + peg + lights + wrist cam).
- A `SimEnv` wrapper class with reset / step / sensor / render API.
- Both interactive viewer and headless rendering paths.
- A smoke-test script that loads the scene, prints sensor values, saves a wrist-cam PNG, and (optionally) opens the viewer for manual inspection.

## What's not in M1 — explicit anti-scope

- Any controller logic: IK, impedance, force capping, spiral search → deferred to M2.
- Any input strategy (scripted noisy-human, keyboard, vision) → deferred to M3.
- Any expert, policy, training, or evaluation code → deferred to M4+.
- Domain randomization or themed lighting → stretch goals after Phase 1.
- Multiple peg / hole geometries → stretch goals after Phase 1.
- Full configuration system (Hydra etc.) → deferred. Smoke test reads from constants in code.

## Build order (estimated effort in parentheses)

### Step 1 — Local Python environment (~30 min)

- Confirm Python ≥ 3.10 available.
- Create a virtual environment in the repo root (`python -m venv .venv`).
- Activate and `pip install -e ".[dev]"` from the repo root.
- Sanity check: `python -c "import mujoco; print(mujoco.__version__)"`.

### Step 2 — Acquire Panda assets (~30 min)

- Clone `mujoco_menagerie` (https://github.com/google-deepmind/mujoco_menagerie).
- Copy the `franka_emika_panda/` directory into `assets/mjcf/menagerie_panda/`. This vendors the model so the repo stays self-contained.
- Confirm the model loads in isolation: `python -m mujoco.viewer --mjcf assets/mjcf/menagerie_panda/panda.xml`.

### Step 3 — Build the wall + holes scene (~2-3 h)

File: `assets/mjcf/wall_with_holes.xml`.

- A vertical wall in front of the robot — either a thin box geometry or a custom mesh.
- 3–5 holes at known fixed positions on the wall surface.
- Each hole has a **chamfered rim**: model this as a slightly larger cone-shaped opening that narrows to the target diameter. The chamfer angle is a parameter we'll want to expose (start with a generous chamfer, e.g., 30°).
- Default geometry: holes are 10 mm diameter at the narrow end; chamfer rim 14 mm.
- Simple visual material (one color for the wall, slightly darker for the hole interiors so they're visually distinguishable in the wrist camera).

Open question to resolve during this step: can we use a single mesh with subtracted cylinder holes, or do we need to compose this from primitives? Both work in MuJoCo. Try primitives first (boolean-subtraction is more painful).

### Step 4 — Build the peg (~1 h)

File: `assets/mjcf/peg.xml`.

- Cylindrical peg, 8 mm diameter, 60 mm long.
- Reasonable mass (~30 g) and friction parameters.
- This file defines the peg as a stand-alone body — it gets attached to the gripper at scene reset.

### Step 5 — Combine into the full scene (~1-2 h)

File: `assets/mjcf/full_scene.xml`.

- `<include>` the menagerie Panda model, the wall, and the peg.
- Position the Panda base at world origin (z up), wall ~50 cm in front of the robot.
- Add a soft ambient light (uniform, low intensity) plus a wrist-mounted spot light pointing forward from the gripper.
- Add a wrist-mounted RGB camera with sensible resolution (start with 128×128, 60° FOV) attached to the same body as the gripper.
- Confirm the model still loads in `mujoco.viewer`.

### Step 6 — `SimEnv` wrapper (~2-3 h)

File: `src/ai_teleop/sim/scene.py`.

Class `SimEnv` with at least:

```python
class SimEnv:
    def __init__(self, scene_path: str, render_mode: Literal["viewer", "headless"]):
        ...

    def reset(self) -> Observation:
        """Reset the sim: arm to base pose, peg pre-grasped at known initial pose,
        target hole chosen, RNG seeded. Returns the initial observation."""
        ...

    def step(self) -> None:
        """Step the physics by one timestep. (No command input yet — M2 adds that.)"""
        ...

    def get_observation(self) -> Observation:
        ...

    def render_wrist_camera(self) -> np.ndarray:
        """Return a (H, W, 3) uint8 frame from the wrist camera."""
        ...

    def close(self) -> None:
        ...
```

Plus a small dataclass in `src/ai_teleop/common/observation.py`:

```python
@dataclass
class Observation:
    joint_positions: np.ndarray      # (7,)
    joint_velocities: np.ndarray     # (7,)
    ee_pose: np.ndarray              # (7,) — position (3) + quaternion (4)
    peg_pose: np.ndarray             # (7,) — privileged ground truth
    hole_poses: np.ndarray           # (N, 7) — privileged ground truth, all holes
    target_hole_index: int           # which hole is "the target" for this trial
    wrist_ft: np.ndarray             # (6,) — Fx Fy Fz Mx My Mz
    sim_time: float
```

Implementation notes:

- Pre-grasp the peg by setting joint positions on a "peg-attachment" weld constraint at reset, or simply setting the peg's free-joint position to match the gripper TCP and closing the gripper enough to grip it. Try the weld approach first — it's deterministic.
- Headless render path uses `mujoco.Renderer`; viewer path uses `mujoco.viewer.launch_passive` (or similar). Same model object underneath, just different rendering surfaces.

### Step 7 — Smoke-test script (~1 h)

File: `scripts/smoke_test_sim.py`.

- Loads the scene in headless mode.
- Calls `reset()`, then steps the sim 100 times.
- Prints the observation each 10 steps.
- Saves a wrist-camera frame as `outputs/m1_wrist_cam.png` for inspection.
- Then loads the scene in viewer mode and runs an idle loop so the user can inspect it manually.

### Step 8 — Verification (~30 min)

Run the smoke test and check the acceptance criteria below.

## Acceptance criteria

- `pip install -e .` succeeds without manual intervention.
- `python -c "from ai_teleop.sim.scene import SimEnv"` works.
- `python scripts/smoke_test_sim.py` runs without errors.
- The saved wrist-camera PNG visually shows the wall + holes + peg.
- The viewer window opens, displays the scene, and supports mouse-driven camera rotation.
- F/T values returned by `get_observation()` are non-zero and roughly match peg weight × gravity (~0.3 N along whichever axis gravity acts on the peg).
- Hole positions reported by `get_observation()` match the positions defined in the MJCF.

## Total estimated effort

**8–10 hours** for someone unfamiliar with MuJoCo, less if comfortable. Spread across 1–2 working sessions.

## Files this milestone touches

```
assets/mjcf/
├── menagerie_panda/        (vendored — copied from external repo)
├── wall_with_holes.xml     (new)
├── peg.xml                 (new)
└── full_scene.xml          (new)

src/ai_teleop/sim/
├── __init__.py             (already exists; populate)
└── scene.py                (new — SimEnv class)

src/ai_teleop/common/
├── __init__.py             (already exists; populate)
└── observation.py          (new — Observation dataclass)

scripts/
└── smoke_test_sim.py       (new)

outputs/
└── m1_wrist_cam.png        (generated by smoke test; gitignored)
```

## Known unknowns / things to figure out during M1

- Exact MuJoCo Python API for offscreen rendering (`mujoco.Renderer`).
- Cleanest way to "pre-grasp" the peg at reset — weld equality constraint vs joint-setting.
- Whether the wall + holes geometry is cleanest as a single mesh, composed primitives, or via boolean subtraction in a CAD-like tool. Try primitives first.
- Whether the menagerie Panda model already exposes a wrist F/T sensor or if we add one in the scene XML.

## Handoff to Milestone 2

When M1 lands, M2 begins. M2 takes the `SimEnv` produced here and adds:

- Operational-space differential IK on top of the joint actuators.
- Direction-dependent impedance control.
- Force-cap watchdog and hold-lock / park-lock behavior.

M2 does *not* require M1 to be perfect — only the API contract above (reset / step / observation / render) needs to be stable. The MJCF can keep iterating.
