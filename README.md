# AI-Assisted Robotic Teleoperation for Precision Insertion

A simulated robotic arm performs peg-in-hole insertions under shared-autonomy control: a human operator provides coarse 6-DoF commands (webcam-tracked hand motion), and a vision-conditioned residual policy issues real-time micro-corrections so insertions reliably succeed even when the human's input is noisy.

Built in [MuJoCo](https://mujoco.org) for the Franka Emika Panda. The residual policy is trained via behavioral cloning against a scripted privileged-info expert. Two configurations — human-only and human + learned residual — are compared head-to-head in a KPI ablation, on a shared always-on impedance backbone.

> **Status**: pre-implementation. Repository is currently being scaffolded. See [project-scope.md](./project-scope.md) for the full project definition and [docs/milestone-1-spec.md](./docs/milestone-1-spec.md) for the current work item.

## Project context

Course project for *Workshop in Autonomous Systems Simulation* (OpenU course 20973, fall 2026). Solo project. Final submission deadline: 2026-08-31.

## Documents

- **[project-scope.md](./project-scope.md)** — full scope, design decisions, KPIs, architecture overview, deferred design questions.
- **[docs/milestone-1-spec.md](./docs/milestone-1-spec.md)** — current milestone: simulation environment online.

## Quick start

Requires [uv](https://github.com/astral-sh/uv) (Python 3.12). **After cloning, run
the one-time setup** from this directory:

```bash
./scripts/setup.sh          # everything needed to run the project
./scripts/setup.sh --dev    # the above plus dev tooling (pytest/ruff/mypy) + docs
```

On Windows (e.g. a run-only copy for the native-camera interactive viewer), use the
PowerShell sibling instead:

```powershell
.\scripts\setup.ps1         # -Dev for the dev tooling + docs
```

That creates the `.venv`, installs the package + extras, enables the git hooks,
and puts a `kvn` launcher on your PATH. The default installs the full **runtime**
stack (policy train/eval, stereo webcam teleop via `stereo-input`, recording, scene
generation); `-D`/`--dev` adds the dev tooling and docs deliverables. Then:

```bash
kvn                       # list every command
kvn smoke --no-viewer     # M1 scene smoke test, headless
kvn sim --seed 7          # generate and view a procedural wall
kvn check                 # the full lint + typecheck + test gate
```

`kvn` (pronounced *"Kevin"*) is the project's command-line front door — one entry
point for the whole workflow instead of `uv run python scripts/...`. `kvn` (or
`kvn --help`) lists commands; `kvn <command> --help` shows a command's flags. Full
reference: **[docs/cli.md](./docs/cli.md)**.

> Don't want the PATH launcher? Everything also works as `uv run kvn <command>`
> straight after `uv pip install -e ".[dev]"`.

## Input strategies

The operator's coarse command source is swappable behind a common seam; pick one
at runtime with `kvn episode --input {scripted,vision}` (default `scripted`):

- **scripted** — a deterministic, seedable "noisy human". No hardware; used for
  data generation and repeatable KPI benchmarking. This is the default.
- **vision** — live **two-webcam stereo** hand tracking via the standalone
  [stereohand](https://github.com/NavehBrenner/stereohand) package: metric 3D hand
  pose + 6-DoF mirroring. Move your hand to drive the arm; lift it out of frame (or
  hold an open palm still for 3 s) to re-anchor; make a fist to squeeze, open your
  hand to release. Needs the `stereo-input` extra, a one-time stereo calibration,
  and the viewer (no `--headless`).

A keyboard fallback was scoped in M8 and **dropped** — `scripted` covers repeatable
benchmarking and `vision` covers the live demo, so nothing needed it.

### Stereo (vision) setup

Install the extra (pulls `stereohand`, which brings its own OpenCV + MediaPipe):

```bash
uv pip install -e ".[dev,stereo-input]"
```

You need two rigidly co-mounted webcams and a one-time ChArUco stereo calibration
(`stereo_calib.json`) — see the [stereohand](https://github.com/NavehBrenner/stereohand)
README for the calibration walkthrough. Then:

```bash
kvn episode --input vision --stereo-calib stereo_calib.json --left 0 --right 2
```

**WSL2** — WSL's kernel has no webcam (UVC) driver, so there's no `/dev/video*`. Stream
both cameras from Windows and pass their URLs to `--left` / `--right`, e.g.
`--left "http://<windows-host>:8080/0" --right "http://<windows-host>:8080/1"`. The
Windows-side camera bridge (`stream_webcams.py`) and a full step-by-step WSL walkthrough
live in the stereohand project.

## License

To be added (likely MIT).
