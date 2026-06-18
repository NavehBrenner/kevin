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
./scripts/setup.sh
```

That creates the `.venv`, installs the package + dev tooling, enables the git
hooks, and puts a `kvn` launcher on your PATH. Then:

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
- **vision** — live webcam hand tracking via [MediaPipe Hands](https://developers.google.com/mediapipe).
  Move your hand to drive the arm; lift it out of frame to clutch/re-center; make
  a fist to squeeze, open your hand to release. Needs the `vision-input` extra and
  the viewer (no `--headless`).
- **keyboard** — developer fallback, *deferred* (not yet implemented).

### Webcam (vision) setup

Install the extra (adds `opencv-python` + `mediapipe`):

```bash
uv pip install -e ".[dev,vision-input]"
```

**Native Linux / macOS** with a local webcam — just run it (device index `0`):

```bash
kvn episode --input vision           # add --camera N to pick another device
```

**WSL2** — WSL's kernel has no webcam (UVC) driver, so there's no `/dev/video0`.
Instead, stream the camera from Windows and have WSL open it by URL. The repo
ships a tiny streamer for the Windows side: [`scripts/stream_webcam.py`](./scripts/stream_webcam.py).

1. **Find the script from Windows.** From the repo dir in WSL, `explorer.exe .`
   opens File Explorer right at the repo — `scripts/stream_webcam.py` is in there.
   (Equivalently it lives under `\\wsl.localhost\<distro>\<repo-path>\scripts\`.)
   It's a single self-contained file; copy it to Windows if you prefer.
2. **On Windows**, install OpenCV and run it (uses the Windows-side webcam):
   ```powershell
   pip install opencv-python
   python stream_webcam.py            # serves http://0.0.0.0:8080/video
   ```
   The first run pops a **Windows Firewall** prompt — allow it on private
   networks, or WSL can't connect.
3. **Find the Windows host IP from WSL.** With mirrored networking
   (`networkingMode=mirrored` in `.wslconfig`) use `localhost`; with the default
   NAT networking it's the gateway: `ip route show default | awk '{print $3}'`.
4. **In WSL**, point the teleop at the stream:
   ```bash
   kvn episode --input vision --camera http://<windows-host>:8080/video
   ```

The MuJoCo viewer opens via WSLg and your hand drives the arm. Full flag
reference: **[docs/cli.md](./docs/cli.md)**.

## License

To be added (likely MIT).
