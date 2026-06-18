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
The fix: run the camera on **Windows**, have it serve an [MJPEG](https://en.wikipedia.org/wiki/Motion_JPEG)
video stream over HTTP, and have the WSL side open that stream by URL. The repo
ships the Windows-side streamer — [`scripts/stream_webcam.py`](./scripts/stream_webcam.py),
a single self-contained file you run with Windows' Python. You do **not** edit
it; you just run it.

Do the steps in order. Steps 1–2 run **in your WSL terminal**; step 3 runs **in
a Windows PowerShell window**; steps 4–5 run **back in your WSL terminal**.

1. **(WSL)** Install the extra and copy the streamer onto your Windows Desktop
   (so it's easy to reach from PowerShell). Run from the repo root:
   ```bash
   uv pip install -e ".[dev,vision-input]"
   cp scripts/stream_webcam.py "$(wslpath "$(powershell.exe -NoProfile -Command \
       '[Environment]::GetFolderPath("Desktop")' | tr -d '\r')")/"
   ```
   The `cp` lands `stream_webcam.py` on your Windows Desktop. (Prefer to do it by
   hand? Run `explorer.exe .` to open the repo in File Explorer and drag
   `scripts\stream_webcam.py` to your Desktop.)

2. **(WSL)** Install Python on Windows if you don't have it — `winget install
   Python.Python.3.12` — then skip to step 3. (Already have Windows Python? Skip
   this.)

3. **(Windows PowerShell)** Open PowerShell (Start → type "PowerShell" → Enter),
   then install OpenCV and start the streamer:
   ```powershell
   pip install opencv-python
   python "$env:USERPROFILE\Desktop\stream_webcam.py"
   ```
   You should see exactly this line, and the window then stays open (that's
   normal — it's serving; leave it running):
   ```
   serving camera 0 at http://0.0.0.0:8080/video  (Ctrl-C to stop)
   ```
   The first run pops a **Windows Firewall** prompt — tick **Private networks**
   and click *Allow access*, or WSL won't be able to connect. (Webcam light on?
   Good. If it says it can't open camera 0, append `--camera 1` and retry.)

4. **(WSL)** Compute the Windows host's address and sanity-check the stream
   *before* launching the sim. Run from the WSL terminal:
   ```bash
   WIN_HOST=$(ip route show default | awk '{print $3}')   # e.g. 172.20.16.1
   echo "$WIN_HOST"                                        # prints that address
   curl -sI "http://$WIN_HOST:8080/video" | head -1        # expect: HTTP/1.0 200 OK
   ```
   `WIN_HOST` is the address WSL reaches Windows on. WSL is a lightweight VM with
   its own network, so Windows isn't `localhost` — it's the NAT gateway, which
   that `ip route` line extracts. The `curl` returning `200 OK` confirms the
   firewall is open and the stream is live. (If you've enabled *mirrored*
   networking — `networkingMode=mirrored` in `%UserProfile%\.wslconfig` — then
   Windows *is* reachable as `localhost`; set `WIN_HOST=localhost` instead.)

5. **(WSL)** Launch the teleop against that stream (reuses `$WIN_HOST` from step 4):
   ```bash
   kvn episode --input vision --camera "http://$WIN_HOST:8080/video" --max-steps 0
   ```
   `--max-steps 0` runs with no step limit (free-play) until you close the viewer
   or `Ctrl-C`; drop it for the default fixed-length episode. A second window
   shows the camera feed with the tracked hand and the controls.

The MuJoCo viewer opens via WSLg and your hand drives the arm. When you're done,
`Ctrl-C` the PowerShell streamer window to release the camera. Full flag
reference: **[docs/cli.md](./docs/cli.md)**.

> **`curl` didn't return `200 OK`?** Almost always one of: the firewall prompt
> got dismissed (re-run the streamer and allow it, or add an inbound rule for
> port 8080), or `$WIN_HOST` is wrong (try `localhost`). Fix that before step 5 —
> the sim can only connect once `curl` succeeds.

## License

To be added (likely MIT).
