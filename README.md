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

## License

To be added (likely MIT).
