# kevin Project Operations

## What's here

The implementation of the AI-assisted teleoperation project, plus its authoritative
specs:

- `project-scope.md` — the authoritative project definition (scope, KPIs, architecture,
  deferred design). **Source of truth for design decisions.** If a decision changes,
  update it here rather than letting code and spec drift.
- `docs/` — milestone specs (currently `milestone-1-spec.md`).
- `src/ai_teleop/` — the package. See its `__init__.py` docstring for the module map.

## Python environment — uv

Managed with [uv](https://github.com/astral-sh/uv). The virtualenv lives at `.venv`
(Python 3.12). **First-time setup after cloning:** `./scripts/setup.sh` — creates
the venv, installs `.[dev]`, enables the git hooks, and installs the `kvn` launcher
on PATH. The individual steps, from this directory:

- `uv venv` — create the venv.
- `uv pip install -e ".[dev]"` — package + dev tooling.
- `uv pip install -e ".[dev,ml,vision-input]"` — full stack incl. torch + mediapipe.
- `uv run python scripts/<script>.py` — run inside the venv.

### Project CLI — `kvn`

`kvn` (pronounced *"Kevin"*) is the project's command-line front door — one entry
point for the whole workflow instead of `uv run python scripts/...`. It's a thin
dispatcher (`src/ai_teleop/cli.py`): simulation/data commands run the matching
script in `scripts/`; dev-gate commands delegate to the poe tasks below. Full
reference: `docs/cli.md`.

| Command | Does |
|---|---|
| `uv run kvn` | list every command |
| `uv run kvn sim [args]` | generate / view a procedural wall (`view_generated_wall.py`) |
| `uv run kvn smoke [args]` | M1 scene smoke test (`smoke_test_sim.py`) |
| `uv run kvn episode [args]` | one end-to-end no-assist episode (`run_episode.py`) |
| `uv run kvn harness [args]` | M2 controller dev harness (`dev_harness_controller.py`) |
| `uv run kvn gen [args]` | generate the BC dataset (`generate_dataset.py`) |
| `uv run kvn check` | the full gate (delegates to `poe check`) |

`kvn <command> --help` forwards to that script's own `argparse`, so it always shows
the authoritative flags. The console script is registered via `[project.scripts]`;
if the relocated-`.venv` stale-shebang issue (below) breaks it, use
`uv run poe cli <command>` or `uv run python -m ai_teleop.cli <command>`.

### Task runner — poe (the dev gate `kvn` delegates to)

Common dev actions are defined as [poethepoet](https://poethepoet.natn.io/)
tasks in `pyproject.toml` (`[tool.poe.tasks]`). `kvn`'s dev-gate commands delegate
here; you can also run them directly with `uv run poe <task>`:

| Command | Does |
|---|---|
| `uv run poe fmt` | `ruff format` the code |
| `uv run poe lint` | `ruff check` |
| `uv run poe typecheck` | `mypy` |
| `uv run poe test` | `pytest` |
| `uv run poe check` | lint + typecheck + test (the full gate, same as CI) |
| `uv run poe sim [args]` | launch the wall viewer (e.g. `uv run poe sim --seed 7`) |
| `uv run poe smoke` | run the sim smoke test |
| `uv run poe cli [args]` | reach the `kvn` CLI without the console script (relocated-venv-safe) |

Prefer these over remembering the underlying commands. Add a new task here
rather than scattering one-off invocations. (mypy/pytest run via `python -m`
inside the tasks because the relocated `.venv` has stale console-script
shebangs — see the hooks note below.)

### Hooks and CI

Local git hooks live in `.githooks/` (version-controlled). Activate them once
per clone — they are *not* enabled automatically:

```
git config core.hooksPath .githooks
```

- **pre-commit** — `ruff format` on staged Python, re-staging what it changed.
- **pre-push** — blocks the push unless `mypy` passes.

CI (`.github/workflows/ci.yml`) gates every **PR into `master`**: it installs
`.[dev,scenegen,ml]` (the `ml` extra brings `torch`, which the M5 dataset-loader
tests import) and must pass `mypy` and `pytest`. The hooks run the same
tools via `uv run` (mypy as `uv run python -m mypy`, since the relocated
`.venv` has stale console-script shebangs).

## Ad-hoc debugging / experiments

When exploring or tuning, **write the snippet as a file under `scripts/dev/`** and run it
with `uv run python scripts/dev/<name>.py`. Do *not* pass code with `python -c "..."` heredocs.
Reasons: the user can read, edit, and re-run the script themselves; the script is reviewable in
the diff; iteration is faster (edit-and-rerun instead of rewriting the whole heredoc); and the
file becomes a permanent artifact (or a deliberately-deleted one) instead of vanishing into
shell history.

`scripts/dev/` is for one-off probes and tuning sweeps. Production-ish runnable scripts
(smoke tests, dev harnesses, data-generation drivers) live directly in `scripts/`.
