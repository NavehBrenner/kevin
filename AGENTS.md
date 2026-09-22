# kevin Project Operations

## What's here

The implementation of the AI-assisted teleoperation project, plus its authoritative
specs:

- `docs/design-document.md` — the authoritative project definition (requirements, architecture,
  design alternatives, scenarios, KPIs, evaluation criteria). **Source of truth for design
  decisions**, including the four runtime contracts in §2.3 that the source cites by name. If a
  decision changes, update it here rather than letting code and doc drift.
- `docs/conclusions.md` — what the measurements support, in one read.
- `docs/results/` — the experiment record; `docs/guides/` — CLI, architecture tour, policy guide;
  `docs/design/` — per-subsystem rationale; `docs/specs/` — the M1–M9 milestone specs.
- `src/ai_teleop/` — the package. See its `__init__.py` docstring for the module map.

`project-wiki/…` citations appear in the milestone specs and in some source comments. That wiki
is a **private** knowledge base in the (non-public) workspace parent, not part of this
repository — the citations are provenance markers, not links a reader can follow.

## Python environment — uv

Managed with [uv](https://github.com/astral-sh/uv). The virtualenv lives at `.venv`
(Python 3.12). **First-time setup after cloning:** `./scripts/setup.sh` — creates
the venv, installs `.[dev]`, enables the git hooks, and installs the `kvn` launcher
on PATH. The individual steps, from this directory:

- `uv venv` — create the venv.
- `uv pip install -e ".[dev]"` — package + dev tooling.
- `uv pip install -e ".[dev,ml,stereo-input]"` — full stack incl. torch + mediapipe.
  (Prefer `./scripts/setup.sh --dev`, which syncs the committed `uv.lock` instead of re-resolving.)
- `uv run python scripts/<script>.py` — run inside the venv.

### Dependency pinning — every ceiling is deliberate

Every requirement in `pyproject.toml`, runtime and extras, carries an **upper bound** at
the major this code is tested against. An open `>=` hands the choice of major to upstream
release timing, which is how `stereohand` silently moved to OpenCV 5: the whole suite
stayed green and live calibration crashed the moment a board entered frame.

`tests/test_dependency_pins.py` enforces both halves — that no declaration is
open-ended, and that the installed environment matches what's declared. `ruff` and
`qualety` are pre-1.0 and gate CI, so their ceiling is the **minor**, not the major.

Two kevin-specific notes:

- `uv.lock` **is** committed here (unlike `stereohand`, which is a library and gitignores
  its lock), so the lock and `pyproject.toml` have to move together. Bump a ceiling and
  re-run `uv lock` in the same change.
- The `stereo-input` extra pulls `opencv-contrib-python` and `mediapipe` **transitively
  through `stereohand`**, so those two majors are decided by *stereohand's* pins. Bumping
  them means bumping stereohand's tag here, not adding a constraint of our own.

### Project CLI — `kvn`

`kvn` (pronounced *"Kevin"*) is the project's command-line front door — one entry
point for the whole workflow instead of `uv run python scripts/...`. It's a thin
dispatcher (`src/ai_teleop/cli.py`): simulation/data commands run the matching
script in `scripts/`; dev-gate commands delegate to the poe tasks below. Full
reference: `docs/guides/cli.md`.

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
| `uv run poe structure` | `qualety` structural rules (see below) |
| `uv run poe test` | `pytest` |
| `uv run poe check` | lint + typecheck + structure + test (the full gate, same as CI) |
| `uv run poe sim [args]` | launch the wall viewer (e.g. `uv run poe sim --seed 7`) |
| `uv run poe smoke` | run the sim smoke test |
| `uv run poe cli [args]` | reach the `kvn` CLI without the console script (relocated-venv-safe) |

Prefer these over remembering the underlying commands. Add a new task here
rather than scattering one-off invocations. (mypy/pytest run via `python -m`
inside the tasks because the relocated `.venv` has stale console-script
shebangs — see the hooks note below.)

### Structural rules (qualety)

`poe structure` runs [qualety](https://github.com/NavehBrenner/qualety) — AST-level
invariants that ruff and mypy do not express (single-use indirection, untested public
exports, missing annotations on public callables). It is part of `poe check` and gates
every PR alongside lint/typecheck/test.

Two things about `qualety.config.json` are deliberate and should not be "tidied":

- **`"ruff": false`.** qualety bundles its own ruff and, before v0.1.4, ran it against a
  standalone config that ignored ours — 104 false `I001` and 16 false `RUF100` whose
  suggested fixes turned a green `ruff check` red. We already gate on ruff via `poe lint`
  with our own `select`/`isort` settings, so the bundled phase is pure duplication and a
  second source of truth. Leave it off.
- **Seven rules are `"off"`, each for a stated reason.** Do not switch one back on
  without re-running `poe structure` and reading the output — the disabled set is not
  arbitrary:

  | rule | why off |
  |---|---|
  | `no-unnecessary-def`, `no-unnecessary-class` | Upstream false positive: a method called as `self._attr.method()` is not counted as a use, so live public methods are reported dead ([qualety#126](https://github.com/NavehBrenner/qualety/issues/126)). ~105 findings across our two repos. Re-enable when that lands. |
  | `no-silent-except` | Upstream false positive: `except X: continue` in a fallback chain, and `pass` that falls through to a documented default, are both read as swallows ([qualety#106](https://github.com/NavehBrenner/qualety/issues/106)). |
  | `no-public-any` | Fires on `**kwargs: Any` forwarding, which is the idiomatic annotation and mypy-strict-clean. All 8 of our hits are that shape. |
  | `public-exports-tested` | Mixed signal — some genuinely untested exports, some flagged despite a test reference. Worth revisiting deliberately as a coverage push, not as a gate. |
  | `no-sys-path-hack` | The `sys.path.insert` preamble in `scripts/` is a deliberate convention (run a script before the package is installed), documented in each script. 57 findings, all intended. |
  | `no-open-without-with` | One site, `dev_harness_controller.py` — a CSV handle streamed across the loop and closed after it, already carrying `# noqa: SIM115` with the reasoning. |

When a disabled rule's upstream issue closes, re-run `poe structure` against a fresh
qualety and drop the entry rather than letting the list ossify.

### Hooks and CI

Local git hooks live in `.githooks/` (version-controlled). Activate them once
per clone — they are *not* enabled automatically:

```
git config core.hooksPath .githooks
```

- **pre-commit** — `ruff format` on staged Python, re-staging what it changed.
- **pre-push** — blocks the push unless `mypy` passes.

CI (`.github/workflows/ci.yml`) gates every **PR into `master`**: it installs
all extras via `uv sync --all-extras` (which includes the `ml` extra bringing
`torch`, used by the M5 dataset-loader tests) and must pass `ruff`, `mypy`,
`qualety` and `pytest`. The hooks run the same
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
