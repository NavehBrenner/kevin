# CLAUDE.md — code/

Conventions for the implementation tree. Inherits project-wide rules from the root
`../CLAUDE.md` (notably the shell-IO file-IO convention).

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

## Git workflow

- **One branch per feature.** Never commit feature work directly to the default
  branch (`master`) — it stays clean and releasable. For each new feature or
  change, create a dedicated branch first (`git switch -c feat/lab-<NN>-<short-name>`, named after its Linear issue — see *Linking PRs to Linear*),
  do the work there, and merge when it's complete and reviewed.
- Keep a branch scoped to a single feature; unrelated fixes get their own branch.
- **Use `git switch` for branch navigation** — `git switch <branch>` to move,
  `git switch -c <branch>` to create. Prefer it over `git checkout` /
  `git checkout -b` (and `git branch` is for listing/deleting, not moving).

### Linking PRs to Linear

Issues are tracked in Linear (team **Lab**, key `LAB`). A PR appears on its
Linear issue automatically **only if the issue identifier is present** in
either:

- the **branch name** — embed it, e.g. `feat/lab-42-impedance-tuning`
  (Linear matches the `lab-NN` substring; the `feat/` prefix and trailing
  slug stay free-form). Linear's *Copy git branch name* yields exactly this.
- the **PR title or description** — e.g. a `Fixes LAB-42` / `Part of LAB-42`
  line. `Fixes`/`Closes` additionally moves the issue to Done on merge.

One PR per issue is the norm; reference every issue a PR addresses. (Requires
the workspace GitHub integration to be connected once, in Linear → Settings →
Features → Integrations → GitHub.)

### Hooks and CI

Local git hooks live in `.githooks/` (version-controlled). Activate them once
per clone — they are *not* enabled automatically:

```
git config core.hooksPath .githooks
```

- **pre-commit** — `ruff format` on staged Python, re-staging what it changed.
- **pre-push** — blocks the push unless `mypy` passes.

CI (`.github/workflows/ci.yml`) gates every **PR into `master`**: it installs
`.[dev,scenegen]` and must pass `mypy` and `pytest`. The hooks run the same
tools via `uv run` (mypy as `uv run python -m mypy`, since the relocated
`.venv` has stale console-script shebangs).

## Code conventions

- Python ≥ 3.10. Favor high-level Python; **no C/C++/Rust extensions, no ROS.**
- Lint/format with ruff; type-check with mypy; test with pytest. Config in `pyproject.toml`.
- Built milestone by milestone — see `docs/`. Respect each milestone's anti-scope;
  don't pull future work forward.
- **Log, don't `print`.** For status/progress/errors in scripts and library code, use
  the project logger (`from ai_teleop.common.log import get_logger`; `log = get_logger("<tag>")`)
  instead of `print`. Scripts wire the shared `--log-level/--quiet/--log-file` flags via
  `add_logging_arguments` + `configure_from_args`. Keep the per-tick control loop
  (`run_episode`, the controller) logging-free. See [`docs/cli.md`](docs/cli.md#logging).

## Ad-hoc debugging / experiments

When exploring or tuning, **write the snippet as a file under `scripts/dev/`** and run it
with `uv run python scripts/dev/<name>.py`. Do *not* pass code with `python -c "..."` heredocs.
Reasons: the user can read, edit, and re-run the script themselves; the script is reviewable in
the diff; iteration is faster (edit-and-rerun instead of rewriting the whole heredoc); and the
file becomes a permanent artifact (or a deliberately-deleted one) instead of vanishing into
shell history.

`scripts/dev/` is for one-off probes and tuning sweeps. Production-ish runnable scripts
(smoke tests, dev harnesses, data-generation drivers) live directly in `scripts/`.

## Variable naming

Prefer **verbose, self-documenting names** unless the fully verbose form is unreasonably long.

| Short | Prefer |
|---|---|
| `m` | `model` |
| `p` / `pos` | `position` |
| `vel` | `velocity` |
| `obs` | `observation` |
| `cfg` | `config` |
| `env` | `environment` (or keep `env` if it appears ≥ 5× in the same scope — see dictionary) |

When a short form is genuinely warranted (long compound nouns, loop counters, math), register
it in `docs/acronym-dictionary.md` so it stays intentional and searchable. New abbreviations
not in that file should be treated as typos during review.

Counter-examples where brevity is fine: loop indices (`i`, `j`), standard math symbols
(`q` for joint angles, `R` for rotation matrix), and names that are established domain
shorthand already in the dictionary.

## Type definitions

**Reuse before defining.** The shared value and interface types already exist — import
them, never re-declare a local pose/command/observation/delta struct:

| Type | Lives in | Re-exported from |
|---|---|---|
| `Command`, `Observation` | `common/command.py`, `common/observation.py` | `ai_teleop.common` |
| `Delta`, `InputStrategy`, `AssistProvider` | `domain/delta.py`, `domain/interfaces.py` | `ai_teleop.domain` |

**Where a new type goes — by reach, narrowest that fits:**

- **Used by one module only** → define it inline in that module (e.g. `NoAssist` lives
  in `delta.py`). Don't spin up a file for a single-consumer type.
- **A sim-independent value type shared across layers** → `common/` (the leaf of the
  dependency DAG — must not import `ai_teleop.sim`).
- **A behavioral interface other layers depend on** → `domain/` as a `Protocol` in
  `interfaces.py` (Dependency Inversion: concretes depend on the abstraction).
- **A subsystem's own config + resolved-spec types** → a `config.py` *inside* that
  subsystem package (e.g. `sim/scenegen/config.py`).
- **An on-disk / serialization contract** (JSON/`.npz` shapes) → a dedicated `schema.py`
  in that package (e.g. `data/schema.py`). Keep it behavior-free with no implementation
  imports, so anything can depend on it without a cycle.

**Which mechanism:**

- Immutable value object / config you don't mutate → `@dataclass(frozen=True)`.
- A spec built up in stages → plain `@dataclass` (methods/`@property` are fine).
- Behavioral interface → `typing.Protocol`; add `@runtime_checkable` only when a test
  asserts conformance — mypy structural typing is the real guarantee.
- On-disk dict shape → `TypedDict`. Optional keys via the base-class + `total=False`
  split, **not** `typing.NotRequired` (3.11+; the project targets 3.10).
- A closed set of string values → `Literal`, aliased at module top (e.g. `HoleShape`).

Respect the dependency DAG: `common/` is the leaf, `domain/` depends on `common/`, and
neither imports `sim/`. That is *why* the contract modules (`schema.py`, `interfaces.py`)
stay behavior-free — a new shared type must not introduce a cycle.
