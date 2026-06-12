# CLAUDE.md — code/

Conventions for the implementation tree. Inherits project-wide rules from the root
`../CLAUDE.md` (notably the shell-IO file-IO convention and the no-commit rule for the
course booklet).

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
(Python 3.12). From this directory:

- `uv venv` — create the venv.
- `uv pip install -e ".[dev]"` — package + dev tooling.
- `uv pip install -e ".[dev,ml,vision-input]"` — full stack incl. torch + mediapipe.
- `uv run python scripts/<script>.py` — run inside the venv.

### Task runner — poe (the "project CLI")

Common dev actions are defined as [poethepoet](https://poethepoet.natn.io/)
tasks in `pyproject.toml` (`[tool.poe.tasks]`). Run them with `uv run poe <task>`:

| Command | Does |
|---|---|
| `uv run poe fmt` | `ruff format` the code |
| `uv run poe lint` | `ruff check` |
| `uv run poe typecheck` | `mypy` |
| `uv run poe test` | `pytest` |
| `uv run poe check` | lint + typecheck + test (the full gate, same as CI) |
| `uv run poe sim [args]` | launch the wall viewer (e.g. `uv run poe sim --seed 7`) |
| `uv run poe smoke` | run the sim smoke test |

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
