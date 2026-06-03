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
