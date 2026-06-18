"""kvn — the project command-line front door (K.V.N, "Kevin").

A single entry point for the whole workflow, so you type ``kvn <command>`` instead
of ``uv run python scripts/<script>.py``. It is a thin dispatcher, not a
reimplementation:

- **App commands** (`sim`, `smoke`, `episode`, `harness`, `gen`) run the matching
  script under ``scripts/`` with the current interpreter. Each script keeps its own
  argparse, so ``kvn <command> --help`` shows that script's real options and every
  flag passes straight through.
- **Dev-gate commands** (`fmt`, `lint`, `typecheck`, `test`, `check`) delegate to the
  poe tasks in ``pyproject.toml`` so there is a single source of truth for the gate.

Run ``kvn`` (or ``kvn --help``) for the command list, ``kvn <command> --help`` for a
command's flags.

Invocation, in order of preference:

    kvn <command> [args]                      # installed console script
    uv run kvn <command> [args]               # via uv, no activation needed
    uv run poe cli <command> [args]           # relocated-venv-safe fallback
    uv run python -m ai_teleop.cli <command>  # always works, no console script

The console-script form needs ``uv pip install -e .`` to have run against the
*current* venv location (a relocated venv leaves a stale shebang — see
``code/CLAUDE.md``); the ``poe cli`` / ``python -m`` forms sidestep that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ai_teleop.common.log import configure_logging, get_logger

# src/ai_teleop/cli.py -> parents[2] is the code/ repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

log = get_logger("cli")

# command -> (script filename under scripts/, one-line help)
APP_COMMANDS: dict[str, tuple[str, str]] = {
    "sim": (
        "view_generated_wall.py",
        "Generate and view a procedural wall scene (viewer or PNGs).",
    ),
    "smoke": (
        "smoke_test_sim.py",
        "M1 scene smoke test: load, step, dump sensors + a wrist-cam PNG.",
    ),
    "episode": ("run_episode.py", "Run one end-to-end no-assist episode through the control seam."),
    "harness": (
        "dev_harness_controller.py",
        "M2 backbone-controller dev harness (five-phase tuning run).",
    ),
    "gen": (
        "generate_dataset.py",
        "Generate the behavioral-cloning dataset (N episodes -> NPZ files).",
    ),
    "train": (
        "train_policy.py",
        "Train the Phase-1 F/T residual via BC and write a deployable checkpoint.",
    ),
}

# command -> (poe task name, one-line help)
DEV_COMMANDS: dict[str, tuple[str, str]] = {
    "fmt": ("fmt", "Format the code (ruff format, after an import-fixing ruff check)."),
    "lint": ("lint", "Lint the code (ruff check --fix)."),
    "typecheck": ("typecheck", "Type-check the package (mypy)."),
    "test": ("test", "Run the test suite (pytest)."),
    "check": ("check", "Full gate: lint + typecheck + test (same as CI)."),
}


def _usage() -> str:
    width = max(len(name) for name in (*APP_COMMANDS, *DEV_COMMANDS))
    app = "\n".join(f"  {name:<{width}}  {summary}" for name, (_, summary) in APP_COMMANDS.items())
    dev = "\n".join(f"  {name:<{width}}  {summary}" for name, (_, summary) in DEV_COMMANDS.items())
    return (
        'kvn — K.V.N project CLI (pronounced "Kevin").\n\n'
        "Usage: kvn <command> [args...]\n"
        "       kvn <command> --help    # the command's own flags\n\n"
        f"Simulation / data commands:\n{app}\n\n"
        f"Dev-gate commands:\n{dev}\n"
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return 0

    command, passthrough = args[0], args[1:]

    if command in APP_COMMANDS:
        script = SCRIPTS_DIR / APP_COMMANDS[command][0]
        cmd = [sys.executable, str(script), *passthrough]
    elif command in DEV_COMMANDS:
        # Delegate to the poe task so pyproject.toml stays the single source of truth.
        cmd = [sys.executable, "-m", "poethepoet", DEV_COMMANDS[command][0], *passthrough]
    else:
        log.error("unknown command %r", command)
        print(_usage(), file=sys.stderr)
        return 2

    # Run from the repo root: the scripts and the poe tasks both assume code/ as cwd.
    return subprocess.call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
