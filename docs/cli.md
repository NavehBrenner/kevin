# `kvn` — the project CLI

**K.V.N** (pronounced *"Kevin"*) is the single command-line front door for the
project. Type `kvn <command>` instead of `uv run python scripts/<script>.py`.

`kvn` is a thin dispatcher, not a reimplementation:

- **Simulation / data commands** run the matching script in [`scripts/`](../scripts)
  with the project interpreter. Each script keeps its own `argparse`, so
  `kvn <command> --help` shows that script's real flags and every option passes
  straight through.
- **Dev-gate commands** delegate to the [poe](https://poethepoet.natn.io/) tasks in
  [`pyproject.toml`](../pyproject.toml), so the gate has a single source of truth.

Source: [`src/ai_teleop/cli.py`](../src/ai_teleop/cli.py).

## Install / invoke

### Recommended: one-time setup (gives you a bare `kvn`)

After cloning, run the setup script once from `code/`:

```bash
./scripts/setup.sh
```

It creates the `.venv`, installs `.[dev]`, enables the git hooks, and drops a `kvn`
launcher in `~/.local/bin` (override with `KVN_BIN_DIR`; extras with
`EXTRAS="dev,ml"`). The launcher calls the venv interpreter directly
(`python -m ai_teleop.cli`), so it keeps working even if the `.venv` is relocated.
Once `~/.local/bin` is on your PATH:

```bash
kvn                 # list all commands
kvn sim --seed 7    # run a command
```

Re-run `./scripts/setup.sh` after moving the repo (it repoints the launcher).

### Without the launcher: `uv run kvn`

`kvn` is also registered as a console script (`[project.scripts]` in
`pyproject.toml`), so after a plain install you can call it through uv:

```bash
uv pip install -e ".[dev]"     # registers the `kvn` command
uv run kvn                     # list all commands
uv run kvn sim --seed 7        # run a command
```

> **Relocated-venv note.** A `.venv` that is moved after creation leaves
> console-script shebangs stale (see [`code/CLAUDE.md`](../CLAUDE.md)). The setup
> launcher avoids this by invoking the interpreter directly. If you skipped setup
> and `uv run kvn` fails for that reason, use one of these — they don't rely on the
> shebang:
>
> ```bash
> uv run poe cli <command> [args]            # via the poe task
> uv run python -m ai_teleop.cli <command>   # via the module
> ```

## Command reference

`kvn <command> --help` always prints the underlying script's full, authoritative
flag list. The tables below are a quick reference.

### Simulation / data commands

| Command | Script | What it does |
|---|---|---|
| `kvn sim` | `view_generated_wall.py` | Generate (or load) a procedural wall and view it — interactive viewer or rendered PNGs. |
| `kvn smoke` | `smoke_test_sim.py` | M1 smoke test: load the scene, step it, dump sensor readings and a wrist-cam PNG. |
| `kvn episode` | `run_episode.py` | Run one end-to-end no-assist episode (scripted human → seam → controller → sim). |
| `kvn harness` | `dev_harness_controller.py` | M2 backbone-controller dev harness: the five-phase tuning/regression run. |
| `kvn gen` | `generate_dataset.py` | Generate the behavioral-cloning dataset: N unattended episodes → one NPZ per episode. |

#### `kvn sim` — view a procedural wall

| Flag | Default | Meaning |
|---|---|---|
| `--seed N` | time-based | RNG seed for wall generation. |
| `--distractors N` | random 0–10 | Number of distractor holes. |
| `--wall-dir PATH` | — | View an existing generated wall instead of generating a new one. |
| `--no-robot` | off | Preview the wall alone (no Panda/peg). |
| `--render` | off | Render PNGs instead of opening an interactive window. |

```bash
uv run kvn sim --seed 7                          # wall in the full scene, interactive
uv run kvn sim --seed 1 --distractors 3 --render # explicit holes, PNGs (headless)
uv run kvn sim --wall-dir outputs/walls/wall_7   # re-view a cached wall
```

#### `kvn smoke` — M1 scene smoke test

| Flag | Default | Meaning |
|---|---|---|
| `--no-viewer` | off | Skip the interactive viewer step (use in CI / over SSH without a display). |

```bash
uv run kvn smoke                # load, step, save wrist-cam PNG, then open viewer
uv run kvn smoke --no-viewer    # headless (CI)
```

#### `kvn episode` — one end-to-end no-assist episode

| Flag | Default | Meaning |
|---|---|---|
| `--headless` | off | Skip the viewer; run the loop and print a one-line summary. |
| `--seed N` | `0` | Seed for the scripted human's noise and the `SimEnv`. |
| `--max-steps N` | script default | Episode step budget (one step = one 2 ms sim tick). |
| `--generated-wall` | off | Run on a freshly generated procedural wall instead of the static scene. |
| `--wall-seed N` | `7` | Seed for `--generated-wall`. |
| `--distractors N` | — | Distractor-hole count for `--generated-wall`. |
| `--max-dpos M` | `0.025` | Controller command clamp in m/step (approach-speed / strictness knob). |

```bash
uv run kvn episode                                  # interactive viewer
uv run kvn episode --headless --seed 7 --max-steps 1500
uv run kvn episode --headless --generated-wall --wall-seed 3 --distractors 4
```

#### `kvn harness` — M2 controller dev harness

Drives the backbone controller through the five milestone-2 phases (waypoint
square → compliance → force-trip → release → park). In `--headless` mode it emits
assertions and a CSV trace for tuning plots.

| Flag | Default | Meaning |
|---|---|---|
| `--headless` | off | Skip the viewer; run assertions and emit the CSV trace. |
| `--force-cap N` | `30.0` | Force-cap watchdog threshold, in newtons. |

```bash
uv run kvn harness                       # interactive viewer
uv run kvn harness --headless            # CI / regression, writes CSV
uv run kvn harness --headless --force-cap 25
```

#### `kvn gen` — generate the BC dataset

Runs N unattended episodes (coverage-randomized scene → realistic noisy human →
analytical expert → controller → sim) and writes **one NPZ trajectory file per
episode**. Episodes are reproducible from `(seed, episode_index)` and cached by
fingerprint. On-disk schema: [`docs/data-schema.md`](data-schema.md).

| Flag | Default | Meaning |
|---|---|---|
| `--episodes N` | `200` | Number of episodes to run. |
| `--out PATH` | `data/runs/dev` | Output directory for the NPZ files. |
| `--seed N` | `0` | Master seed. |
| `--max-steps N` | script default | Per-episode step cap. |
| `--max-dpos M` | script default | Controller command clamp in m/step. |
| `--expert-d-far M` | script default | Distance (m) at which the expert starts engaging. |
| `--force` | off | Regenerate even if a cached episode with a matching fingerprint exists. |

```bash
uv run kvn gen --episodes 200 --out data/runs/dev
uv run kvn gen --episodes 5 --out /tmp/smoke --max-steps 800
```

### Dev-gate commands

These mirror the poe tasks (`kvn check` ≡ `uv run poe check`). They take no flags;
arguments pass through to the underlying tool.

| Command | Runs |
|---|---|
| `kvn fmt` | `ruff format` (after an import-fixing `ruff check`). |
| `kvn lint` | `ruff check --fix`. |
| `kvn typecheck` | `mypy`. |
| `kvn test` | `pytest`. |
| `kvn check` | `lint` + `typecheck` + `test` — the full gate, same as CI. |

```bash
uv run kvn check                 # the full pre-push / CI gate
uv run kvn test tests/test_foo.py -k case   # extra args pass through to pytest
```

## Adding a command

- **A new runnable script** → add it to `scripts/`, then add one line to
  `APP_COMMANDS` in [`src/ai_teleop/cli.py`](../src/ai_teleop/cli.py). The script's
  own `argparse` is the source of truth for its flags; `kvn` just forwards them.
- **A new dev-gate action** → add a poe task in `pyproject.toml`, then add one line
  to `DEV_COMMANDS`. Don't reimplement the command inside the CLI.
