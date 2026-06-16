# scripts/

Executable entry points for the project workflow. Prefer the **`kvn` CLI** as the
front door — it dispatches to these scripts and forwards their flags
(`kvn <command> --help`). See [`../docs/cli.md`](../docs/cli.md). Running a script
file directly with `uv run python scripts/<name>.py` still works.

One-time project setup (not a `kvn` command — it bootstraps `kvn` itself):

- `setup.sh`                   — create the venv, install the package, enable hooks, install the `kvn` launcher on PATH.

Current scripts and their `kvn` command:

- `view_generated_wall.py`     (`kvn sim`)     — generate / view a procedural wall.
- `smoke_test_sim.py`          (`kvn smoke`)   — M1: load scene, dump sensors + wrist-cam PNG.
- `run_episode.py`             (`kvn episode`) — M3: one end-to-end no-assist episode.
- `dev_harness_controller.py`  (`kvn harness`) — M2: backbone-controller dev harness.
- `generate_dataset.py`        (`kvn gen`)     — M4: unattended BC data generation.

Planned (later milestones):

- `train_policy.py`     — post-M4: BC training on logged data.
- `run_eval.py`         — post-M4: ablation orchestration + KPI computation.
- `run_demo.py`         — final: live webcam-driven shared-autonomy demo.

When you add a runnable script here, register it in `APP_COMMANDS` in
`../src/ai_teleop/cli.py` so it gets a `kvn` command.
