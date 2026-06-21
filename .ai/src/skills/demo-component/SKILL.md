---
name: demo-component
description: "Run the script/test that visualizes a given component and show its behavior (video/PNG/log). Use for 'is there a test to see how the scripted human looks', 'how can I visualise X', 'show me X working'."
trigger: /demo-component
---

# /demo-component

Find and run the thing that lets you *see* a component behave, then surface the output.

## Usage

```
/demo-component scripted human       # run the demo that shows the scripted human
/demo-component wall generation      # visualize a generated wall
/demo-component <component>          # general
```

## Context this needs

- **Runnable entry points:** `code/scripts/` (e.g. `run_episode.py`, `view_generated_wall.py`,
  `dev_harness_controller.py`, `generate_dataset.py`) and `code/scripts/dev/` (probes,
  `record_*.py`, `sweep_*.py`). Prefer poe tasks: `uv run poe sim`, `uv run poe smoke`.
- **How to run:** from `code/`, `uv run python scripts/<name>.py [args]`. Console tools go
  via `python -m` (stale `.venv` shebangs).
- **Rendering:** the MuJoCo viewer is interactive (the user runs it); for headless/recorded
  output use the `record_*` dev scripts that emit MP4/PNG. To turn any HTML into PDF/PNG
  use `lib/render_html.py` (Windows-Chrome path) — but for a slide DECK use /export-deck.
- This box has no native browser/poppler — see memory `windows-chrome-rendering`.

## Procedure

1. **Map component → demo.** Search `code/scripts*` and `code/tests/` for the script that
   exercises the named component (the graphify graph in `graphify-out/` can answer
   "which script exercises X" fast). If several, pick the most visual one.
2. **Tell the user the command** and whether it's interactive (viewer pops up — they run it)
   or headless (you can run it and produce a file).
3. **Run headless demos yourself** (`uv run python scripts/...`). For interactive viewers,
   give the exact `uv run poe sim ...` / `run_episode.py` command and the flags that matter
   (e.g. `--seed`, `--max-steps`).
4. **Surface output:** Read produced PNGs; for MP4 rasterize a frame or two and Read them.
   Report what the behavior shows; note determinism (seeded vs. fresh per run) when relevant.

## Notes

- Overlaps the built-in `/run` and `/verify` skills — this one is the project-specific
  "which script shows component X" shortcut. Use `/run` for launching the app generally.
- If no demo exists for a component, say so and offer to add a `scripts/dev/` probe.
