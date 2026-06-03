# D1 Design Review — deliverables

Materials for the mid-July course design review (D1, ~35% of grade).

## What's here

- `architecture.svg` — system architecture diagram (runtime control loop + eval observer + offline data-gen callout).
- `sequence.svg` — sequence chart for one 100 Hz control step.
- `build_deck.py` — Python script that builds `design-review.pptx` from the SVGs + structured slide content.
- `design-review.pptx` — *generated* by `build_deck.py`; not checked into git (see `.gitignore`).

The SVGs are standalone deliverables and can be embedded elsewhere (the deck embeds them as rasterized PNGs internally for compatibility).

## How to build the deck

From this directory:

```bash
# one-time: install the docs extras (python-pptx + cairosvg)
uv pip install -e "../..[docs]"     # from this folder, ../.. == code/

# build
uv run python build_deck.py
# → writes ./design-review.pptx
```

`cairosvg` has a system dep (`libcairo2`) that may not be installed. If install fails:

```bash
# Debian/Ubuntu
sudo apt install libcairo2

# Alternative: pre-rasterize the SVGs yourself, the script will pick up the PNGs
rsvg-convert -w 2000 architecture.svg -o architecture.png
rsvg-convert -w 2000 sequence.svg     -o sequence.png
uv run python build_deck.py
```

## Editing content

Slide content lives directly in `build_deck.py` as per-slide functions (`slide_title`, `slide_problem`, …). Edit text in the source, re-run, get a new `.pptx`.

To tweak the diagrams: edit the `.svg` files directly (any vector editor, or just open in a text editor — they're simple by design), then re-run `build_deck.py` to refresh the embedded raster.

## Slide map (17 slides)

| # | Title | Purpose |
|---|-------|---------|
| 1 | Title | Hook + identification |
| 2 | The problem | Why shared autonomy |
| 3 | What we're building | Concrete project picture |
| 4 | Goals and requirements | Functional / architectural / reliability |
| 5 | How we'll measure success | KPI list + comparison setup |
| 6 | High-level approach | Three-layer narration |
| 7 | System architecture | Full-slide architecture diagram |
| 8 | One control step | Full-slide sequence chart |
| 9 | What we actually learn | ML contribution detail |
| 10 | Two phases, deliberately | F/T-only vs vision split rationale |
| 11 | Alternatives considered | Design alternatives table (booklet requirement) |
| 12 | Evaluation scenarios | Easy / Standard / Hard configs |
| 13 | Three-way KPI comparison | Eval methodology |
| 14 | Risks and mitigations | R1–R4 |
| 15 | Milestone roadmap | Timeline through 2026-08-31 |
| 16 | Where we are today | M1 done + M2 next |
| 17 | Questions & discussion | Explicit feedback prompts |
