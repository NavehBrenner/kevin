---
name: export-deck
description: "Export an HTML slide deck to PDF (and/or PPTX/PNG) via Windows Chrome, with the deck print-CSS fixes baked in and real-PDF verification. Use for 'export the deck to pdf', 'make a pptx of the deck', 'render <x>.html'."
trigger: /export-deck
---

# /export-deck

Turn an HTML deck (e.g. `design-review-deck.html`) into a clean PDF — and optionally a
PPTX you can edit — handling all the WSL/Windows-Chrome gotchas automatically.

## Usage

```
/export-deck design-review-deck.html            # -> PDF, verified
/export-deck design-review-deck.html --pptx     # PDF + editable PPTX
/export-deck page.html --png                     # single screenshot
```

## Context this needs

- **No native browser/poppler on this WSL box** — render via Windows Chrome over `/mnt/c`.
  All of that (Windows %TEMP% staging, `wslpath -w`, throwaway `--user-data-dir`, deck
  print-CSS overrides, real-PDF verification) is encoded in:
  `python3 ~/autonomous-systems-workshop/.claude/skills/lib/render_html.py`.
- Background memory: `windows-chrome-rendering` (the why behind each flag).

## Procedure

1. **PDF:**
   ```
   python3 ~/autonomous-systems-workshop/.claude/skills/lib/render_html.py \
     <deck>.html --pdf <deck>.pdf --deck --verify
   ```
   `--deck` injects the slide print-CSS (force backgrounds, one-slide-per-page, kill the
   fade-in-blank-slide-1 bug, flatten gradient-text titles). `--verify` rasterizes the
   **real** exported PDF (PyMuPDF) to a PNG — Read it to confirm; never trust a
   re-screenshot of the HTML.
2. **Eyeball more pages** if needed: re-run the verify snippet for other page indices
   (the script prints the rasterized PNG path).
3. **PPTX (editable copy)** when asked: build with `python-pptx` in a throwaway venv
   (`python3 -m venv /tmp/pptxvenv && /tmp/pptxvenv/bin/pip install python-pptx`). The
   reliable route for a faithful copy is one rasterized slide PNG per page (use
   `render_html.py --png` per slide, or PyMuPDF page pixmaps from the PDF) placed
   full-bleed on 13.33in×7.5in (16:9) slides. If the user wants *text-editable* slides,
   say that requires reconstructing content from the HTML and confirm scope first.
4. Report output paths.

## Notes

- Inject CSS into a temp copy only — never edit the source HTML (the script already does
  this).
- Deck geometry is 1280×720 → 16:9; `@page { size:1280px 720px; margin:0 }`.
