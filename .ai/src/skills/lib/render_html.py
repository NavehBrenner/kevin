#!/usr/bin/env python3
"""Render an HTML file to PDF and/or PNG using **Windows** Chrome over /mnt/c.

This WSL2 box has no native chrome/chromium/wkhtmltopdf/weasyprint/poppler, so we
drive the Windows Chrome binary via interop. Encodes the gotchas learned the hard
way (see memory: windows-chrome-rendering):

  * stage every file under Windows %TEMP% (UNC \\wsl.localhost paths fail silently)
  * pass Windows paths (wslpath -w)
  * always use a throwaway --user-data-dir (else it attaches to the GUI session)
  * for a slide DECK, inject print-CSS overrides into a temp copy (never edit source):
      - force backgrounds, one-slide-per-page, disable fade animations (blank-slide bug),
        flatten gradient-text titles (stray bbox + file bloat).
  * verify by rasterizing the REAL exported PDF with PyMuPDF, not a re-screenshot.

Examples
--------
  python3 render_html.py deck.html --pdf deck.pdf --deck --verify
  python3 render_html.py page.html --png shot.png
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

CHROME_CANDIDATES = [
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]

DECK_PRINT_CSS = """
<style id="__render_overrides__">
@page { size: 1280px 720px; margin: 0 }
* { print-color-adjust: exact !important; -webkit-print-color-adjust: exact !important }
#stage { display: block !important }
@media print {
  * { animation: none !important }
  .slide { opacity: 1 !important }
}
.title h1 {
  background: none !important;
  -webkit-text-fill-color: #0d9488 !important;
  color: #0d9488 !important;
}
</style>
"""


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    sys.exit("no Windows Chrome/Edge found under /mnt/c")


def win_temp_dir() -> str:
    """Return the Windows %TEMP% as a Linux /mnt/c path."""
    out = subprocess.run(
        ["cmd.exe", "/c", "echo %TEMP%"], capture_output=True, text=True
    ).stdout.strip()
    return subprocess.run(["wslpath", "-u", out], capture_output=True, text=True).stdout.strip()


def to_win(path: str) -> str:
    return subprocess.run(
        ["wslpath", "-w", os.path.abspath(path)], capture_output=True, text=True
    ).stdout.strip()


def stage_html(src: str, win_temp: str, deck: bool) -> str:
    """Copy src into Windows %TEMP%, optionally injecting deck print-CSS. Returns linux path."""
    html = open(src, encoding="utf-8").read()
    if deck:
        if "</head>" in html:
            html = html.replace("</head>", DECK_PRINT_CSS + "</head>", 1)
        else:
            html = DECK_PRINT_CSS + html
    dst = os.path.join(win_temp, f"render_{uuid.uuid4().hex}.html")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(html)
    return dst


def verify_pdf(pdf_path: str, page: int):
    venv = "/tmp/pdfvenv"
    if not os.path.exists(f"{venv}/bin/python"):
        subprocess.run(["python3", "-m", "venv", venv], check=True)
        subprocess.run([f"{venv}/bin/pip", "install", "-q", "pymupdf"], check=True)
    snippet = (
        "import fitz,sys;"
        "d=fitz.open(sys.argv[1]);"
        "print('pages=',d.page_count);"
        "d[int(sys.argv[2])].get_pixmap(dpi=110).save(sys.argv[3])"
    )
    out_png = f"/tmp/render_verify_p{page}.png"
    subprocess.run([f"{venv}/bin/python", "-c", snippet, pdf_path, str(page), out_png], check=True)
    print(f"verify: rasterized real PDF page {page} -> {out_png} (Read it to eyeball)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("html")
    p.add_argument("--pdf", help="output PDF path (linux)")
    p.add_argument("--png", help="output PNG path (linux)")
    p.add_argument("--deck", action="store_true", help="inject slide-deck print CSS")
    p.add_argument("--window-size", default="1280,720", help="PNG window size")
    p.add_argument("--verify", action="store_true", help="rasterize the real PDF to a PNG")
    p.add_argument("--verify-page", type=int, default=0)
    args = p.parse_args()

    if not args.pdf and not args.png:
        sys.exit("give --pdf and/or --png")

    chrome = find_chrome()
    win_temp = win_temp_dir()
    staged = stage_html(args.html, win_temp, args.deck)
    win_profile = to_win(os.path.join(win_temp, f"prof_{uuid.uuid4().hex}"))

    try:
        if args.pdf:
            win_out = os.path.join(win_temp, f"out_{uuid.uuid4().hex}.pdf")
            cmd = [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={win_profile}",
                f"--print-to-pdf={to_win(win_out)}",
                to_win(staged),
            ]
            subprocess.run(cmd, check=False, capture_output=True, text=True)
            if not os.path.exists(win_out):
                sys.exit("PDF was not produced (check Chrome interop / paths)")
            shutil.copyfile(win_out, args.pdf)
            print(f"PDF -> {args.pdf}")
            if args.verify:
                verify_pdf(args.pdf, args.verify_page)

        if args.png:
            win_out = os.path.join(win_temp, f"out_{uuid.uuid4().hex}.png")
            cmd = [
                chrome,
                "--headless",
                "--disable-gpu",
                f"--user-data-dir={win_profile}",
                f"--screenshot={to_win(win_out)}",
                f"--window-size={args.window_size}",
                to_win(staged),
            ]
            subprocess.run(cmd, check=False, capture_output=True, text=True)
            if not os.path.exists(win_out):
                sys.exit("PNG was not produced (check Chrome interop / paths)")
            shutil.copyfile(win_out, args.png)
            print(f"PNG -> {args.png}")
    finally:
        try:
            os.remove(staged)
        except OSError:
            pass


if __name__ == "__main__":
    main()
