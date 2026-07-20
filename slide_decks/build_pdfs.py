#!/usr/bin/env python3
"""Render animated logistic decks to high-quality, vector PDFs.

Usage:
    python3 build_pdfs.py

Adapted from intro_research_stats_admin/lessons/t_tests/build_pdfs.py.

These decks are a single 1600x900 canvas with many animation steps.
We export only the final step (everything revealed) via
?still=1&step=N-1.
"""
from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).parent.resolve()

DECKS = [
    (
        "intro_to_logistic_regression",
        "intro_to_logistic_regression.pdf",
    ),
    (
        "logistic_regression_coefficient_interpretation",
        "logistic_regression_coefficient_interpretation.pdf",
    ),
]

DECK_W = 1600
DECK_H = 900

PRINT_CSS = """
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: #ffffff !important;
    overflow: hidden !important;
    width: %(w)dpx !important;
    height: %(h)dpx !important;
}
#stage {
    position: static !important;
    inset: auto !important;
    padding: 0 !important;
    margin: 0 !important;
    width: %(w)dpx !important;
    height: %(h)dpx !important;
    display: block !important;
    overflow: hidden !important;
}
#deck {
    position: relative !important;
    width: %(w)dpx !important;
    height: %(h)dpx !important;
    transform: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    margin: 0 !important;
}
#tools, #hint, #counter, #ink {
    display: none !important;
}
.panel, .grp, .frag, .ols {
    transition: none !important;
}
""" % {"w": DECK_W, "h": DECK_H}


def count_steps(html_path: Path) -> int:
    """Count entries in the STEP_CHANGES array."""
    text = html_path.read_text(encoding="utf-8")
    m = re.search(r"const\s+STEP_CHANGES\s*=\s*\[(.*?)\];", text, re.S)
    if not m:
        raise RuntimeError(f"STEP_CHANGES not found in {html_path}")
    body = m.group(1)
    return len(re.findall(r"\{", body))


def build_deck_pdf(page, lesson_dir: Path, out_pdf: Path) -> None:
    index_html = lesson_dir / "index.html"
    if not index_html.is_file():
        raise FileNotFoundError(index_html)

    n_steps = count_steps(index_html)
    if n_steps == 0:
        raise RuntimeError(f"No steps found in {index_html}")

    final_step = n_steps - 1  # 0-based index of the fully revealed state
    print(f"  {lesson_dir.name}: final step {n_steps}/{n_steps}", flush=True)

    writer = PdfWriter()
    base_url = index_html.resolve().as_uri()
    url = f"{base_url}?still=1&step={final_step}"
    page.goto(url, wait_until="networkidle")

    page.wait_for_function(
        """([expected]) => {
            const el = document.getElementById('counter');
            return el && el.textContent.includes(`step ${expected} /`);
        }""",
        arg=[n_steps],
        timeout=15000,
    )
    page.evaluate("document.fonts && document.fonts.ready")
    page.wait_for_timeout(400)

    page.add_style_tag(content=PRINT_CSS)
    page.evaluate(
        """() => {
            const deck = document.getElementById('deck');
            if (deck) deck.style.transform = 'none';
        }"""
    )
    page.wait_for_timeout(150)

    pdf_bytes = page.pdf(
        width=f"{DECK_W}px",
        height=f"{DECK_H}px",
        print_background=True,
        margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        prefer_css_page_size=False,
    )

    reader = PdfReader(io.BytesIO(pdf_bytes))
    for p in reader.pages:
        writer.add_page(p)

    with out_pdf.open("wb") as f:
        writer.write(f)
    print(f"    -> {out_pdf} ({out_pdf.stat().st_size / 1024:.0f} KB)", flush=True)


def main() -> int:
    t0 = time.time()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport={"width": DECK_W, "height": DECK_H},
            device_scale_factor=2,
        )
        page = context.new_page()
        for slug, out_name in DECKS:
            lesson_dir = HERE / slug
            out_pdf = lesson_dir / out_name
            build_deck_pdf(page, lesson_dir, out_pdf)
        browser.close()

    print(f"Done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
