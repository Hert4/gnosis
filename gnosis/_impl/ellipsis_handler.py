"""Detect and handle OCR2-produced ellipsis cells.

Nanonets-OCR2-3B sometimes outputs literal `<td>...</td>` or `<th>...</th>`
when it truncates wide tables. We DO NOT retry OCR (constraint: OCR params
must stay unchanged — prompt/DPI/max_tokens). Instead we:

  1. Count affected cells per page for metadata
  2. Append the raw pdfplumber text layer (if present) as supplementary
     context so downstream LLM has some fallback text to work with.
"""

from __future__ import annotations

import re

_ELLIPSIS_CELL_RE = re.compile(r"<(t[dh])>\s*\.{3,}\s*</\1>", re.IGNORECASE)


def count_ellipsis_cells(markdown: str) -> int:
    """Return number of <td>...</td> or <th>...</th> literal cells."""
    return len(_ELLIPSIS_CELL_RE.findall(markdown))


def has_ellipsis_cells(markdown: str) -> bool:
    return bool(_ELLIPSIS_CELL_RE.search(markdown))


def handle_ellipsis_page(
    markdown: str,
    page: int,
    pdfplumber_text: str = "",
) -> tuple[str, dict]:
    """Detect ellipsis cells and attach pdfplumber fallback.

    Returns:
        (augmented_markdown, metadata)

    metadata keys:
        has_ellipsis: bool
        affected_cells: int
        fallback_appended: bool
    """
    n = count_ellipsis_cells(markdown)
    meta = {
        "has_ellipsis": n > 0,
        "affected_cells": n,
        "fallback_appended": False,
    }

    if n == 0:
        return markdown, meta

    # Append pdfplumber text as supplementary material. Only if non-trivial.
    pdf_text = (pdfplumber_text or "").strip()
    if pdf_text and len(pdf_text) >= 50:
        note = (
            f"\n\n<!-- OCR_PARTIAL: {n} cells returned as '...' by OCR on page {page}. "
            f"Raw PDF text layer appended below as supplementary context. -->\n\n"
            f"**[Supplementary PDF text layer — page {page}]**\n\n"
            f"{pdf_text}\n"
        )
        markdown = markdown + note
        meta["fallback_appended"] = True
    else:
        # No fallback — just annotate
        note = (
            f"\n\n<!-- OCR_PARTIAL: {n} cells returned as '...' by OCR on page {page}. "
            f"No PDF text layer available for fallback. -->\n"
        )
        markdown = markdown + note

    return markdown, meta
