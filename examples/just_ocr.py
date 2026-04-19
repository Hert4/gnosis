"""Example: use only the OCR parser — no indexing, no retrieval.

Runs Nanonets-OCR2-3B (sglang API) on pages that lack a text layer, then
normalizes HTML tables. Prints each page's normalized markdown.

Prereq: sglang serving Nanonets-OCR2-3B at http://127.0.0.1:30000 (optional;
pdfplumber handles text pages).

Usage:
    python examples/just_ocr.py path/to/doc.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from gnosis.parsers import (
    EllipsisHandlerParser,
    OCR2SglangParser,
    PdfplumberParser,
    TableNormalizerParser,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python examples/just_ocr.py <pdf>")
        sys.exit(1)
    pdf = sys.argv[1]

    doc = PdfplumberParser().parse(pdf)
    # OCR only scan pages (char_count < 300)
    doc = OCR2SglangParser().parse(pdf, document=doc)
    doc = TableNormalizerParser().parse(pdf, document=doc)
    doc = EllipsisHandlerParser().parse(pdf, document=doc)

    for pg in doc.pages[:5]:  # show first 5
        print(f"\n========== Page {pg.page_num} ({pg.page_type}) ==========")
        print(pg.markdown[:800])

    print(f"\nTotal: {doc.total_pages} pages, {len(doc.tables)} tables extracted")


if __name__ == "__main__":
    main()
