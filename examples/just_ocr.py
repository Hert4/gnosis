"""Example: use only the OCR parser — no indexing, no retrieval.

Runs Nanonets-OCR2-3B on pages that lack a text layer, then normalizes
HTML tables. Prints each page's normalized markdown.

Backend is any OpenAI-compatible endpoint (sglang default, or pass
`api_base`/`api_key` to OCR2Parser for vLLM / OpenRouter / LM Studio / ...).
Falls back to local transformers if GPU + model cached.

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
    OCR2Parser,
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
    doc = OCR2Parser().parse(pdf, document=doc)
    doc = TableNormalizerParser().parse(pdf, document=doc)
    doc = EllipsisHandlerParser().parse(pdf, document=doc)

    for pg in doc.pages[:5]:  # show first 5
        print(f"\n========== Page {pg.page_num} ({pg.page_type}) ==========")
        print(pg.markdown[:800])

    print(f"\nTotal: {doc.total_pages} pages, {len(doc.tables)} tables extracted")


if __name__ == "__main__":
    main()
