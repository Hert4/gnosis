"""pdfplumber parser — extracts raw text + classifies page type.

Produces a fresh Document when called with a PDF path. Designed as the
Tier-0 parser: fast, no OCR, just reads the PDF text layer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Document, Page

_MIN_TEXT_CHARS = 50


@register("parser", "pdfplumber")
class PdfplumberParser:
    """Extract text per page via pdfplumber. Classifies each as text/scan."""

    def __init__(self, min_text_chars: int = _MIN_TEXT_CHARS) -> None:
        self.min_text_chars = min_text_chars

    def parse(self, source: Any, *,
              document: Document | None = None,
              **kwargs) -> Document:
        import pdfplumber

        pdf_path = Path(str(source))
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: list[Page] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            total = len(pdf.pages)
            for i, pg in enumerate(pdf.pages):
                text = pg.extract_text() or ""
                has_text = len(text.strip()) >= self.min_text_chars
                pages.append(Page(
                    page_num=i + 1,
                    raw_text=text,
                    markdown=text,
                    page_type="text" if has_text else "scan",
                    has_text_layer=has_text,
                    meta={"char_count": len(text.strip())},
                ))

        if document is None:
            doc_id = _doc_id_from_path(pdf_path)
            document = Document(
                doc_id=doc_id,
                name=pdf_path.name,
                total_pages=total,
                pages=pages,
                meta={"source_path": str(pdf_path)},
            )
        else:
            # Enrichment mode — merge pages keyed by page_num
            existing = {p.page_num: p for p in document.pages}
            for p in pages:
                if p.page_num in existing:
                    e = existing[p.page_num]
                    if not e.raw_text:
                        e.raw_text = p.raw_text
                    if not e.markdown:
                        e.markdown = p.markdown
                    e.has_text_layer = e.has_text_layer or p.has_text_layer
                    e.meta.update(p.meta)
                else:
                    document.pages.append(p)
            document.total_pages = max(document.total_pages, total)

        return document


def _doc_id_from_path(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
