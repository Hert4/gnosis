"""pdftohtml + DOM tree parser — wraps smartsearch/text_extractor.py.

Produces structured text preserving table/heading hierarchy for pages that
have a real text layer (unlike pdfplumber which gives flat text).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnosis._vendor import ensure_agent_search_on_path
from gnosis.core.registry import register
from gnosis.core.schema import Document


@register("parser", "text_extractor")
class TextExtractorParser:
    """Structured text via pdftohtml → DOM tree."""

    def __init__(self, batch_size: int = 50) -> None:
        self.batch_size = batch_size
        self._extractor = None

    def _get_extractor(self):
        if self._extractor is None:
            ensure_agent_search_on_path()
            from smartsearch.text_extractor import TextExtractor
            self._extractor = TextExtractor()
        return self._extractor

    def parse(self, source: Any, *,
              document: Document | None = None,
              **kwargs) -> Document:
        pdf_path = Path(str(source))
        if document is None:
            raise ValueError(
                "TextExtractorParser requires input Document (run pdfplumber first)"
            )

        extractor = self._get_extractor()
        all_pages = [p.page_num for p in document.pages]
        by_num = {p.page_num: p for p in document.pages}

        for i in range(0, len(all_pages), self.batch_size):
            batch = all_pages[i:i + self.batch_size]
            try:
                results = extractor.extract_pages(str(pdf_path), batch)
            except Exception as e:
                document.meta.setdefault("text_extractor_errors", []).append(str(e))
                continue
            for page_num, text in results.items():
                pg = by_num.get(page_num)
                if pg is None:
                    continue
                # Only override if no OCR has run yet — OCR output wins
                if not pg.meta.get("ocr2_done"):
                    pg.markdown = text
                    pg.meta["text_extractor_done"] = True

        return document
