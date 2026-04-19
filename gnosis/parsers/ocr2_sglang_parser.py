"""OCR2 parser — wraps Nanonets-OCR2-3B via sglang API or local fallback.

Preserves exact OCR call params (prompt, DPI, max_tokens) from the existing
`smartsearch/ocr2_engine.py`. Only OCR pages that lack sufficient text
(page_type='scan') unless `ocr_all=True`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnosis._vendor import ensure_agent_search_on_path
from gnosis.core.registry import register
from gnosis.core.schema import Document, Page


@register("parser", "ocr2_sglang")
class OCR2SglangParser:
    """OCR pages via Nanonets-OCR2-3B (sglang API or local transformers)."""

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:30000",
        model_name: str = "nanonets/Nanonets-OCR2-3B",
        max_tokens: int = 4096,
        timeout: int = 120,
        dpi: int = 250,
        ocr_all: bool = False,
        min_text_chars: int = 300,
    ) -> None:
        self.api_base = api_base
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.dpi = dpi
        self.ocr_all = ocr_all
        self.min_text_chars = min_text_chars
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            ensure_agent_search_on_path()
            from smartsearch.ocr2_engine import OCR2Engine
            self._engine = OCR2Engine(
                api_base=self.api_base,
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        return self._engine

    def parse(self, source: Any, *,
              document: Document | None = None,
              **kwargs) -> Document:
        pdf_path = Path(str(source))
        if document is None:
            # Need pages from a source parser first — don't re-OCR from scratch
            raise ValueError(
                "OCR2SglangParser requires an input Document (run pdfplumber first)"
            )

        engine = self._get_engine()
        if not engine.is_available():
            # sglang offline + no local model — skip, Document unchanged
            document.meta["ocr2_skipped"] = "engine_unavailable"
            return document

        # Pick pages to OCR
        pages_to_ocr: list[int] = []
        for p in document.pages:
            needs_ocr = self.ocr_all or (
                p.meta.get("char_count", 0) < self.min_text_chars
            )
            if needs_ocr and not p.meta.get("ocr2_done"):
                pages_to_ocr.append(p.page_num)

        if not pages_to_ocr:
            return document

        results = engine.ocr_pdf_pages(str(pdf_path), pages_to_ocr, dpi=self.dpi)

        by_num = {p.page_num: p for p in document.pages}
        for page_num, md in results.items():
            pg = by_num.get(page_num)
            if pg is None:
                continue
            pg.markdown = md
            pg.meta["ocr2_done"] = True
            pg.meta["ocr2_raw"] = md

        document.meta.setdefault("ocr2_markdowns", {}).update(
            {str(k): v for k, v in results.items()}
        )
        return document
