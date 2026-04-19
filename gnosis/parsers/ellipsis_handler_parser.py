"""Wraps parsing/ellipsis_handler.py — detects `...` cells + pdfplumber fallback."""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Document


@register("parser", "ellipsis_handler")
class EllipsisHandlerParser:
    """Count ellipsis cells per page and append raw_text fallback when found."""

    def parse(self, source: Any, *,
              document: Document | None = None,
              **kwargs) -> Document:
        if document is None:
            raise ValueError("EllipsisHandlerParser requires a Document")


        from gnosis._impl.ellipsis_handler import handle_ellipsis_page

        ellipsis_pages: dict[int, dict] = {}
        for pg in document.pages:
            if not pg.markdown:
                continue
            augmented, meta = handle_ellipsis_page(
                pg.markdown,
                page=pg.page_num,
                pdfplumber_text=pg.raw_text,
            )
            if meta.get("has_ellipsis"):
                pg.markdown = augmented
                pg.meta["has_ellipsis"] = True
                pg.meta["affected_cells"] = meta["affected_cells"]
                pg.meta["fallback_appended"] = meta["fallback_appended"]
                ellipsis_pages[pg.page_num] = {
                    "affected_cells": meta["affected_cells"],
                    "fallback_appended": meta["fallback_appended"],
                }

        if ellipsis_pages:
            document.meta.setdefault("ellipsis_pages", {}).update(ellipsis_pages)
        return document
