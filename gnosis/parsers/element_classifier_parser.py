"""Wraps parsing/element_classifier.py — regex tag captions/footnotes."""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Document


@register("parser", "element_classifier")
class ElementClassifierParser:
    """Tag captions + footnotes in each page's markdown."""

    def parse(self, source: Any, *,
              document: Document | None = None,
              **kwargs) -> Document:
        if document is None:
            raise ValueError("ElementClassifierParser requires a Document")


        from gnosis._impl.element_classifier import tag_elements

        all_captions: list[dict] = []
        all_footnotes: list[dict] = []
        for pg in document.pages:
            if not pg.markdown:
                continue
            tagged, meta = tag_elements(pg.markdown)
            pg.markdown = tagged
            for c in meta.get("captions", []):
                all_captions.append({**c, "page": pg.page_num})
            for f in meta.get("footnotes", []):
                all_footnotes.append({**f, "page": pg.page_num})

        if all_captions:
            document.meta.setdefault("captions", []).extend(all_captions)
        if all_footnotes:
            document.meta.setdefault("footnotes", []).extend(all_footnotes)
        return document
