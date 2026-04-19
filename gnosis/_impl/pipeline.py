"""Main parsing pipeline — orchestrate per-page normalization.

Takes raw OCR2 markdown (output of Nanonets-OCR2-3B, produced by the
existing `ocr2_engine.py`) and converts it into LLM-friendly markdown:

  raw OCR2 markdown
    → detect `...` cells, append pdfplumber fallback
    → parse HTML tables → resolve colspan/rowspan → render KV or pipe
    → tag captions + footnotes

Does NOT call OCR2 itself. Does NOT change OCR parameters. This module
runs as a pure post-processing layer.
"""

from __future__ import annotations

from gnosis._impl import ellipsis_handler, element_classifier, table_normalizer
from gnosis._impl.html_table_parser import parse_tables_from_markdown
from gnosis._impl.native_schema import Document, Section, Table


def normalize_page(
    raw_md: str,
    page: int,
    pdfplumber_text: str = "",
) -> tuple[str, list[Table], dict]:
    """Normalize a single page.

    Args:
        raw_md: OCR2 raw markdown output for this page (HTML tables inline).
        page: 1-indexed page number.
        pdfplumber_text: PDF text layer for this page (fallback on ellipsis).

    Returns:
        (normalized_markdown, extracted_tables, metadata)

    metadata keys:
        has_ellipsis, affected_cells, fallback_appended, captions, footnotes
    """
    # 1. Detect ellipsis cells in raw OCR (metadata only, don't append yet —
    #    appending must happen AFTER table normalization so the fallback text
    #    is not swallowed into a truncated <table> range).
    ellipsis_count = ellipsis_handler.count_ellipsis_cells(raw_md)
    has_ellipsis = ellipsis_count > 0

    # 2. Extract Table objects from raw markdown (before replacement)
    tables = parse_tables_from_markdown(raw_md, page=page)

    # 3. Rewrite HTML tables in-place as KV / pipe markdown
    md = table_normalizer.normalize_tables_in_markdown(raw_md, page=page)

    # 4. Append pdfplumber fallback AFTER table normalization if ellipsis detected.
    #    (handle_ellipsis_page re-counts by regex, but after normalization the
    #    `...` literals are gone — use our pre-normalization count directly.)
    fallback_appended = False
    if has_ellipsis:
        pdf_text = (pdfplumber_text or "").strip()
        if pdf_text and len(pdf_text) >= 50:
            md = md + (
                f"\n\n<!-- OCR_PARTIAL: {ellipsis_count} cells returned as '...' "
                f"by OCR on page {page}. Raw PDF text layer appended below as "
                f"supplementary context. -->\n\n"
                f"**[Supplementary PDF text layer — page {page}]**\n\n"
                f"{pdf_text}\n"
            )
            fallback_appended = True
        else:
            md = md + (
                f"\n\n<!-- OCR_PARTIAL: {ellipsis_count} cells returned as '...' "
                f"by OCR on page {page}. No PDF text layer available for fallback. -->\n"
            )

    # 5. Tag captions and footnotes
    md, element_meta = element_classifier.tag_elements(md)

    metadata = {
        "has_ellipsis": has_ellipsis,
        "affected_cells": ellipsis_count,
        "fallback_appended": fallback_appended,
        **element_meta,
    }
    return md, tables, metadata


def build_document(
    raw_ocr_markdowns: dict[int, str],
    pdfplumber_texts: dict[int, str] | None,
    total_pages: int,
    doc_name: str,
    sections: list[Section] | None = None,
) -> Document:
    """Build a full Document object from per-page OCR2 markdowns.

    Args:
        raw_ocr_markdowns: {page_num: raw OCR2 markdown string}.
        pdfplumber_texts: {page_num: pdfplumber-extracted text} or None.
        total_pages: total page count.
        doc_name: document identifier.
        sections: optional Section list (e.g., from tree_index).

    Returns:
        Document with normalized page_markdowns, aggregated tables, and
        ellipsis_pages metadata.
    """
    pdfplumber_texts = pdfplumber_texts or {}

    page_markdowns: dict[int, str] = {}
    all_tables: list[Table] = []
    ellipsis_pages: dict[int, dict] = {}

    for page_num, raw_md in raw_ocr_markdowns.items():
        pdf_text = pdfplumber_texts.get(page_num, "")
        normalized, tables, meta = normalize_page(raw_md, page=page_num,
                                                  pdfplumber_text=pdf_text)
        page_markdowns[page_num] = normalized
        all_tables.extend(tables)
        if meta.get("has_ellipsis"):
            ellipsis_pages[page_num] = {
                "affected_cells": meta.get("affected_cells", 0),
                "fallback_appended": meta.get("fallback_appended", False),
            }

    return Document(
        name=doc_name,
        total_pages=total_pages,
        sections=sections or [],
        tables=all_tables,
        figures=[],
        page_markdowns=page_markdowns,
        ellipsis_pages=ellipsis_pages,
    )
