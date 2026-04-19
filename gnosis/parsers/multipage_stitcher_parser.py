"""Wraps parsing/multipage_stitcher.py — merges tables spanning multiple pages."""

from __future__ import annotations

from typing import Any

from gnosis._vendor import ensure_agent_search_on_path
from gnosis.core.registry import register
from gnosis.core.schema import Document, Table


@register("parser", "multipage_stitcher")
class MultipageStitcherParser:
    """Detect + merge consecutive-page tables with matching headers.

    Operates on ``document.tables`` after table_normalizer has populated it.
    Replaces spanning Table objects with one merged Table; per-page
    markdown is left untouched (retrieval still references individual pages).
    """

    def parse(self, source: Any, *,
              document: Document | None = None,
              **kwargs) -> Document:
        if document is None:
            raise ValueError("MultipageStitcherParser requires a Document")

        ensure_agent_search_on_path()
        from parsing.html_table_parser import parse_tables_from_markdown
        from parsing.multipage_stitcher import stitch_document

        # Re-parse per-page tables so we can feed stitcher its native format
        # (parsing.schema.Table objects keyed by page)
        page_tables: dict[int, list] = {}
        for pg in document.pages:
            if not pg.markdown:
                continue
            tbls = parse_tables_from_markdown(pg.markdown, page=pg.page_num)
            if tbls:
                page_tables[pg.page_num] = tbls

        if not page_tables:
            return document

        stitched, report = stitch_document(page_tables)
        if not report:
            return document

        # Convert merged Tables (native schema) to framework Tables and attach
        merged_framework_tables: list[Table] = []
        for entry in report:
            pg0 = entry["spans_pages"][0]
            native = stitched[pg0][-1]
            merged_framework_tables.append(Table(
                pages=list(entry["spans_pages"]),
                n_rows=native.n_rows,
                n_cols=native.n_cols,
                flat_headers=native.flat_headers(),
                body_rows=native.body_rows(),
                meta={
                    "merged": True,
                    "rows_from_each_page": entry["rows_from_each_page"],
                    "headers_empty": entry.get("headers_empty", False),
                },
            ))
        document.tables.extend(merged_framework_tables)
        document.meta.setdefault("multipage_report", []).extend(report)
        return document
