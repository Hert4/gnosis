"""Wraps parsing/table_normalizer.py — rewrites HTML tables as KV/pipe markdown."""

from __future__ import annotations

from typing import Any

from gnosis._vendor import ensure_agent_search_on_path
from gnosis.core.registry import register
from gnosis.core.schema import Document, Table


@register("parser", "table_normalizer")
class TableNormalizerParser:
    """Normalize HTML tables inside each page's markdown.

    Also extracts structured Table objects and appends them to
    `document.tables`.
    """

    def parse(self, source: Any, *,
              document: Document | None = None,
              **kwargs) -> Document:
        if document is None:
            raise ValueError("TableNormalizerParser requires an input Document")

        ensure_agent_search_on_path()
        from parsing.html_table_parser import parse_tables_from_markdown
        from parsing.table_normalizer import normalize_tables_in_markdown

        for pg in document.pages:
            if not pg.markdown:
                continue
            # Extract structured tables first (before rewriting)
            tables = parse_tables_from_markdown(pg.markdown, page=pg.page_num)
            for t in tables:
                document.tables.append(Table(
                    pages=[pg.page_num],
                    n_rows=t.n_rows,
                    n_cols=t.n_cols,
                    flat_headers=t.flat_headers(),
                    body_rows=t.body_rows(),
                    title=t.title,
                    caption=t.caption,
                    meta={"raw_html": t.raw_html},
                ))
            # Rewrite markdown in place
            pg.markdown = normalize_tables_in_markdown(pg.markdown, page=pg.page_num)
            pg.meta["table_normalized"] = True

        return document
