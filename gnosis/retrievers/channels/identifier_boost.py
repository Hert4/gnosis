"""Identifier boost channel — match numeric/code identifiers in node titles.

Extracted from engine.py:712-736. When the query contains digit-bearing
tokens (e.g. "515", "99/2025", "3.2"), boost tree nodes whose titles also
contain those tokens.
"""

from __future__ import annotations

import re
from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Hit

_ID_RE = re.compile(r"[A-Za-z]*\d+(?:[./\-]\d+)*[A-Za-z]*")


@register("channel", "identifier_boost")
class IdentifierBoostChannel:
    """Boost nodes whose title contains identifier tokens from the query."""

    def __init__(self, tree_indexer=None, boost: float = 25.0) -> None:
        self.tree_indexer = tree_indexer
        self.boost = boost

    def search(self, query: str, *, top_k: int = 20,
               context: dict[str, Any] | None = None) -> list[Hit]:
        if not self.tree_indexer or not self.tree_indexer.is_ready:
            return []

        # Extract identifier-like tokens from query (must contain a digit)
        q_ids = {
            tok for tok in _ID_RE.findall(query)
            if any(c.isdigit() for c in tok) and len(tok) >= 2
        }
        if not q_ids:
            return []

        from nanoindex.utils.tree_ops import iter_nodes
        tree = self.tree_indexer.raw
        doc_id = (context or {}).get("doc_id", "")
        page_texts = (context or {}).get("page_texts", {})

        hits_by_page: dict[int, Hit] = {}
        for node in iter_nodes(tree.tree.structure):
            if not node.title or not node.start_index:
                continue
            if not any(tok in node.title for tok in q_ids):
                continue
            span = (node.end_index or node.start_index) - node.start_index + 1
            per_page = self.boost / max(1.0, span ** 0.5)
            for p in range(node.start_index, (node.end_index or node.start_index) + 1):
                current = hits_by_page.get(p)
                if current is None or per_page > current.score:
                    hits_by_page[p] = Hit(
                        chunk_id=f"page_{p}",
                        doc_id=doc_id,
                        text=page_texts.get(p, ""),
                        score=per_page,
                        channel="identifier_boost",
                        meta={"page": p, "matched_ids": sorted(q_ids)},
                    )

        ordered = sorted(hits_by_page.values(), key=lambda h: -h.score)
        return ordered[:top_k] if top_k else ordered
