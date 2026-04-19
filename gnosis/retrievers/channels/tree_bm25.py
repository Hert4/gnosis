"""Tree BM25 channel — tree_index.search_tree with title-overlap boost.

Extracted from engine.py:695-710. For each BM25-matched node, the score is
boosted by title-token overlap and distributed across the node's pages
with sqrt decay.
"""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Hit


@register("channel", "tree_bm25")
class TreeBm25Channel:
    """BM25 over tree nodes, emit one Hit per covered page."""

    def __init__(
        self,
        tree_indexer=None,
        top_k_nodes: int = 3,
        title_overlap_bonus: float = 0.5,
    ) -> None:
        self.tree_indexer = tree_indexer
        self.top_k_nodes = top_k_nodes
        self.title_overlap_bonus = title_overlap_bonus

    def search(self, query: str, *, top_k: int = 20,
               context: dict[str, Any] | None = None) -> list[Hit]:
        if not self.tree_indexer or not self.tree_indexer.is_ready:
            return []

        tree = self.tree_indexer.raw
        results = tree.search_tree(query, top_k=self.top_k_nodes)
        if not results:
            return []

        from nanoindex.utils.tree_ops import find_node

        q_tokens = {t for t in query.lower().split() if len(t) >= 3}
        doc_id = (context or {}).get("doc_id", "")
        page_texts = (context or {}).get("page_texts", {})

        hits_by_page: dict[int, Hit] = {}
        for nid, title, score in results:
            node = find_node(tree.tree.structure, nid)
            if not node or not node.start_index:
                continue

            title_lower = (title or "").lower()
            overlap = sum(1 for t in q_tokens if t in title_lower)
            boosted = score * (1 + self.title_overlap_bonus * overlap)

            span = (node.end_index or node.start_index) - node.start_index + 1
            per_page = boosted / max(1.0, span ** 0.5)

            for p in range(node.start_index, (node.end_index or node.start_index) + 1):
                if p in hits_by_page:
                    # Keep best score
                    if per_page > hits_by_page[p].score:
                        hits_by_page[p].score = per_page
                else:
                    hits_by_page[p] = Hit(
                        chunk_id=f"page_{p}",
                        doc_id=doc_id,
                        text=page_texts.get(p, ""),
                        score=per_page,
                        channel="tree_bm25",
                        meta={"page": p, "node_id": nid, "node_title": title},
                    )

        ordered = sorted(hits_by_page.values(), key=lambda h: -h.score)
        return ordered[:top_k] if top_k else ordered
