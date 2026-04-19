"""Entity graph 2-hop channel.

Extracted from engine.py:738-754. Extract query entities, match them in
the graph, expand 2 hops, map matched nodes to pages.
"""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Hit


@register("channel", "entity_2hop")
class Entity2HopChannel:
    """Entity-graph based retrieval."""

    def __init__(
        self,
        entity_graph_indexer=None,
        tree_indexer=None,
        hops: int = 2,
        cap_nodes: int = 12,
        seed_boost: float = 8.0,
        hop_boost: float = 3.0,
    ) -> None:
        self.entity_graph_indexer = entity_graph_indexer
        self.tree_indexer = tree_indexer
        self.hops = hops
        self.cap_nodes = cap_nodes
        self.seed_boost = seed_boost
        self.hop_boost = hop_boost

    def search(self, query: str, *, top_k: int = 20,
               context: dict[str, Any] | None = None) -> list[Hit]:
        eg = self.entity_graph_indexer
        ti = self.tree_indexer
        if not eg or not eg.is_ready:
            return []
        if not ti or not ti.is_ready:
            return []

        graph = eg.raw
        tree = ti.raw

        matched = graph.find_nodes(query)
        if not matched:
            return []

        seed_ids = {nid for nid, _ in matched}
        expanded = graph.expand_nodes(seed_ids, hops=self.hops)
        ordered = list(seed_ids) + [n for n in (seed_ids | expanded) if n not in seed_ids]
        ordered = ordered[: self.cap_nodes]

        from nanoindex.utils.tree_ops import find_node

        doc_id = (context or {}).get("doc_id", "")
        page_texts = (context or {}).get("page_texts", {})

        hits_by_page: dict[int, Hit] = {}
        for nid in ordered:
            node = find_node(tree.tree.structure, nid)
            if not node or not node.start_index:
                continue
            is_seed = nid in seed_ids
            sc = self.seed_boost if is_seed else self.hop_boost
            for p in range(node.start_index, (node.end_index or node.start_index) + 1):
                current = hits_by_page.get(p)
                if current is None or sc > current.score:
                    hits_by_page[p] = Hit(
                        chunk_id=f"page_{p}",
                        doc_id=doc_id,
                        text=page_texts.get(p, ""),
                        score=sc,
                        channel="entity_2hop",
                        meta={"page": p, "node_id": nid, "is_seed": is_seed},
                    )
        ordered_hits = sorted(hits_by_page.values(), key=lambda h: -h.score)
        return ordered_hits[:top_k] if top_k else ordered_hits
