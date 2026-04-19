"""Hybrid chatbot retriever — composes all channels, rerank, neighbor expand.

Replicates the combine/rerank logic from engine.py:_chatbot_query
(lines 794-820): accumulate channel scores per page, then final rerank as
``0.4 * merged + 0.6 * pure_bm25_on_original_query``, then expand ±1
neighbor pages.
"""

from __future__ import annotations

from typing import Any

from gnosis.core.protocols import ChannelProtocol
from gnosis.core.registry import register
from gnosis.core.schema import Hit


@register("retriever", "hybrid_chatbot")
class HybridChatbotRetriever:
    """Accumulate channel scores → rerank → neighbor expand."""

    def __init__(
        self,
        channels: list[ChannelProtocol] | None = None,
        bm25_indexer=None,
        merged_weight: float = 0.4,
        bm25_weight: float = 0.6,
        rerank_pool: int = 30,
        final_top_k: int = 15,
        neighbor_radius: int = 1,
        total_pages: int = 0,
    ) -> None:
        self.channels: list[ChannelProtocol] = list(channels or [])
        self.bm25_indexer = bm25_indexer
        self.merged_weight = merged_weight
        self.bm25_weight = bm25_weight
        self.rerank_pool = rerank_pool
        self.final_top_k = final_top_k
        self.neighbor_radius = neighbor_radius
        self.total_pages = total_pages

    def retrieve(self, query: str, *, top_k: int = 30,
                 context: dict[str, Any] | None = None) -> list[Hit]:
        ctx = context or {}

        # Gather from all channels
        accum: dict[str, float] = {}
        channel_scores: dict[str, dict[str, float]] = {}
        page_texts = ctx.get("page_texts", {})
        doc_id = ctx.get("doc_id", "")

        for ch in self.channels:
            try:
                hits = ch.search(query, top_k=self.rerank_pool * 2, context=ctx)
            except Exception:
                continue
            for h in hits:
                accum[h.chunk_id] = accum.get(h.chunk_id, 0.0) + h.score
                channel_scores.setdefault(h.chunk_id, {})[ch.name] = h.score

        if not accum:
            return []

        # Rerank: merged vs pure BM25 on ORIGINAL query
        bm25_pure: dict[int, float] = {}
        if self.bm25_indexer is not None:
            bm25_pure = dict(self.bm25_indexer.query(query, top_k=100))

        top_candidates = sorted(accum.items(), key=lambda x: -x[1])[: self.rerank_pool]

        reranked: list[Hit] = []
        for chunk_id, merged in top_candidates:
            page = _page_from_chunk(chunk_id)
            bm = bm25_pure.get(page, 0.0) if page else 0.0
            final = self.merged_weight * merged + self.bm25_weight * bm
            reranked.append(Hit(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=page_texts.get(page, "") if page else "",
                score=final,
                channel="hybrid_chatbot",
                channel_scores=channel_scores.get(chunk_id, {}),
                meta={"page": page, "merged": merged, "bm25_pure": bm},
            ))
        reranked.sort(key=lambda h: -h.score)
        top = reranked[: self.final_top_k]

        # Neighbor expansion (±N)
        if self.neighbor_radius > 0:
            ranked_pages = {h.meta.get("page") for h in top if h.meta.get("page")}
            to_add: set[int] = set()
            for h in top:
                p = h.meta.get("page")
                if not p:
                    continue
                for delta in range(-self.neighbor_radius, self.neighbor_radius + 1):
                    if delta == 0:
                        continue
                    nb = p + delta
                    if nb < 1:
                        continue
                    if self.total_pages and nb > self.total_pages:
                        continue
                    if nb in ranked_pages:
                        continue
                    to_add.add(nb)
            for nb in sorted(to_add):
                top.append(Hit(
                    chunk_id=f"page_{nb}",
                    doc_id=doc_id,
                    text=page_texts.get(nb, ""),
                    score=0.1,  # tiny — just to bring neighbor into context
                    channel="neighbor",
                    meta={"page": nb, "neighbor_of_ranked": True},
                ))

        if top_k is not None and top_k > 0:
            return top[:top_k]
        return top


def _page_from_chunk(chunk_id: str) -> int | None:
    if chunk_id.startswith("page_"):
        try:
            return int(chunk_id.split("_", 1)[1])
        except ValueError:
            return None
    return None
