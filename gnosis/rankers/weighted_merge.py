"""Weighted merge ranker — mirrors the 0.4 merged + 0.6 BM25 rerank in
``smartsearch/engine.py:_chatbot_query`` line 794-802.

This is a NO-OP in practice because HybridChatbotRetriever already applies
this formula. Kept as a standalone ranker so users can compose it from
PipelineBuilder without relying on the hybrid retriever (e.g., if they
feed hits from elsewhere).
"""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Hit


@register("ranker", "weighted_merge")
class WeightedMergeRanker:
    """Sort by ``score`` descending, keep top-N."""

    def __init__(self, top_k: int = 15) -> None:
        self.top_k = top_k

    def rank(self, hits: list[Hit], *, query: str = "",
             context: dict[str, Any] | None = None) -> list[Hit]:
        ordered = sorted(hits, key=lambda h: -h.score)
        return ordered[: self.top_k] if self.top_k else ordered
