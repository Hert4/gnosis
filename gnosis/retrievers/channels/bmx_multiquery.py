"""BMX multi-query channel.

Extracted from engine.py:756-791. Asks an LLM for 3 query variants, runs
each via PageBM25, sums scores (original boosted 2.0, variants 0.4).
"""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Hit

_EXPANSION_SYS = (
    "You generate search queries for document retrieval. "
    "Output exactly 3 queries, one per line."
)
_EXPANSION_USER = (
    "Question: {q}\n\n"
    "Generate 3 different search queries to find this in a Vietnamese "
    "accounting/legal document.\n"
    "- Query 1: Include specific account code (TK xxx) if applicable\n"
    "- Query 2: Use different terminology/synonyms\n"
    "- Query 3: Focus on the key concept only\n"
    "Output 3 lines only:"
)


@register("channel", "bmx_multiquery")
class BmxMultiQueryChannel:
    """BMX page search + LLM-generated query expansion."""

    def __init__(
        self,
        bm25_indexer=None,
        llm_client=None,
        llm_model: str | None = None,
        original_boost: float = 2.0,
        expansion_boost: float = 0.4,
        original_top_k: int = 15,
        expansion_top_k: int = 8,
    ) -> None:
        self.bm25_indexer = bm25_indexer
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.original_boost = original_boost
        self.expansion_boost = expansion_boost
        self.original_top_k = original_top_k
        self.expansion_top_k = expansion_top_k

    def _generate_expansions(self, query: str) -> list[str]:
        if not self.llm_client or not self.llm_model:
            return []
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": _EXPANSION_SYS},
                    {"role": "user", "content": _EXPANSION_USER.format(q=query)},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            raw = (resp.choices[0].message.content or "").strip().splitlines()
        except Exception:
            return []
        return [
            q.strip().lstrip("0123456789.-) ").strip('"').strip("'")
            for q in raw if len(q.strip()) > 5
        ]

    def search(self, query: str, *, top_k: int = 20,
               context: dict[str, Any] | None = None) -> list[Hit]:
        if not self.bm25_indexer:
            return []

        scores_by_page: dict[int, float] = {}
        # Original query
        for p, sc in self.bm25_indexer.query(query, top_k=self.original_top_k):
            scores_by_page[p] = scores_by_page.get(p, 0.0) + sc * self.original_boost

        # LLM-generated expansions
        for exp in self._generate_expansions(query):
            for p, sc in self.bm25_indexer.query(exp, top_k=self.expansion_top_k):
                scores_by_page[p] = scores_by_page.get(p, 0.0) + sc * self.expansion_boost

        if not scores_by_page:
            return []

        doc_id = (context or {}).get("doc_id", "")
        page_texts = (context or {}).get("page_texts", {})
        hits = [
            Hit(
                chunk_id=f"page_{p}",
                doc_id=doc_id,
                text=page_texts.get(p, ""),
                score=sc,
                channel="bmx_multiquery",
                meta={"page": p},
            )
            for p, sc in sorted(scores_by_page.items(), key=lambda x: -x[1])
        ]
        return hits[:top_k] if top_k else hits
