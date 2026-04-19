"""Reciprocal Rank Fusion — alternative fusion.

Not used by the smartsearch_v4 preset (which uses weighted_merge), but
provided for users who want RRF-style fusion instead. Formula:
``score = sum(1 / (k + rank))`` across channels.
"""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Hit


@register("ranker", "rrf")
class ReciprocalRankFusion:
    """RRF fusion given channel_scores dict on each Hit."""

    def __init__(self, k: int = 60, top_k: int = 15) -> None:
        self.k = k
        self.top_k = top_k

    def rank(self, hits: list[Hit], *, query: str = "",
             context: dict[str, Any] | None = None) -> list[Hit]:
        # Group by chunk_id so we can see all channel hits for a chunk
        by_chunk: dict[str, list[Hit]] = {}
        for h in hits:
            by_chunk.setdefault(h.chunk_id, []).append(h)

        # Build per-channel rank maps
        channels: dict[str, list[str]] = {}
        for h in hits:
            channels.setdefault(h.channel, []).append(h.chunk_id)
        rank_maps: dict[str, dict[str, int]] = {}
        for ch, chunk_list in channels.items():
            # Stable unique preserve-order, assign 1-based rank
            seen = set()
            ordered = [c for c in chunk_list if not (c in seen or seen.add(c))]
            rank_maps[ch] = {cid: i + 1 for i, cid in enumerate(ordered)}

        out: list[Hit] = []
        for chunk_id, group in by_chunk.items():
            rrf_score = sum(
                1.0 / (self.k + rmap.get(chunk_id, 10_000))
                for rmap in rank_maps.values()
            )
            # Keep the richest Hit (longest text / most meta)
            best = max(group, key=lambda h: len(h.text))
            merged = Hit(
                chunk_id=best.chunk_id,
                doc_id=best.doc_id,
                text=best.text,
                score=rrf_score,
                channel="rrf",
                channel_scores={h.channel: h.score for h in group},
                meta=best.meta,
            )
            out.append(merged)
        out.sort(key=lambda h: -h.score)
        return out[: self.top_k] if self.top_k else out
