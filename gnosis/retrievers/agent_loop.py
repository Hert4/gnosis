"""Agent loop retriever — wraps smartsearch-v4 visual tool-calling fallback.

For scan-only docs where text extraction is incomplete, the engine's
``_agent_loop`` provides 19 visual tools (look_at_page, scan_pages,
parallel_scan, zoom, etc.). This retriever delegates to a pre-built
SmartSearchV4 engine instance.

This is a pragmatic wrap — the 19 tools live in smartsearch-v4 and aren't
extracted into individual plugins. They can be migrated incrementally
later.
"""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Answer, Hit


@register("retriever", "agent_loop")
class AgentLoopRetriever:
    """Delegate to SmartSearchV4 engine's visual agent loop.

    Unlike channel-based retrievers, this doesn't return Hits directly —
    it returns a synthetic Hit containing the full agent-produced answer.
    The synthesizer downstream passes this through as the final Answer.
    """

    def __init__(self, engine=None) -> None:
        self.engine = engine

    def retrieve(self, query: str, *, top_k: int = 30,
                 context: dict[str, Any] | None = None) -> list[Hit]:
        if self.engine is None:
            return []
        chat_history = (context or {}).get("chat_history", [])
        try:
            answer_text = self.engine._agent_loop(
                query, on_event=None, chat_history=chat_history,
            )
        except Exception as e:
            return [Hit(
                chunk_id="agent_loop_error",
                doc_id=(context or {}).get("doc_id", ""),
                text=f"agent_loop error: {e}",
                score=0.0,
                channel="agent_loop",
                meta={"error": str(e)},
            )]
        return [Hit(
            chunk_id="agent_loop_result",
            doc_id=(context or {}).get("doc_id", ""),
            text=answer_text,
            score=1.0,
            channel="agent_loop",
            meta={"full_answer": True},
        )]
