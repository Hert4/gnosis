"""LLM tree navigation channel.

Sends the document outline + question to an LLM, which returns 5-10 section
IDs. Maps each selected section to its page range, emits one Hit per
covered page at a fixed high score (default 20.0).

Extracted from ``smartsearch/engine.py:_chatbot_query`` step 1 (l.654-692).
Preserves original prompt + scoring.
"""

from __future__ import annotations

import json
import re
from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Hit

_SYS = (
    "You are a document retrieval expert. Given a document outline and a "
    "question, select the 5-10 most relevant section IDs. Prefer specific "
    "child sections over broad parent sections. Return ONLY a JSON array of IDs."
)
_USER = "Question: {q}\n\nDocument outline:\n{outline}\n\nJSON array of relevant section IDs:"


@register("channel", "llm_tree_nav")
class LlmTreeNavChannel:
    """LLM picks section IDs from tree outline."""

    def __init__(
        self,
        tree_indexer=None,                      # TreeIndexIndexer instance
        llm_client=None,                        # OpenAI-compatible client
        llm_model: str | None = None,
        boost: float = 20.0,
        max_outline_chars: int = 30000,
        outline_depth: int = 2,
    ) -> None:
        self.tree_indexer = tree_indexer
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.boost = boost
        self.max_outline_chars = max_outline_chars
        self.outline_depth = outline_depth

    def search(self, query: str, *, top_k: int = 20,
               context: dict[str, Any] | None = None) -> list[Hit]:
        if not self.tree_indexer or not self.tree_indexer.is_ready:
            return []
        if not self.llm_client or not self.llm_model:
            return []

        tree = self.tree_indexer.raw
        outline = tree.get_outline(max_depth=self.outline_depth)
        if len(outline) > self.max_outline_chars:
            outline = outline[:self.max_outline_chars] + "\n... (truncated)"

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": _USER.format(q=query, outline=outline)},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            return []

        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            nav_ids = json.loads(m.group())
        except (ValueError, json.JSONDecodeError):
            return []

        from nanoindex.utils.tree_ops import find_node

        doc_id = (context or {}).get("doc_id", "")
        hits: list[Hit] = []
        seen_pages: set[int] = set()
        for nid in nav_ids:
            node = find_node(tree.tree.structure, str(nid))
            if not node or not node.start_index:
                continue
            end = node.end_index or node.start_index
            for p in range(node.start_index, end + 1):
                if p in seen_pages:
                    continue
                seen_pages.add(p)
                hits.append(Hit(
                    chunk_id=f"page_{p}",
                    doc_id=doc_id,
                    text=(context or {}).get("page_texts", {}).get(p, ""),
                    score=self.boost,
                    channel="llm_tree_nav",
                    meta={"page": p, "section_id": str(nid)},
                ))
        return hits[:top_k] if top_k else hits
