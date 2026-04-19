"""LLM-flat router — zero-embed, LLM picks from a list of DocMeta.

Best for small corpora (<= ~50 docs) where the combined DocMeta (summary +
headings + top entities) fits in one LLM prompt. Single LLM call per query
to pick doc(s); then delegates to each picked doc's Pipeline.

For larger corpora, use summary_embed_router (not yet implemented) or
hierarchical_router (C2S-style).
"""

from __future__ import annotations

import json
import re
from typing import Any

from gnosis.core.pipeline import Pipeline
from gnosis.core.registry import register
from gnosis.core.schema import Answer, DocMeta, Hit


_ROUTER_SYS = (
    "You are a document router. Given a user question and a list of documents, "
    "pick the most relevant doc_ids (1-3). Return ONLY a JSON array of doc_id strings. "
    "If none apply, return []."
)


@register("router", "llm_flat")
class LLMFlatRouter:
    """Corpus-level router: LLM-picks docs, then queries their Pipelines."""

    def __init__(
        self,
        *,
        llm_client=None,
        llm_model: str | None = None,
        max_picks: int = 3,
        max_prompt_chars: int = 25_000,
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.max_picks = max_picks
        self.max_prompt_chars = max_prompt_chars
        self._docs: dict[str, tuple[DocMeta, Pipeline]] = {}

    # ─────────────── Registration ───────────────

    def add_document(self, doc_meta: DocMeta, pipeline: Pipeline) -> None:
        self._docs[doc_meta.doc_id] = (doc_meta, pipeline)

    def remove_document(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

    @property
    def docs(self) -> dict[str, DocMeta]:
        return {did: dm for did, (dm, _p) in self._docs.items()}

    # ─────────────── Routing ───────────────

    def _build_doc_list_prompt(self) -> str:
        """Compact list of (doc_id, title, summary, top entities) for LLM."""
        lines: list[str] = []
        for did, (dm, _p) in self._docs.items():
            entities = ", ".join(dm.top_entities[:8]) if dm.top_entities else ""
            lines.append(
                f"- id={did} title=\"{dm.title or dm.name}\"\n"
                f"  summary: {(dm.summary or '')[:300]}\n"
                f"  entities: {entities}\n"
                f"  pages: {dm.total_pages}"
            )
        text = "\n".join(lines)
        if len(text) > self.max_prompt_chars:
            text = text[: self.max_prompt_chars] + "\n... (truncated)"
        return text

    def route(self, query: str, *,
              context: dict[str, Any] | None = None) -> list[str]:
        """Ask LLM which doc_ids are relevant. Return ordered list."""
        if not self._docs:
            return []
        if not self.llm_client or not self.llm_model:
            # Fallback: return all docs
            return list(self._docs.keys())[: self.max_picks]

        doc_list = self._build_doc_list_prompt()
        prompt = (
            f"Question: {query}\n\n"
            f"Available documents:\n{doc_list}\n\n"
            f"Pick up to {self.max_picks} most relevant doc_ids.\n"
            f"Output JSON array, e.g. [\"id1\", \"id2\"]."
        )
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": _ROUTER_SYS},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            return list(self._docs.keys())[: self.max_picks]

        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            ids = json.loads(m.group())
        except (ValueError, json.JSONDecodeError):
            return []

        # Filter to known doc_ids; preserve LLM's order
        ordered = [did for did in ids if did in self._docs]
        return ordered[: self.max_picks]

    # ─────────────── Query ───────────────

    def query(self, query: str, *,
              chat_history: list[dict[str, str]] | None = None,
              context: dict[str, Any] | None = None) -> Answer:
        picked = self.route(query, context=context)
        if not picked:
            return Answer(text="Không tìm thấy tài liệu phù hợp.",
                          meta={"router": "llm_flat", "picked": []})

        # Query each picked doc; collect answers
        per_doc_answers: list[dict[str, Any]] = []
        all_used_hits: list[Hit] = []
        for did in picked:
            _dm, pipeline = self._docs[did]
            try:
                ans = pipeline.query(query, chat_history=chat_history)
                per_doc_answers.append({
                    "doc_id": did,
                    "text": ans.text,
                    "citations": ans.citations,
                })
                all_used_hits.extend(ans.used_chunks)
            except Exception as e:
                per_doc_answers.append({
                    "doc_id": did,
                    "text": f"[error in {did}: {e}]",
                    "citations": [],
                })

        # If only 1 picked, return its answer directly; else concat labeled
        if len(per_doc_answers) == 1:
            first = per_doc_answers[0]
            return Answer(
                text=first["text"],
                citations=first["citations"],
                used_chunks=all_used_hits,
                meta={"router": "llm_flat", "picked": picked},
            )

        combined = "\n\n".join(
            f"## Từ tài liệu {a['doc_id']}\n{a['text']}"
            for a in per_doc_answers
        )
        return Answer(
            text=combined,
            citations=[c for a in per_doc_answers for c in a["citations"]],
            used_chunks=all_used_hits,
            meta={"router": "llm_flat", "picked": picked},
        )
