"""Chatbot-style synthesizer — single LLM call with Vietnamese system prompt.

Mirrors the synthesis portion of ``smartsearch/engine.py:_chatbot_query``
(l.907-1050). Preserves:
- Vietnamese system prompt emphasizing grounding, source citation
- Chat history injection (last 8 turns)
- Smart truncate on assembled context
- Optional reflection step (off by default — orchestrated via extra ranker
  pass in the existing codebase)
"""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Answer, Hit
from gnosis.rankers.smart_truncate_ranker import smart_truncate

_DEFAULT_SYSTEM = (
    "Bạn là trợ lý đã đọc và hiểu toàn bộ tài liệu. Trả lời DỰA HOÀN TOÀN vào "
    "context.\n\n"
    "## Nguyên tắc\n"
    "1. Chỉ dùng thông tin trong context. Không suy diễn ngoài phạm vi tài liệu.\n"
    "2. Trích dẫn trang khi có — '(p.X)' hoặc '(trang X)'.\n"
    "3. Nếu không tìm thấy, nói rõ 'Không tìm thấy trong tài liệu'.\n"
    "4. Giữ thuật ngữ gốc (mã TK, tên biểu mẫu, mã biểu).\n"
    "5. Trả lời ngắn gọn, đầy đủ.\n"
)


@register("synthesizer", "chatbot_llm")
class ChatbotLLMSynthesizer:
    """One-shot LLM call with Vietnamese system prompt + retrieved context."""

    def __init__(
        self,
        llm_client=None,
        llm_model: str | None = None,
        system_prompt: str = _DEFAULT_SYSTEM,
        max_context_chars: int = 25000,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        history_turns: int = 8,
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history_turns = history_turns

    def _build_context(self, hits: list[Hit]) -> str:
        parts: list[str] = []
        for h in hits:
            page = h.meta.get("page")
            header = f"[Trang {page}]" if page else f"[Chunk {h.chunk_id}]"
            parts.append(f"{header}\n{h.text.strip()}")
        ctx = "\n\n---\n\n".join(parts)
        return smart_truncate(ctx, self.max_context_chars)

    def synthesize(
        self,
        query: str,
        hits: list[Hit],
        *,
        chat_history: list[dict[str, str]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> Answer:
        if not self.llm_client or not self.llm_model:
            return Answer(
                text="[synthesizer not configured — missing llm_client/model]",
                used_chunks=hits,
            )

        ctx_str = self._build_context(hits)

        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if chat_history:
            recent = [
                m for m in chat_history[-self.history_turns * 2:]
                if m.get("role") in ("user", "assistant")
            ]
            messages.extend(recent)
        messages.append({"role": "user", "content": (
            f"=== NỘI DUNG TÀI LIỆU ===\n{ctx_str}\n=== HẾT ===\n\n"
            f"Câu hỏi: {query}\nTrả lời:"
        )})

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return Answer(text=f"[synthesis error: {e}]", used_chunks=hits)

        citations = [
            {"page": h.meta.get("page"), "chunk_id": h.chunk_id,
             "channel": h.channel, "score": h.score}
            for h in hits if h.meta.get("page")
        ]
        return Answer(
            text=answer,
            citations=citations,
            used_chunks=hits,
            meta={"context_chars": len(ctx_str)},
        )
