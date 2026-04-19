"""Smart truncate ranker — limits total context chars without cutting mid-tag.

Preserves full Hits that fit the budget; drops any Hit that would push the
cumulative char count over ``max_chars``. This prevents the synthesizer
from being fed a context that will later be hard-cut mid-row by downstream
truncation.

For in-place text truncation, use ``smart_truncate`` utility directly
(see ``gnosis.rankers.smart_truncate_ranker.smart_truncate``).
"""

from __future__ import annotations

from typing import Any

from gnosis.core.registry import register
from gnosis.core.schema import Hit


def smart_truncate(text: str, max_chars: int) -> str:
    """Cut ``text`` at a safe boundary (page sep → paragraph → line).

    Identical to ``smartsearch/engine.py:smart_truncate``.
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    lower = int(max_chars * 0.85)
    for sep in ("\n\n---\n\n", "\n\n", "\n"):
        idx = window.rfind(sep)
        if idx >= lower:
            return window[:idx] + f"\n\n... (truncated, {len(text) - idx} chars)"
    return window + "\n... (truncated)"


@register("ranker", "smart_truncate")
class SmartTruncateRanker:
    """Drop Hits past a cumulative char budget (preserves hit boundaries)."""

    def __init__(self, max_chars: int = 25000) -> None:
        self.max_chars = max_chars

    def rank(self, hits: list[Hit], *, query: str = "",
             context: dict[str, Any] | None = None) -> list[Hit]:
        kept: list[Hit] = []
        used = 0
        for h in hits:
            add = len(h.text) + 20   # +20 accounts for separator chars
            if used + add > self.max_chars:
                break
            kept.append(h)
            used += add
        return kept
