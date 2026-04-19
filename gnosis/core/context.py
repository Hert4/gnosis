"""Request context — shared state across pipeline stages within one query.

Stages write/read here so downstream stages know what earlier stages did
(e.g., retriever stores top-K page numbers, synthesizer reads them for
citation generation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:
    query: str = ""
    chat_history: list[dict[str, str]] = field(default_factory=list)
    doc_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)
