"""Event emitter for pipeline progress (SSE-compatible).

Pipelines emit events at key points (tool_call, tool_result, synthesizing,
readiness_change). Consumers can attach listeners to stream UI updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    type: str                                   # tool_call | tool_result | synthesizing | readiness | error | log
    step: int | None = None
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.step is not None:
            out["step"] = self.step
        if self.tool is not None:
            out["tool"] = self.tool
        if self.args:
            out["args"] = self.args
        if self.summary is not None:
            out["summary"] = self.summary
        if self.data:
            out.update(self.data)
        return out


class EventEmitter:
    """Lightweight multi-listener emitter. Listeners are called synchronously."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[Event], None]] = []

    def on(self, listener: Callable[[Event], None]) -> None:
        self._listeners.append(listener)

    def emit(self, event: Event | dict[str, Any]) -> None:
        if isinstance(event, dict):
            event = Event(
                type=event.pop("type", "log"),
                step=event.pop("step", None),
                tool=event.pop("tool", None),
                args=event.pop("args", {}),
                summary=event.pop("summary", None),
                data=event,
            )
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                # Listeners never break the pipeline
                pass

    def clear(self) -> None:
        self._listeners = []
