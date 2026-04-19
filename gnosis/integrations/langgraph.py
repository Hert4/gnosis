"""LangGraph adapters — node factories for graph-based agents.

Usage:
    from gnosis.integrations.langgraph import make_retrieval_node, make_parse_node

    graph.add_node("retrieve", make_retrieval_node(my_retriever))
    graph.add_node("parse", make_parse_node(my_parser))

Nodes are plain callables (state -> state) so they work with LangGraph
StateGraph without additional dependencies at framework-core level.
"""

from __future__ import annotations

from typing import Any, Callable

from gnosis.core.protocols import (
    ParserProtocol,
    RetrieverProtocol,
    SynthesizerProtocol,
)


def make_parse_node(parser: ParserProtocol,
                    source_key: str = "source",
                    output_key: str = "document") -> Callable[[dict], dict]:
    """Create a LangGraph-compatible node that runs a Parser.

    Reads ``state[source_key]``, writes parsed Document to ``state[output_key]``.
    """
    def _node(state: dict[str, Any]) -> dict[str, Any]:
        doc = parser.parse(state[source_key])
        return {**state, output_key: doc}
    return _node


def make_retrieval_node(retriever: RetrieverProtocol,
                        query_key: str = "query",
                        output_key: str = "hits",
                        context_key: str | None = None) -> Callable[[dict], dict]:
    """Node that calls a Retriever.

    Reads ``state[query_key]``, optionally pulls ``state[context_key]`` for
    context (else empty), writes hits to ``state[output_key]``.
    """
    def _node(state: dict[str, Any]) -> dict[str, Any]:
        ctx = state.get(context_key, {}) if context_key else {}
        hits = retriever.retrieve(state[query_key], context=ctx)
        return {**state, output_key: hits}
    return _node


def make_synthesis_node(synth: SynthesizerProtocol,
                        query_key: str = "query",
                        hits_key: str = "hits",
                        output_key: str = "answer",
                        history_key: str | None = "chat_history") -> Callable[[dict], dict]:
    """Node that calls a Synthesizer and stores the Answer."""
    def _node(state: dict[str, Any]) -> dict[str, Any]:
        history = state.get(history_key, []) if history_key else []
        ans = synth.synthesize(state[query_key], state[hits_key], chat_history=history)
        return {**state, output_key: ans}
    return _node
