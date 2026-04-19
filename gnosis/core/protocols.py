"""Layer protocols — what each plugin must implement.

Uses PEP 544 Protocol (structural typing) instead of ABC so plugins don't
need to inherit. Any class with matching methods satisfies the protocol.
Enables wrapping third-party code without modifying it.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from gnosis.core.schema import (
    Answer,
    Chunk,
    DocMeta,
    Document,
    Hit,
    Page,
    Table,
)


# ────────────────────────── Parser ──────────────────────────

@runtime_checkable
class ParserProtocol(Protocol):
    """Turn raw input (PDF path, bytes, image) into a Document / Pages.

    Multiple parsers may run in sequence — later parsers enrich earlier ones
    (e.g., OCR adds text, TableNormalizer normalizes HTML tables in-place).
    """

    name: str                                   # unique plugin name

    def parse(self, source: Any, *,
              document: Document | None = None,
              **kwargs) -> Document:
        """Parse `source`. If `document` provided, enrich it; else build fresh."""
        ...


# ────────────────────────── Indexer ──────────────────────────

@runtime_checkable
class IndexerProtocol(Protocol):
    """Build a queryable index from a Document.

    Stateful: indexers keep their index in-memory (or persisted). Retrievers
    query the built index.
    """

    name: str

    def build(self, document: Document, **kwargs) -> None:
        """Build index from Document. Replaces any previous state."""
        ...

    def update(self, document: Document, **kwargs) -> None:
        """Incremental update — used when new pages arrive (tier 1/2)."""
        ...


# ────────────────────────── Retriever ──────────────────────────

@runtime_checkable
class ChannelProtocol(Protocol):
    """One retrieval signal producer — e.g., BM25, tree-nav, entity-hop."""

    name: str

    def search(self, query: str, *, top_k: int = 20,
               context: dict[str, Any] | None = None) -> list[Hit]:
        """Return a ranked list of Hits for `query`."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Top-level retriever — may compose multiple channels."""

    name: str

    def retrieve(self, query: str, *, top_k: int = 30,
                 context: dict[str, Any] | None = None) -> list[Hit]:
        """Return merged/ranked Hits across all channels."""
        ...


# ────────────────────────── Ranker ──────────────────────────

@runtime_checkable
class RankerProtocol(Protocol):
    """Rerank / fuse Hits and optionally truncate to fit a context budget."""

    name: str

    def rank(self, hits: list[Hit], *, query: str = "",
             context: dict[str, Any] | None = None) -> list[Hit]:
        """Return reordered Hits (may be a subset)."""
        ...


# ────────────────────────── Synthesizer ──────────────────────────

@runtime_checkable
class SynthesizerProtocol(Protocol):
    """Produce a final Answer from a query + ranked Hits."""

    name: str

    def synthesize(self, query: str, hits: list[Hit], *,
                   chat_history: list[dict[str, str]] | None = None,
                   context: dict[str, Any] | None = None) -> Answer:
        """Generate the final Answer."""
        ...


# ────────────────────────── Router ──────────────────────────

@runtime_checkable
class RouterProtocol(Protocol):
    """Select relevant document(s) from a corpus and delegate to their pipelines."""

    name: str

    def add_document(self, doc_meta: DocMeta, pipeline: Any) -> None:
        """Register a document + its per-doc Pipeline."""
        ...

    def route(self, query: str, *,
              context: dict[str, Any] | None = None) -> list[str]:
        """Return ordered list of doc_ids to query."""
        ...

    def query(self, query: str, *,
              chat_history: list[dict[str, str]] | None = None,
              context: dict[str, Any] | None = None) -> Answer:
        """Route + query selected doc(s) + synthesize combined Answer."""
        ...
