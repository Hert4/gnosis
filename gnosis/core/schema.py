"""Common types flowing between framework layers.

These are intentionally minimal — each layer may carry additional metadata
in `meta` dicts. All types are frozen dataclasses so they can be hashed and
passed safely across threads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Page:
    """A single page of a document — raw + optional normalized content."""
    page_num: int
    raw_text: str = ""
    markdown: str = ""
    page_type: str = "text"          # text | scan | table | mixed
    has_text_layer: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Table:
    """Parsed table spanning one or more pages."""
    pages: list[int]
    n_rows: int
    n_cols: int
    flat_headers: list[str] = field(default_factory=list)
    body_rows: list[list[str]] = field(default_factory=list)
    title: str = ""
    caption: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Parsed document — output of Parse layer, input to Index layer."""
    doc_id: str
    name: str
    total_pages: int
    pages: list[Page] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit — page, section, paragraph, cell, etc."""
    chunk_id: str
    doc_id: str
    text: str
    source_type: str = "page"        # page | section | table_row | entity
    source_ref: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    """A retrieval candidate — produced by Retrievers, consumed by Rankers."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    channel: str = ""                # which channel produced this hit
    channel_scores: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    """Final synthesis output — returned to caller."""
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    used_chunks: list[Hit] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocMeta:
    """Lightweight document metadata for router layer.

    Does NOT include full text — only summary + headings + top entities so a
    router can pick which document(s) to query without loading full content.
    """
    doc_id: str
    name: str
    title: str = ""
    summary: str = ""
    headings: list[str] = field(default_factory=list)
    top_entities: list[str] = field(default_factory=list)
    total_pages: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
