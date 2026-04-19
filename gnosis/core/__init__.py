"""Core framework — protocols, schema, pipeline, registry."""

from gnosis.core.events import Event, EventEmitter
from gnosis.core.pipeline import Pipeline, PipelineBuilder
from gnosis.core.protocols import (
    IndexerProtocol,
    ParserProtocol,
    RankerProtocol,
    RetrieverProtocol,
    RouterProtocol,
    SynthesizerProtocol,
)
from gnosis.core.registry import PluginRegistry, register
from gnosis.core.schema import (
    Answer,
    Chunk,
    DocMeta,
    Document,
    Hit,
    Page,
    Table,
)

__all__ = [
    "Event", "EventEmitter",
    "Pipeline", "PipelineBuilder",
    "ParserProtocol", "IndexerProtocol", "RetrieverProtocol",
    "RankerProtocol", "SynthesizerProtocol", "RouterProtocol",
    "PluginRegistry", "register",
    "Answer", "Chunk", "DocMeta", "Document", "Hit", "Page", "Table",
]
