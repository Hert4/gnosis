"""Retrival Framework — modular retrieval/QA framework.

Layered architecture: Parsers → Indexers → Retrievers → Rankers → Synthesizers.
Optional outer layer: Routers for multi-document selection.

Each layer defines a Protocol; concrete plugins register via @register decorator.
Pipelines compose layers via PipelineBuilder. Integrations with LangChain /
LangGraph available as optional extras.
"""

from gnosis.core.pipeline import Pipeline, PipelineBuilder
from gnosis.core.registry import PluginRegistry, register
from gnosis.core.schema import Answer, Chunk, Document, DocMeta, Hit, Page

__version__ = "0.1.0"
__all__ = [
    "Pipeline", "PipelineBuilder",
    "PluginRegistry", "register",
    "Answer", "Chunk", "Document", "DocMeta", "Hit", "Page",
]
