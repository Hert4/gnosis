"""Pipeline orchestrator + builder.

A Pipeline wires concrete plugins across layers. Two construction paths:

  1. Programmatic:
        Pipeline().parse(OCR2()).index(BM25()).retrieve(Hybrid()).synthesize(Gemini()).build()

  2. Declarative (PipelineConfig → concrete plugins via registry):
        Pipeline.from_config(PipelineConfig(...))

At query time, the Pipeline passes a PipelineContext through each stage.
Indexers accumulate state; Retrievers query the built indexes; Rankers
reorder; Synthesizers produce the final Answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnosis.core.config import PipelineConfig, StageConfig
from gnosis.core.context import PipelineContext
from gnosis.core.events import EventEmitter
from gnosis.core.protocols import (
    IndexerProtocol,
    ParserProtocol,
    RankerProtocol,
    RetrieverProtocol,
    SynthesizerProtocol,
)
from gnosis.core.registry import PluginRegistry
from gnosis.core.schema import Answer, Document, Hit


class Pipeline:
    """Ordered composition of parser → indexer → retriever → ranker → synthesizer.

    Holds plugin instances + in-memory state (current Document, built indexes).
    Thread-safety is the caller's responsibility.
    """

    def __init__(
        self,
        parsers: list[ParserProtocol] | None = None,
        indexers: list[IndexerProtocol] | None = None,
        retrievers: list[RetrieverProtocol] | None = None,
        rankers: list[RankerProtocol] | None = None,
        synthesizers: list[SynthesizerProtocol] | None = None,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self.parsers: list[ParserProtocol] = list(parsers or [])
        self.indexers: list[IndexerProtocol] = list(indexers or [])
        self.retrievers: list[RetrieverProtocol] = list(retrievers or [])
        self.rankers: list[RankerProtocol] = list(rankers or [])
        self.synthesizers: list[SynthesizerProtocol] = list(synthesizers or [])
        self.events = event_emitter or EventEmitter()
        self.document: Document | None = None

    # ────────────────── Ingestion ──────────────────

    def load_document(self, source: Any, **kwargs) -> Document:
        """Run all parsers in order, then build all indexers."""
        doc: Document | None = None
        for parser in self.parsers:
            doc = parser.parse(source, document=doc, **kwargs)
            self.events.emit({"type": "log", "summary": f"parsed via {parser.name}"})
        if doc is None:
            raise RuntimeError("No parsers configured — cannot build Document")
        self.document = doc
        for indexer in self.indexers:
            indexer.build(doc)
            self.events.emit({"type": "log", "summary": f"indexed via {indexer.name}"})
        return doc

    # ────────────────── Query ──────────────────

    def query(
        self,
        question: str,
        *,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int = 30,
        on_event: Any = None,
    ) -> Answer:
        if on_event is not None:
            self.events.on(on_event)

        ctx = PipelineContext(
            query=question,
            chat_history=chat_history or [],
            doc_id=self.document.doc_id if self.document else "",
        )

        # Retrieval across all retrievers; merge by dedup on chunk_id
        all_hits: list[Hit] = []
        seen: set[str] = set()
        for retriever in self.retrievers:
            for hit in retriever.retrieve(question, top_k=top_k, context=ctx.state):
                if hit.chunk_id in seen:
                    continue
                seen.add(hit.chunk_id)
                all_hits.append(hit)

        # Rerank sequentially
        for ranker in self.rankers:
            all_hits = ranker.rank(all_hits, query=question, context=ctx.state)

        ctx.set("ranked_hits", all_hits)

        # Synthesize (first synth wins — multiple allowed for A/B testing)
        if not self.synthesizers:
            raise RuntimeError("No synthesizer configured")
        synth = self.synthesizers[0]
        return synth.synthesize(
            question, all_hits,
            chat_history=chat_history,
            context=ctx.state,
        )

    # ────────────────── Construction helpers ──────────────────

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig | dict | str | Path,
        globals_override: dict[str, Any] | None = None,
    ) -> Pipeline:
        if isinstance(config, (str, Path)):
            config = PipelineConfig.from_yaml(config)
        elif isinstance(config, dict):
            config = PipelineConfig.from_dict(config)

        shared = {**config.globals_, **(globals_override or {})}

        def _instantiate(layer: str, stage: StageConfig):
            cls_ = PluginRegistry.get(layer, stage.type)
            kwargs = {**shared.get(layer, {}), **stage.config}
            return cls_(**kwargs) if kwargs else cls_()

        return cls(
            parsers=[_instantiate("parser", s) for s in config.parsers],
            indexers=[_instantiate("indexer", s) for s in config.indexers],
            retrievers=[_instantiate("retriever", s) for s in config.retrievers],
            rankers=[_instantiate("ranker", s) for s in config.rankers],
            synthesizers=[_instantiate("synthesizer", s) for s in config.synthesizers],
        )


class PipelineBuilder:
    """Fluent builder for programmatic construction.

    Usage:
        p = (PipelineBuilder()
             .parse(OCR2())
             .index(BM25())
             .retrieve(Hybrid())
             .rank(WeightedMerge())
             .synthesize(GeminiChat())
             .build())
    """

    def __init__(self) -> None:
        self._parsers: list[ParserProtocol] = []
        self._indexers: list[IndexerProtocol] = []
        self._retrievers: list[RetrieverProtocol] = []
        self._rankers: list[RankerProtocol] = []
        self._synthesizers: list[SynthesizerProtocol] = []

    def parse(self, parser: ParserProtocol) -> PipelineBuilder:
        self._parsers.append(parser)
        return self

    def index(self, indexer: IndexerProtocol) -> PipelineBuilder:
        self._indexers.append(indexer)
        return self

    def retrieve(self, retriever: RetrieverProtocol) -> PipelineBuilder:
        self._retrievers.append(retriever)
        return self

    def rank(self, ranker: RankerProtocol) -> PipelineBuilder:
        self._rankers.append(ranker)
        return self

    def synthesize(self, synth: SynthesizerProtocol) -> PipelineBuilder:
        self._synthesizers.append(synth)
        return self

    def build(self) -> Pipeline:
        return Pipeline(
            parsers=self._parsers,
            indexers=self._indexers,
            retrievers=self._retrievers,
            rankers=self._rankers,
            synthesizers=self._synthesizers,
        )
