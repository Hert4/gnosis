"""LangChain adapters — wrap framework components as LangChain primitives.

Users can drop-in:
    from gnosis.integrations.langchain import FrameworkRetrieverAdapter
    lc_retriever = FrameworkRetrieverAdapter(my_framework_retriever)
    chain = RetrievalQA.from_chain_type(llm, retriever=lc_retriever)

Requires ``langchain-core`` installed. Import errors gated with a clear
message.
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    _LC_AVAILABLE = True
except ImportError as _e:
    _LC_AVAILABLE = False
    _IMPORT_ERR = _e

from gnosis.core.protocols import ParserProtocol, RetrieverProtocol


def _require_langchain() -> None:
    if not _LC_AVAILABLE:
        raise ImportError(
            "langchain-core not installed. "
            "Install with: pip install retrival-framework[langchain]"
        ) from _IMPORT_ERR


class FrameworkRetrieverAdapter:
    """Wrap a framework RetrieverProtocol as a LangChain BaseRetriever.

    Each framework Hit becomes a LangChain Document with the Hit's text as
    page_content and Hit's meta (plus score + channel) as metadata.
    """

    def __init__(self, retriever: RetrieverProtocol,
                 context: dict[str, Any] | None = None) -> None:
        _require_langchain()
        self._retriever = retriever
        self._context = context or {}

        # Build a BaseRetriever subclass dynamically so we only inherit when
        # langchain is actually importable.
        class _Inner(BaseRetriever):
            _r: Any
            _ctx: dict[str, Any]

            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
            ) -> list:
                hits = self._r.retrieve(query, context=self._ctx)
                return [
                    LCDocument(
                        page_content=h.text,
                        metadata={
                            "chunk_id": h.chunk_id,
                            "doc_id": h.doc_id,
                            "score": h.score,
                            "channel": h.channel,
                            **h.meta,
                        },
                    )
                    for h in hits
                ]

        inner = _Inner()
        inner._r = retriever
        inner._ctx = self._context
        self._lc_retriever = inner

    def as_retriever(self):
        """Return the underlying LangChain BaseRetriever instance."""
        return self._lc_retriever

    # Also expose __call__ so users can pass this directly
    def __call__(self, *args, **kwargs):
        return self._lc_retriever(*args, **kwargs)


class FrameworkDocumentLoader:
    """Wrap a framework ParserProtocol as a LangChain DocumentLoader.

    Emits one LangChain Document per Page in the parsed framework Document.
    """

    def __init__(self, parser: ParserProtocol, source: Any) -> None:
        _require_langchain()
        self._parser = parser
        self._source = source

    def load(self) -> list:
        doc = self._parser.parse(self._source)
        return [
            LCDocument(
                page_content=pg.markdown or pg.raw_text,
                metadata={
                    "doc_id": doc.doc_id,
                    "doc_name": doc.name,
                    "page": pg.page_num,
                    "page_type": pg.page_type,
                    **pg.meta,
                },
            )
            for pg in doc.pages
        ]

    def lazy_load(self):
        return iter(self.load())
