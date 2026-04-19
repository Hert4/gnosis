"""PageBM25 indexer — wraps smartsearch/index.py:PageBM25.

Indexes per-page markdown (post-parse) for keyword search. Retrievers
query via ``indexer.query(text, top_k)``.
"""

from __future__ import annotations

from gnosis.core.registry import register
from gnosis.core.schema import Document


@register("indexer", "page_bm25")
class PageBM25Indexer:
    """Page-level BM25 or BMX index over ``page.markdown``."""

    def __init__(self, mode: str = "bmx", k1: float = 1.5, b: float = 0.75) -> None:
        self.mode = mode
        self.k1 = k1
        self.b = b
        self._bm25 = None

    def _make(self):

        from gnosis._impl.index import PageBM25
        self._bm25 = PageBM25(mode=self.mode, k1=self.k1, b=self.b)

    def build(self, document: Document, **kwargs) -> None:
        self._make()
        page_texts = {p.page_num: p.markdown for p in document.pages if p.markdown}
        self._bm25.build(page_texts)

    def update(self, document: Document, **kwargs) -> None:
        # PageBM25 rebuilds fully — cheap since doc fits in memory.
        self.build(document)

    # Convenience queries used by retrievers
    def query(self, text: str, top_k: int = 5) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        return self._bm25.query(text, top_k=top_k)

    def multi_query(self, texts: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        return self._bm25.multi_query(texts, top_k=top_k)

    @property
    def raw(self):
        return self._bm25
