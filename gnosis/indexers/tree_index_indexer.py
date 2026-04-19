"""Tree index — wraps smartsearch/tree_index.py:DocumentTreeIndex.

Builds hierarchical sections from page markdowns. Optionally runs refiner
and enricher afterwards for LLM-generated summaries on nodes.
"""

from __future__ import annotations

from gnosis._vendor import ensure_agent_search_on_path
from gnosis.core.registry import register
from gnosis.core.schema import Document


@register("indexer", "tree_index")
class TreeIndexIndexer:
    """Hierarchical tree built from markdown headings."""

    def __init__(
        self,
        *,
        run_refiner: bool = False,
        run_enricher: bool = False,
        llm_client=None,
        llm_model: str | None = None,
    ) -> None:
        self.run_refiner = run_refiner
        self.run_enricher = run_enricher
        self.llm_client = llm_client
        self.llm_model = llm_model
        self._tree_index = None

    def _ensure(self):
        ensure_agent_search_on_path()
        from smartsearch.tree_index import DocumentTreeIndex
        if self._tree_index is None:
            self._tree_index = DocumentTreeIndex()
        return self._tree_index

    def build(self, document: Document, **kwargs) -> None:
        tree = self._ensure()
        page_markdowns = {p.page_num: p.markdown for p in document.pages if p.markdown}
        if not page_markdowns:
            return

        tree.build_from_pages(
            page_markdowns,
            doc_name=document.name or document.doc_id,
            total_pages=document.total_pages,
        )

        if self.run_refiner and self.llm_client and self.llm_model:
            from smartsearch.refiner import TreeRefiner
            TreeRefiner(client=self.llm_client, model=self.llm_model).refine(tree.tree)

        if self.run_enricher and self.llm_client and self.llm_model:
            from smartsearch.enricher import TreeEnricher
            TreeEnricher(client=self.llm_client, model=self.llm_model).enrich(tree.tree)

        # Snapshot sections to framework schema for downstream consumers
        from nanoindex.utils.tree_ops import iter_nodes
        document.sections = [
            {
                "id": n.node_id,
                "title": n.title,
                "level": n.level,
                "page_start": n.start_index,
                "page_end": n.end_index or n.start_index,
                "summary": (n.summary or "")[:300],
            }
            for n in iter_nodes(tree.tree.structure) if n.start_index
        ]

    def update(self, document: Document, **kwargs) -> None:
        self.build(document)

    @property
    def raw(self):
        return self._tree_index

    @property
    def is_ready(self) -> bool:
        return bool(self._tree_index and self._tree_index.is_ready)
