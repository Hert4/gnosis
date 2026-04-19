"""Entity graph indexer — wraps smartsearch/entity_graph.py:DocumentEntityGraph.

Requires an LLM client to extract entities + relationships from tree nodes.
If no LLM client configured, skips graph build gracefully.
"""

from __future__ import annotations

from gnosis._vendor import ensure_agent_search_on_path
from gnosis.core.registry import register
from gnosis.core.schema import Document


@register("indexer", "entity_graph")
class EntityGraphIndexer:
    """Entity+relationship graph built from tree nodes."""

    def __init__(
        self,
        *,
        llm_client=None,
        llm_model: str | None = None,
        tree_indexer=None,
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model
        self._tree_indexer = tree_indexer
        self._graph = None

    def link_tree(self, tree_indexer) -> None:
        """Inject the TreeIndexIndexer so we can read its built tree."""
        self._tree_indexer = tree_indexer

    def build(self, document: Document, **kwargs) -> None:
        ensure_agent_search_on_path()
        from smartsearch.entity_graph import DocumentEntityGraph

        if not self.llm_client or not self.llm_model:
            document.meta["entity_graph_skipped"] = "no_llm_client"
            return

        if self._tree_indexer is None or not self._tree_indexer.is_ready:
            document.meta["entity_graph_skipped"] = "tree_not_built"
            return

        graph = DocumentEntityGraph()
        graph.build_from_tree(
            self._tree_indexer.raw.tree,
            client=self.llm_client,
            model=self.llm_model,
            verbose=False,
        )
        self._graph = graph

        # Snapshot to framework schema
        document.entities = [
            {
                "name": e.name,
                "type": e.entity_type,
                "description": e.description,
                "source_node_ids": list(e.source_node_ids),
            }
            for e in graph._graph_data.entities
        ]

    def update(self, document: Document, **kwargs) -> None:
        self.build(document)

    @property
    def raw(self):
        return self._graph

    @property
    def is_ready(self) -> bool:
        return bool(self._graph and self._graph.is_ready)
