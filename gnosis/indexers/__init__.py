"""Indexers — turn a parsed Document into queryable index(es).

Each indexer keeps its index in-memory; Retrievers query via the indexer
instance. Pipelines hold the indexer list so retrievers can reference by
name or by type lookup.
"""

from gnosis.indexers import (  # noqa: F401
    entity_graph_indexer,
    page_bm25_indexer,
    tree_index_indexer,
)
from gnosis.indexers.entity_graph_indexer import EntityGraphIndexer
from gnosis.indexers.page_bm25_indexer import PageBM25Indexer
from gnosis.indexers.tree_index_indexer import TreeIndexIndexer

__all__ = ["EntityGraphIndexer", "PageBM25Indexer", "TreeIndexIndexer"]
