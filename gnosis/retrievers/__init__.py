"""Retrievers — query → candidate Hits.

Two tiers:

- **Channels** (plugin layer "channel"): single retrieval signal
  (LLM tree nav, BM25, identifier boost, entity hop, multi-query BMX).
- **Composers** (plugin layer "retriever"): orchestrate multiple channels,
  merge scores, rerank, expand neighbors.

An ``agent_loop`` retriever wraps the visual tool-calling fallback from
smartsearch-v4 for scan-only documents.
"""

from gnosis.retrievers.channels import (  # noqa: F401
    bmx_multiquery,
    entity_2hop,
    identifier_boost,
    llm_tree_nav,
    tree_bm25,
)
from gnosis.retrievers import hybrid_chatbot  # noqa: F401
from gnosis.retrievers.hybrid_chatbot import HybridChatbotRetriever
from gnosis.retrievers.channels.bmx_multiquery import BmxMultiQueryChannel
from gnosis.retrievers.channels.entity_2hop import Entity2HopChannel
from gnosis.retrievers.channels.identifier_boost import IdentifierBoostChannel
from gnosis.retrievers.channels.llm_tree_nav import LlmTreeNavChannel
from gnosis.retrievers.channels.tree_bm25 import TreeBm25Channel

__all__ = [
    "HybridChatbotRetriever",
    "BmxMultiQueryChannel",
    "Entity2HopChannel",
    "IdentifierBoostChannel",
    "LlmTreeNavChannel",
    "TreeBm25Channel",
]
