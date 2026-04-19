"""Routers — multi-document selection layer.

For N documents, pick 1-3 relevant ones based on the query and delegate to
their per-doc Pipelines. Complementary to (not replacement for) single-doc
retrieval.
"""

from gnosis.routers import llm_flat_router  # noqa: F401
from gnosis.routers.llm_flat_router import LLMFlatRouter

__all__ = ["LLMFlatRouter"]
