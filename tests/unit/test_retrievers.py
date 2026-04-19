"""Unit tests for retriever channels + hybrid composer.

All channels tested with mocked indexers to isolate scoring logic.
Heavy end-to-end path exercised in integration tests (phase 5+).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import gnosis.retrievers  # registers all channels + retriever  # noqa: F401
from gnosis.core.registry import PluginRegistry


# ────────────────── Registration ──────────────────


def test_all_channels_and_retriever_registered():
    expected_channels = {
        "llm_tree_nav", "tree_bm25", "identifier_boost",
        "entity_2hop", "bmx_multiquery",
    }
    assert expected_channels <= set(PluginRegistry.list("channel"))
    assert "hybrid_chatbot" in PluginRegistry.list("retriever")


# ────────────────── Channels (no LLM / no tree) ──────────────────


def test_llm_tree_nav_returns_empty_when_tree_missing():
    from gnosis.retrievers import LlmTreeNavChannel
    ch = LlmTreeNavChannel(tree_indexer=None, llm_client=None, llm_model=None)
    assert ch.search("q", context={}) == []


def test_tree_bm25_returns_empty_when_tree_missing():
    from gnosis.retrievers import TreeBm25Channel
    ch = TreeBm25Channel(tree_indexer=None)
    assert ch.search("q", context={}) == []


def test_identifier_boost_no_digits_returns_empty():
    from gnosis.retrievers import IdentifierBoostChannel

    # Stub tree with is_ready=True but no digit tokens in query
    tree = SimpleNamespace(is_ready=True, raw=None)
    ch = IdentifierBoostChannel(tree_indexer=tree)
    assert ch.search("how to use system", context={}) == []


def test_entity_2hop_returns_empty_when_graph_missing():
    from gnosis.retrievers import Entity2HopChannel
    ch = Entity2HopChannel(entity_graph_indexer=None, tree_indexer=None)
    assert ch.search("q", context={}) == []


def test_bmx_multiquery_uses_indexer_only_when_no_llm():
    from gnosis.retrievers import BmxMultiQueryChannel

    # Stub bm25 indexer returning deterministic scores
    idx = MagicMock()
    idx.query = MagicMock(return_value=[(1, 0.9), (2, 0.5)])
    ch = BmxMultiQueryChannel(bm25_indexer=idx, llm_client=None, llm_model=None)

    hits = ch.search("alpha", context={"doc_id": "d"})
    assert len(hits) == 2
    assert hits[0].channel == "bmx_multiquery"
    assert hits[0].meta["page"] in (1, 2)
    # Only original query path, expansion list is empty
    assert idx.query.call_count == 1


# ────────────────── Hybrid composer ──────────────────


def test_hybrid_merges_channel_scores_and_reranks():
    from gnosis.retrievers import HybridChatbotRetriever

    class StubChannel:
        def __init__(self, name, hits):
            self.name = name
            self._hits = hits
        def search(self, q, top_k=20, context=None):
            return self._hits

    from gnosis.core.schema import Hit

    ch_a = StubChannel("a", [
        Hit(chunk_id="page_1", doc_id="d", text="", score=10, channel="a"),
        Hit(chunk_id="page_2", doc_id="d", text="", score=5, channel="a"),
    ])
    ch_b = StubChannel("b", [
        Hit(chunk_id="page_1", doc_id="d", text="", score=3, channel="b"),
        Hit(chunk_id="page_3", doc_id="d", text="", score=7, channel="b"),
    ])

    # Stub bm25 indexer (raw BM25 on original query)
    bm25 = MagicMock()
    bm25.query = MagicMock(return_value=[(1, 2.0), (2, 1.0), (3, 3.0)])

    r = HybridChatbotRetriever(
        channels=[ch_a, ch_b],
        bm25_indexer=bm25,
        merged_weight=0.4,
        bm25_weight=0.6,
        final_top_k=3,
        neighbor_radius=0,
        total_pages=10,
    )

    hits = r.retrieve("q", context={"doc_id": "d"})
    assert len(hits) == 3
    # Page 1 has merged = 13, bm25 = 2 → 0.4*13 + 0.6*2 = 6.4
    # Page 2 has merged = 5,  bm25 = 1 → 0.4*5  + 0.6*1 = 2.6
    # Page 3 has merged = 7,  bm25 = 3 → 0.4*7  + 0.6*3 = 4.6
    # Order should be: 1 (6.4), 3 (4.6), 2 (2.6)
    pages = [h.meta["page"] for h in hits]
    assert pages == [1, 3, 2]


def test_hybrid_neighbor_expansion():
    from gnosis.retrievers import HybridChatbotRetriever
    from gnosis.core.schema import Hit

    class StubChannel:
        name = "s"
        def search(self, q, top_k=20, context=None):
            return [Hit(chunk_id="page_5", doc_id="d", text="", score=10, channel="s")]

    bm25 = MagicMock(); bm25.query = MagicMock(return_value=[])
    r = HybridChatbotRetriever(
        channels=[StubChannel()],
        bm25_indexer=bm25,
        final_top_k=5,
        neighbor_radius=1,
        total_pages=10,
    )
    hits = r.retrieve("q", context={"doc_id": "d"})
    pages = sorted({h.meta["page"] for h in hits})
    # Page 5 + neighbors 4 and 6
    assert 5 in pages and 4 in pages and 6 in pages


def test_hybrid_empty_when_no_channels_fire():
    from gnosis.retrievers import HybridChatbotRetriever
    r = HybridChatbotRetriever(channels=[], bm25_indexer=None)
    assert r.retrieve("q") == []


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    tests = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
             if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  OK  {name}")
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
