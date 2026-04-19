"""End-to-end test: build a Pipeline manually (without preset), load HDSD
cached OCR markdown, index it, retrieve, verify Hits look sensible.

Does NOT call a real LLM — passes None for LLM-dependent channels/synth.
Exercises: pdfplumber-style Document → page_bm25 + tree_index →
identifier_boost + tree_bm25 + bmx_multiquery (no LLM expansion) →
hybrid composer.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import gnosis.indexers  # noqa: F401
import gnosis.retrievers  # noqa: F401
import gnosis.rankers  # noqa: F401
from gnosis.core.pipeline import PipelineBuilder
from gnosis.core.schema import Document, Page
from gnosis.indexers import PageBM25Indexer, TreeIndexIndexer
from gnosis.rankers import WeightedMergeRanker
from gnosis.retrievers import (
    BmxMultiQueryChannel,
    HybridChatbotRetriever,
    IdentifierBoostChannel,
    TreeBm25Channel,
)
from gnosis.synthesizers import ChatbotLLMSynthesizer

_CACHE = (_ROOT.parent / "agent-search" / "smartsearch-v4" / "output"
          / "storage" / "Hướng dẫn cấu hình 2FA_Marketing"
          / "ocr2_markdowns.json")


def _load_document() -> Document:
    data = {int(k): v for k, v in json.loads(_CACHE.read_text(encoding="utf-8")).items()}
    pages = [Page(page_num=p, markdown=data[p], raw_text="") for p in sorted(data)]
    return Document(doc_id="2fa", name="2FA Marketing",
                    total_pages=max(data), pages=pages)


def test_full_pipeline_build_and_retrieve():
    if not _CACHE.exists():
        import pytest; pytest.skip(f"Cache not present: {_CACHE}")

    doc = _load_document()

    # Indexers
    bm25 = PageBM25Indexer(mode="bmx")
    tree = TreeIndexIndexer()

    # Channels (no LLM - only deterministic channels)
    channels = [
        TreeBm25Channel(tree_indexer=tree, top_k_nodes=3),
        IdentifierBoostChannel(tree_indexer=tree, boost=25.0),
        BmxMultiQueryChannel(bm25_indexer=bm25, llm_client=None, llm_model=None),
    ]
    retriever = HybridChatbotRetriever(
        channels=channels,
        bm25_indexer=bm25,
        final_top_k=10,
        neighbor_radius=1,
        total_pages=doc.total_pages,
    )

    # Build pipeline using builder
    pipe = (PipelineBuilder()
            .index(bm25)
            .index(tree)
            .retrieve(retriever)
            .rank(WeightedMergeRanker(top_k=10))
            .synthesize(ChatbotLLMSynthesizer())  # will bail on missing LLM
            .build())

    # Skip ingest stage since we pre-built Document
    pipe.document = doc
    for idx in pipe.indexers:
        idx.build(doc)

    # Retrieve directly (bypassing full query so we don't need LLM)
    ctx = {
        "doc_id": doc.doc_id,
        "page_texts": {p.page_num: p.markdown for p in doc.pages},
    }
    hits = retriever.retrieve("Marketing phân hệ", context=ctx)

    assert len(hits) > 0, "expected some hits"
    # Every hit should have channel_scores populated (from hybrid composer)
    non_neighbor = [h for h in hits if h.channel != "neighbor"]
    assert any(h.channel_scores for h in non_neighbor), (
        "expected at least one non-neighbor hit with channel_scores"
    )
    # Hits should be sorted descending by score
    scores = [h.score for h in hits if h.channel != "neighbor"]
    assert scores == sorted(scores, reverse=True)


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
