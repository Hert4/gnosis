"""Unit tests for framework/indexers/ — verify registration + wrapping."""

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

import gnosis.indexers  # register all 3  # noqa: F401
from gnosis.core.registry import PluginRegistry
from gnosis.core.schema import Document, Page


def test_all_indexers_registered():
    expected = {"page_bm25", "tree_index", "entity_graph"}
    registered = set(PluginRegistry.list("indexer"))
    assert expected <= registered, f"missing: {expected - registered}"


def test_page_bm25_builds_and_queries():
    from gnosis.indexers import PageBM25Indexer

    idx = PageBM25Indexer(mode="bmx")
    doc = Document(
        doc_id="d", name="t", total_pages=3,
        pages=[
            Page(page_num=1, markdown="alpha beta gamma"),
            Page(page_num=2, markdown="gamma delta epsilon"),
            Page(page_num=3, markdown="zeta eta theta"),
        ],
    )
    idx.build(doc)
    results = idx.query("gamma", top_k=3)
    # Pages 1 and 2 both contain "gamma"
    pages_hit = {p for p, _score in results}
    assert pages_hit >= {1, 2}


def test_page_bm25_empty_query_returns_empty():
    from gnosis.indexers import PageBM25Indexer
    idx = PageBM25Indexer()
    # Not built yet → empty
    assert idx.query("anything") == []


def test_tree_index_builds_sections_from_headings():
    from gnosis.indexers import TreeIndexIndexer

    md1 = "# Chapter 1\n\nSome content in chapter 1.\n\n## Section 1.1\n\nSubsection content here."
    md2 = "# Chapter 2\n\nChapter 2 body text follows here."
    doc = Document(
        doc_id="d", name="test", total_pages=2,
        pages=[
            Page(page_num=1, markdown=md1),
            Page(page_num=2, markdown=md2),
        ],
    )
    idx = TreeIndexIndexer()
    idx.build(doc)
    assert idx.is_ready is True
    # sections populated onto Document
    assert len(doc.sections) >= 2


def test_entity_graph_skips_without_llm():
    from gnosis.indexers import EntityGraphIndexer

    idx = EntityGraphIndexer()  # no llm_client
    doc = Document(doc_id="d", name="t", total_pages=1,
                   pages=[Page(page_num=1, markdown="content")])
    idx.build(doc)
    assert doc.meta.get("entity_graph_skipped") == "no_llm_client"
    assert idx.is_ready is False


def test_tree_then_entity_on_real_cached_data():
    """Build tree + snapshot sections from real cache. LLM steps skipped."""
    cache = (_ROOT.parent / "agent-search" / "smartsearch-v4" / "output"
             / "storage" / "Hướng dẫn cấu hình 2FA_Marketing"
             / "ocr2_markdowns.json")
    if not cache.exists():
        import pytest
        pytest.skip(f"Cache not present: {cache}")

    data = {int(k): v for k, v in
            json.loads(cache.read_text(encoding="utf-8")).items()}
    pages = [Page(page_num=p, markdown=data[p]) for p in sorted(data)]

    doc = Document(doc_id="2fa", name="2FA Marketing",
                   total_pages=max(data.keys()),
                   pages=pages)

    from gnosis.indexers import TreeIndexIndexer
    tree = TreeIndexIndexer()
    tree.build(doc)
    assert tree.is_ready is True
    assert len(doc.sections) > 0


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
