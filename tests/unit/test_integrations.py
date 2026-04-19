"""Tests for langchain + langgraph integrations.

LangChain tests skip if langchain-core not installed.
LangGraph tests are pure callable — no dep needed.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from gnosis.core.schema import Answer, Document, Hit, Page


# ─────────────── LangGraph (pure Python, no deps) ───────────────


def test_langgraph_parse_node():
    from gnosis.integrations.langgraph import make_parse_node

    class StubParser:
        name = "stub"
        def parse(self, source, document=None, **kw):
            return Document(doc_id="d", name=str(source), total_pages=1)

    node = make_parse_node(StubParser(), source_key="src", output_key="doc")
    state = {"src": "file.pdf"}
    out = node(state)
    assert out["doc"].name == "file.pdf"


def test_langgraph_retrieval_node():
    from gnosis.integrations.langgraph import make_retrieval_node

    class StubRetriever:
        name = "stub"
        def retrieve(self, q, top_k=30, context=None):
            return [Hit("c1", "d", f"ans:{q}", 1.0, "stub")]

    node = make_retrieval_node(StubRetriever(), query_key="q", output_key="h")
    state = {"q": "what?"}
    out = node(state)
    assert out["h"][0].text == "ans:what?"


def test_langgraph_synthesis_node():
    from gnosis.integrations.langgraph import make_synthesis_node

    class StubSynth:
        name = "stub"
        def synthesize(self, q, hits, chat_history=None, context=None):
            return Answer(text=f"A({q})={hits[0].text}")

    node = make_synthesis_node(StubSynth(),
                               query_key="q", hits_key="h", output_key="a",
                               history_key=None)
    state = {"q": "x", "h": [Hit("c", "d", "hello", 1.0, "s")]}
    out = node(state)
    assert out["a"].text == "A(x)=hello"


# ─────────────── LangChain (skipped if dep missing) ───────────────


def test_langchain_retriever_adapter():
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("langchain-core not installed")

    from gnosis.integrations.langchain import FrameworkRetrieverAdapter

    class StubRetriever:
        name = "stub"
        def retrieve(self, q, top_k=30, context=None):
            return [Hit("c1", "d", "content A", 0.9, "x", meta={"page": 5}),
                    Hit("c2", "d", "content B", 0.5, "y", meta={"page": 7})]

    adapter = FrameworkRetrieverAdapter(StubRetriever())
    lc_retriever = adapter.as_retriever()

    docs = lc_retriever.invoke("query")
    assert len(docs) == 2
    assert docs[0].page_content == "content A"
    assert docs[0].metadata["score"] == 0.9
    assert docs[0].metadata["page"] == 5


def test_langchain_document_loader():
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("langchain-core not installed")

    from gnosis.integrations.langchain import FrameworkDocumentLoader

    class StubParser:
        name = "stub"
        def parse(self, source, document=None, **kw):
            return Document(
                doc_id="d", name="n", total_pages=2,
                pages=[
                    Page(page_num=1, markdown="p1", page_type="text"),
                    Page(page_num=2, markdown="p2", page_type="scan"),
                ],
            )

    loader = FrameworkDocumentLoader(StubParser(), source="x.pdf")
    docs = loader.load()
    assert len(docs) == 2
    assert docs[0].page_content == "p1"
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page_type"] == "scan"


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
        except Exception as e:
            if "skip" in str(e).lower() or "Skipped" in str(type(e).__name__):
                print(f"  SKIP {name}: {e}")
            else:
                import traceback
                print(f"  FAIL/ERR {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} ok (skips not counted as fail)")
    sys.exit(0 if failed == 0 else 1)
