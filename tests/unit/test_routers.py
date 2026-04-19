"""Unit tests for routers."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from gnosis.core.registry import PluginRegistry
from gnosis.core.schema import Answer, DocMeta
from gnosis.routers import LLMFlatRouter


def test_router_registered():
    assert "llm_flat" in PluginRegistry.list("router")


def test_empty_router_returns_empty():
    r = LLMFlatRouter()
    assert r.route("q") == []
    ans = r.query("q")
    assert "Không tìm thấy" in ans.text


def _make_stub_pipeline(answer_text: str):
    p = MagicMock()
    p.query = MagicMock(return_value=Answer(text=answer_text, used_chunks=[]))
    return p


def test_route_falls_back_to_all_when_no_llm():
    r = LLMFlatRouter(llm_client=None, max_picks=5)
    r.add_document(DocMeta(doc_id="d1", name="n1", summary="s1"),
                   _make_stub_pipeline("a1"))
    r.add_document(DocMeta(doc_id="d2", name="n2", summary="s2"),
                   _make_stub_pipeline("a2"))
    picked = r.route("anything")
    assert set(picked) == {"d1", "d2"}


def test_router_delegates_to_picked_docs():
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='["d1"]'))]
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_resp)

    r = LLMFlatRouter(llm_client=mock_client, llm_model="test", max_picks=3)
    p1 = _make_stub_pipeline("answer from doc1")
    p2 = _make_stub_pipeline("answer from doc2")
    r.add_document(DocMeta(doc_id="d1", name="n1"), p1)
    r.add_document(DocMeta(doc_id="d2", name="n2"), p2)

    ans = r.query("something specific")
    assert "answer from doc1" in ans.text
    assert ans.meta["picked"] == ["d1"]
    p1.query.assert_called_once()
    p2.query.assert_not_called()


def test_router_combines_multiple_picks():
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='["d1", "d2"]'))]
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_resp)

    r = LLMFlatRouter(llm_client=mock_client, llm_model="test", max_picks=3)
    r.add_document(DocMeta(doc_id="d1", name="n1"),
                   _make_stub_pipeline("ans1"))
    r.add_document(DocMeta(doc_id="d2", name="n2"),
                   _make_stub_pipeline("ans2"))
    ans = r.query("q spanning both")
    assert "ans1" in ans.text and "ans2" in ans.text
    assert set(ans.meta["picked"]) == {"d1", "d2"}


def test_router_ignores_unknown_ids():
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content='["ghost", "d1"]'))]
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_resp)

    r = LLMFlatRouter(llm_client=mock_client, llm_model="test", max_picks=3)
    r.add_document(DocMeta(doc_id="d1", name="n1"),
                   _make_stub_pipeline("only real"))
    ans = r.query("q")
    assert ans.meta["picked"] == ["d1"]
    assert "only real" in ans.text


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
