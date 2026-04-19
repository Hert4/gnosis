"""Unit tests for rankers + synthesizers."""

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

import gnosis.rankers  # noqa: F401
import gnosis.synthesizers  # noqa: F401
from gnosis.core.registry import PluginRegistry
from gnosis.core.schema import Hit


def test_rankers_registered():
    expected = {"weighted_merge", "rrf", "smart_truncate"}
    assert expected <= set(PluginRegistry.list("ranker"))


def test_synthesizer_registered():
    assert "chatbot_llm" in PluginRegistry.list("synthesizer")


def test_weighted_merge_sorts_and_topk():
    from gnosis.rankers import WeightedMergeRanker
    ranker = WeightedMergeRanker(top_k=2)
    hits = [
        Hit("a", "d", "", 3.0, "x"),
        Hit("b", "d", "", 7.0, "x"),
        Hit("c", "d", "", 5.0, "x"),
    ]
    out = ranker.rank(hits)
    assert [h.chunk_id for h in out] == ["b", "c"]


def test_rrf_combines_channels():
    from gnosis.rankers import ReciprocalRankFusion
    # chunk_id 'a' ranked #1 on channel x AND channel y → highest RRF score
    # chunk_id 'b' only on channel x → lower
    hits = [
        Hit("a", "d", "", 100, "x"),
        Hit("b", "d", "", 50, "x"),
        Hit("a", "d", "", 100, "y"),
        Hit("c", "d", "", 20, "y"),
    ]
    out = ReciprocalRankFusion(k=1, top_k=3).rank(hits)
    assert out[0].chunk_id == "a"


def test_smart_truncate_ranker_respects_budget():
    from gnosis.rankers import SmartTruncateRanker
    hits = [
        Hit("a", "d", "x" * 1000, 1, "y"),
        Hit("b", "d", "x" * 1000, 1, "y"),
        Hit("c", "d", "x" * 1000, 1, "y"),
    ]
    out = SmartTruncateRanker(max_chars=1500).rank(hits)
    assert len(out) == 1  # only one hit fits under 1500 char budget


def test_smart_truncate_text_prefers_boundary():
    from gnosis.rankers.smart_truncate_ranker import smart_truncate
    text = "a" * 100 + "\n\n---\n\n" + "b" * 100
    # Budget larger than text → no truncate
    assert smart_truncate(text, 500) == text
    # Budget that catches the separator
    trunc = smart_truncate(text, 110)
    assert "truncated" in trunc
    # Should cut at or before the separator
    assert "b" * 100 not in trunc


def test_synth_without_llm_returns_notice():
    from gnosis.synthesizers import ChatbotLLMSynthesizer
    s = ChatbotLLMSynthesizer(llm_client=None, llm_model=None)
    a = s.synthesize("q", [Hit("a", "d", "text", 1, "y")])
    assert "not configured" in a.text


def test_synth_build_context_assembles_pages():
    from gnosis.synthesizers import ChatbotLLMSynthesizer
    s = ChatbotLLMSynthesizer(max_context_chars=10_000)
    hits = [
        Hit("page_1", "d", "content of page 1", 1.0, "ch", meta={"page": 1}),
        Hit("page_2", "d", "content of page 2", 0.5, "ch", meta={"page": 2}),
    ]
    ctx = s._build_context(hits)
    assert "[Trang 1]" in ctx and "[Trang 2]" in ctx


def test_synth_calls_llm_and_returns_citations():
    from gnosis.synthesizers import ChatbotLLMSynthesizer

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Câu trả lời (p.1)."))]
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_resp)

    s = ChatbotLLMSynthesizer(llm_client=mock_client, llm_model="test-model")
    hits = [Hit("page_1", "d", "content", 1, "ch", meta={"page": 1})]
    ans = s.synthesize("q?", hits)
    assert ans.text == "Câu trả lời (p.1)."
    assert len(ans.citations) == 1
    assert ans.citations[0]["page"] == 1


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
