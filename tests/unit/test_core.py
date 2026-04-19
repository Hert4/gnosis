"""Unit tests for framework/core/ — protocols, schema, registry, pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnosis.core.config import PipelineConfig, StageConfig
from gnosis.core.events import Event, EventEmitter
from gnosis.core.pipeline import Pipeline, PipelineBuilder
from gnosis.core.protocols import (
    IndexerProtocol,
    ParserProtocol,
    RetrieverProtocol,
    SynthesizerProtocol,
)
from gnosis.core.registry import PluginRegistry, make, register
from gnosis.core.schema import Answer, Chunk, DocMeta, Document, Hit, Page


# ────────────────── Schema ──────────────────


def test_schema_construct_and_default():
    p = Page(page_num=1)
    assert p.page_num == 1
    assert p.raw_text == ""
    assert isinstance(p.meta, dict)

    d = Document(doc_id="d1", name="test", total_pages=5)
    assert d.total_pages == 5
    assert d.pages == []

    h = Hit(chunk_id="c1", doc_id="d1", text="...", score=0.8)
    assert h.channel == ""


# ────────────────── Registry ──────────────────


def test_registry_register_and_get():
    PluginRegistry.clear()

    @register("parser", "dummy_parser")
    class DummyParser:
        def parse(self, source, **kw):
            return Document(doc_id="x", name="x", total_pages=0)

    cls = PluginRegistry.get("parser", "dummy_parser")
    assert cls is DummyParser
    assert DummyParser.name == "dummy_parser"
    assert "dummy_parser" in PluginRegistry.list("parser")


def test_registry_missing_raises():
    PluginRegistry.clear()
    import pytest
    with pytest.raises(KeyError):
        PluginRegistry.get("parser", "nonexistent")


def test_registry_bad_layer_raises():
    import pytest
    with pytest.raises(ValueError):
        PluginRegistry.register_class("not_a_layer", "x", object)


def test_make_helper_instantiates():
    PluginRegistry.clear()

    @register("ranker", "pass_through")
    class PT:
        def __init__(self, multiplier: float = 1.0) -> None:
            self.mult = multiplier

        def rank(self, hits, **kw):
            return hits

    p = make("ranker", "pass_through", multiplier=2.0)
    assert p.mult == 2.0


# ────────────────── Events ──────────────────


def test_event_emitter_dispatches():
    em = EventEmitter()
    received: list[Event] = []
    em.on(lambda e: received.append(e))
    em.emit({"type": "log", "summary": "hello"})
    assert len(received) == 1
    assert received[0].type == "log"
    assert received[0].summary == "hello"


def test_event_listener_exception_does_not_break():
    em = EventEmitter()
    em.on(lambda e: 1 / 0)  # broken listener
    good_got: list[Event] = []
    em.on(lambda e: good_got.append(e))
    em.emit({"type": "log", "summary": "ok"})
    # Second listener still runs
    assert len(good_got) == 1


# ────────────────── Config ──────────────────


def test_config_from_dict_strings():
    cfg = PipelineConfig.from_dict({
        "pipeline": {
            "parse": ["a", "b"],
            "retrieve": [{"type": "hybrid", "top_k": 10}],
        }
    })
    assert [s.type for s in cfg.parsers] == ["a", "b"]
    assert cfg.retrievers[0].type == "hybrid"
    assert cfg.retrievers[0].config == {"top_k": 10}


def test_stage_config_accepts_string_or_dict():
    s = StageConfig.from_any("foo")
    assert s.type == "foo" and s.config == {}
    s2 = StageConfig.from_any({"type": "bar", "k": 1})
    assert s2.type == "bar" and s2.config == {"k": 1}


# ────────────────── Pipeline ──────────────────


def test_pipeline_builder_end_to_end():
    PluginRegistry.clear()

    class FakeParser:
        name = "fake_parser"
        def parse(self, source, document=None, **kw):
            return Document(doc_id="d", name="n", total_pages=1,
                            pages=[Page(page_num=1, raw_text="hello world")])

    class FakeIndexer:
        name = "fake_indexer"
        def __init__(self):
            self.built = False
        def build(self, doc, **kw):
            self.built = True
        def update(self, doc, **kw):
            pass

    class FakeRetriever:
        name = "fake_retriever"
        def retrieve(self, query, top_k=30, context=None):
            return [Hit(chunk_id="c1", doc_id="d",
                        text=f"answer for {query}", score=1.0,
                        channel="fake")]

    class PassRanker:
        name = "pass"
        def rank(self, hits, query="", context=None):
            return hits

    class EchoSynth:
        name = "echo"
        def synthesize(self, query, hits, chat_history=None, context=None):
            return Answer(text=hits[0].text if hits else "empty",
                          used_chunks=hits)

    pipeline = (PipelineBuilder()
                .parse(FakeParser())
                .index(FakeIndexer())
                .retrieve(FakeRetriever())
                .rank(PassRanker())
                .synthesize(EchoSynth())
                .build())

    doc = pipeline.load_document("dummy")
    assert doc.doc_id == "d"
    assert pipeline.indexers[0].built is True

    ans = pipeline.query("what?")
    assert "answer for what?" in ans.text


def test_pipeline_from_config_via_registry():
    PluginRegistry.clear()

    @register("parser", "cfg_parser")
    class CP:
        def parse(self, source, document=None, **kw):
            return Document(doc_id="c", name="c", total_pages=0)

    @register("indexer", "cfg_idx")
    class CI:
        def build(self, doc, **kw): pass
        def update(self, doc, **kw): pass

    @register("retriever", "cfg_ret")
    class CR:
        def retrieve(self, q, top_k=30, context=None):
            return [Hit("c1", "c", "x", 1.0)]

    @register("synthesizer", "cfg_syn")
    class CS:
        def synthesize(self, q, hits, chat_history=None, context=None):
            return Answer(text="done")

    pipe = Pipeline.from_config({
        "pipeline": {
            "parse": ["cfg_parser"],
            "index": ["cfg_idx"],
            "retrieve": ["cfg_ret"],
            "synthesize": ["cfg_syn"],
        }
    })
    pipe.load_document("x")
    ans = pipe.query("q")
    assert ans.text == "done"


# ────────────────── Protocol structural check ──────────────────


def test_protocols_structurally_satisfied():
    class OK:
        name = "ok"
        def parse(self, source, document=None, **kw):
            return Document(doc_id="x", name="x", total_pages=0)

    # Runtime-checkable protocol isinstance
    assert isinstance(OK(), ParserProtocol)


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
