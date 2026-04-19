"""Tests for smartsearch_v4 preset + shim.

Verifies:
- Preset builds a Pipeline with all expected stages wired
- Shim exposes legacy attribute/method surface
- Shim's load_document returns the expected dict shape
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

from gnosis.presets import smartsearch_v4
from gnosis.shims import SmartSearchV4Shim


def test_preset_builds_pipeline():
    pipeline = smartsearch_v4.build(
        llm_client=None,
        llm_model=None,
        enable_ocr2=False,            # skip OCR2 to avoid needing sglang
        enable_text_extractor=False,  # skip to avoid needing real PDF
    )

    # Parsers present
    assert len(pipeline.parsers) >= 4  # pdfplumber + 3 enrichments
    # Indexers: 3
    assert len(pipeline.indexers) == 3
    # Retriever
    assert len(pipeline.retrievers) == 1
    # Ranker + synth
    assert len(pipeline.rankers) == 1
    assert len(pipeline.synthesizers) == 1

    # Shared refs attached for shim use
    assert hasattr(pipeline, "_bm25_idx")
    assert hasattr(pipeline, "_tree_idx")
    assert hasattr(pipeline, "_entity_idx")
    assert hasattr(pipeline, "_retriever")


def test_shim_exposes_legacy_surface():
    shim = SmartSearchV4Shim(
        answer_base_url="https://example/v1",
        answer_model="test",
        answer_api_key="AIzaStub",
        chunking_base_url="https://example/v1",
        chunking_model="test",
        chunking_api_key="AIzaStub",
        storage_dir="/tmp/stub",
    )

    # Read-only attributes exist and are None/empty pre-load
    assert shim.readiness == 0
    assert shim._tree_index is None
    assert shim._entity_graph is None
    assert shim._page_texts == {}
    assert shim._extracted_pages == set()
    assert shim.export_structured() == {}
    assert "readiness" in shim.processing_state


def test_shim_load_document_on_small_pdf():
    """Use HDSD PDF if present — should complete Pipeline stages
    (pdfplumber + parsers + index), even without OCR sglang."""
    pdf = (_ROOT.parent / "agent-search" / "documents"
           / "Hướng dẫn cấu hình 2FA_Marketing.pdf")
    if not pdf.exists():
        import pytest
        pytest.skip(f"PDF not present: {pdf}")

    shim = SmartSearchV4Shim(
        answer_base_url="https://example/v1",
        answer_model="test-model",
        answer_api_key="stub",
        chunking_base_url="https://example/v1",
        chunking_model="test-model",
        chunking_api_key="stub",
        storage_dir="/tmp/stub",
    )
    # Bypass OCR for this test — monkey-patch pipeline parsers
    # Keep only parsers that don't need external services
    shim._pipeline.parsers = [
        p for p in shim._pipeline.parsers
        if p.name in ("pdfplumber", "table_normalizer", "ellipsis_handler",
                      "element_classifier")
    ]

    status = shim.load_document(str(pdf))
    expected_keys = {"track", "total_pages", "readiness",
                     "text_pages", "scan_pages", "load_time_ms"}
    assert expected_keys <= set(status.keys())
    assert status["total_pages"] > 0
    assert shim.readiness == 2
    assert len(shim._page_texts) > 0


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
