"""Unit + smoke tests for framework/parsers/ wrappers.

Verifies:
- All 7 parsers register themselves
- Each can be looked up by name
- Document flows through enrichment parsers correctly
- Real data (cached thong-tu OCR markdowns) parses through the chain

No network / no OCR model invocation — only exercises the parser wrappers
using data already on disk.
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

import gnosis.parsers  # registers all 7  # noqa: F401
from gnosis.core.registry import PluginRegistry
from gnosis.core.schema import Document, Page


def test_all_parsers_registered():
    expected = {
        "pdfplumber", "ocr2", "text_extractor",
        "table_normalizer", "multipage_stitcher",
        "element_classifier", "ellipsis_handler",
    }
    registered = set(PluginRegistry.list("parser"))
    missing = expected - registered
    assert not missing, f"missing plugins: {missing}. registered: {registered}"


def test_parser_has_name_attr():
    from gnosis.parsers import TableNormalizerParser
    p = TableNormalizerParser()
    assert p.name == "table_normalizer"


def test_table_normalizer_rewrites_html_to_kv():
    from gnosis.parsers import TableNormalizerParser

    # Synthetic page with HTML table
    raw_html = (
        "<table>"
        "<thead><tr>"
        + "".join(f"<th>C{i}</th>" for i in range(10)) + "</tr></thead>"
        "<tbody><tr>"
        + "".join(f"<td>v{i}</td>" for i in range(10)) + "</tr></tbody>"
        "</table>"
    )
    doc = Document(
        doc_id="d1", name="t", total_pages=1,
        pages=[Page(page_num=1, markdown=raw_html, raw_text="")],
    )

    out = TableNormalizerParser().parse("ignored", document=doc)
    assert "<table>" not in out.pages[0].markdown
    # Wide table → KV render
    assert "Dòng 1" in out.pages[0].markdown
    # Table also extracted as structured
    assert len(out.tables) == 1
    assert out.tables[0].n_cols == 10


def test_element_classifier_tags_captions():
    from gnosis.parsers import ElementClassifierParser

    md = "Introduction text.\nBảng 1.2: Something important here\nEnd text."
    doc = Document(
        doc_id="d", name="t", total_pages=1,
        pages=[Page(page_num=1, markdown=md, raw_text=md)],
    )
    out = ElementClassifierParser().parse("ignored", document=doc)
    assert "CAPTION: Bảng 1.2" in out.pages[0].markdown
    captions = out.meta.get("captions", [])
    assert len(captions) == 1
    assert captions[0]["kind"] == "Bảng"


def test_ellipsis_handler_flags_pages():
    from gnosis.parsers import EllipsisHandlerParser

    md = (
        "<table><tr><td>a</td><td>...</td><td>c</td></tr></table>"
    )
    doc = Document(
        doc_id="d", name="t", total_pages=1,
        pages=[Page(page_num=1, markdown=md,
                    raw_text="Detailed fallback text layer from PDF extraction " * 5)],
    )
    out = EllipsisHandlerParser().parse("ignored", document=doc)
    assert out.pages[0].meta.get("has_ellipsis") is True
    assert out.pages[0].meta.get("affected_cells") == 1
    assert 1 in out.meta.get("ellipsis_pages", {})


def test_table_normalizer_on_real_cached_page():
    """Parse page 37 of thong-tu cache through framework table_normalizer."""
    cache = (_ROOT.parent / "agent-search" / "smartsearch-v4" / "output"
             / "storage"
             / "Thong-tu-so-99-TT-BTC-ngay-27-10-2025-ve-huong-dan-che-ke-toan-doanh-nghiep"
             / "ocr2_markdowns.json")
    if not cache.exists():
        import pytest
        pytest.skip(f"Cache not present: {cache}")

    data = json.loads(cache.read_text(encoding="utf-8"))
    md_37 = data["37"]

    from gnosis.parsers import TableNormalizerParser
    doc = Document(
        doc_id="d", name="thong-tu", total_pages=1,
        pages=[Page(page_num=37, markdown=md_37, raw_text="")],
    )
    out = TableNormalizerParser().parse("ignored", document=doc)
    # p.37 has 22-col wide table → KV format
    assert "<table>" not in out.pages[0].markdown
    assert len(out.tables) >= 1
    # Normalization shrinks the page dramatically
    assert len(out.pages[0].markdown) < len(md_37)


def test_multipage_stitcher_on_cached_data():
    """Detect + merge the 4-page span on thong-tu p.611-614."""
    cache = (_ROOT.parent / "agent-search" / "smartsearch-v4" / "output"
             / "storage"
             / "6e1ac34408d8_Thong-tu-so-99-TT-BTC-ngay-27-10-2025-ve-huong-dan-che-ke-toan-doanh-nghiep"
             / "ocr2_markdowns.json")
    if not cache.exists():
        import pytest
        pytest.skip(f"Cache not present: {cache}")

    data = {int(k): v for k, v in
            json.loads(cache.read_text(encoding="utf-8")).items()}
    pages_wanted = [611, 612, 613, 614]

    doc = Document(
        doc_id="d", name="thong-tu", total_pages=max(pages_wanted),
        pages=[Page(page_num=pg, markdown=data[pg], raw_text="")
               for pg in pages_wanted if pg in data],
    )

    from gnosis.parsers import MultipageStitcherParser
    out = MultipageStitcherParser().parse("ignored", document=doc)
    report = out.meta.get("multipage_report", [])
    assert report, "expected at least one merged span"
    # The 611-614 span should be detected
    span_pages = {tuple(r["spans_pages"]) for r in report}
    assert (611, 612, 613, 614) in span_pages


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
