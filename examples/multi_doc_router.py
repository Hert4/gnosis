"""Example: multi-document router.

Register N PDFs, each with its own Pipeline. LLM router picks the best
doc(s) for each question and delegates to their pipelines.

Usage:
    python examples/multi_doc_router.py path1.pdf path2.pdf ... "question"
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT.parent / "agent-search" / "smartsearch-v4" / ".env")

from openai import OpenAI

from gnosis.core.schema import DocMeta
from gnosis.presets import smartsearch_v4
from gnosis.routers import LLMFlatRouter


def _clean_key(k: str) -> str:
    m = re.search(r"AIza[A-Za-z0-9_\-]+", k)
    return m.group() if m else k.strip()


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python examples/multi_doc_router.py <pdf1> <pdf2> ... <question>")
        sys.exit(1)

    *pdfs, question = sys.argv[1:]
    if len(pdfs) < 1:
        print("Need at least one PDF.")
        sys.exit(1)

    api_key = _clean_key(os.getenv("CHUNKING_API_KEY", ""))
    base_url = os.getenv("CHUNKING_BASE_URL",
                         "https://generativelanguage.googleapis.com/v1beta/openai/")
    if not api_key.startswith("AIza"):
        print("API key missing — router falls back to querying all docs.")
        llm = None
    else:
        llm = OpenAI(base_url=base_url, api_key=api_key)

    router = LLMFlatRouter(llm_client=llm, llm_model="gemini-2.5-flash")

    for pdf in pdfs:
        pdf = Path(pdf)
        print(f"[compile] {pdf.name}")
        pipeline = smartsearch_v4.build(llm_client=llm, llm_model="gemini-2.5-flash")
        # Minimal pipeline — skip OCR and text_extractor for speed
        pipeline.parsers = [p for p in pipeline.parsers
                            if p.name in ("pdfplumber", "table_normalizer")]
        pipeline.load_document(str(pdf))
        doc = pipeline.document
        # Build a lightweight DocMeta for the router
        meta = DocMeta(
            doc_id=doc.doc_id,
            name=pdf.name,
            title=pdf.stem,
            summary=(doc.pages[0].markdown or "")[:400] if doc.pages else "",
            headings=[s["title"] for s in doc.sections[:10]],
            top_entities=[e["name"] for e in doc.entities[:10]],
            total_pages=doc.total_pages,
        )
        router.add_document(meta, pipeline)

    print(f"\n[query] {question}")
    ans = router.query(question)
    print(f"\n[picked] {ans.meta.get('picked')}")
    print(f"\n[answer]\n{ans.text}")


if __name__ == "__main__":
    main()
