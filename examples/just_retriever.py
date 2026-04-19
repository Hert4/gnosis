"""Example: use only the retriever tier — no synthesis.

Good for exploration or feeding the Hits into your own LLM chain.

Usage:
    python examples/just_retriever.py path/to/doc.pdf "your question"
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

from gnosis.indexers import PageBM25Indexer, TreeIndexIndexer
from gnosis.parsers import PdfplumberParser, TableNormalizerParser
from gnosis.retrievers import (
    BmxMultiQueryChannel,
    HybridChatbotRetriever,
    IdentifierBoostChannel,
    TreeBm25Channel,
)


def _clean_key(k: str) -> str:
    m = re.search(r"AIza[A-Za-z0-9_\-]+", k)
    return m.group() if m else k.strip()


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python examples/just_retriever.py <pdf> <question>")
        sys.exit(1)
    pdf = sys.argv[1]
    question = sys.argv[2]

    api_key = _clean_key(os.getenv("CHUNKING_API_KEY", ""))
    base_url = os.getenv("CHUNKING_BASE_URL",
                         "https://generativelanguage.googleapis.com/v1beta/openai/")
    llm = OpenAI(base_url=base_url, api_key=api_key) if api_key.startswith("AIza") else None

    # Parse
    doc = PdfplumberParser().parse(pdf)
    doc = TableNormalizerParser().parse(pdf, document=doc)

    # Index
    bm25 = PageBM25Indexer(mode="bmx")
    tree = TreeIndexIndexer()
    bm25.build(doc)
    tree.build(doc)

    # Retrieve
    channels = [
        TreeBm25Channel(tree_indexer=tree),
        IdentifierBoostChannel(tree_indexer=tree),
        BmxMultiQueryChannel(bm25_indexer=bm25, llm_client=llm, llm_model="gemini-2.5-flash"),
    ]
    retriever = HybridChatbotRetriever(
        channels=channels, bm25_indexer=bm25,
        final_top_k=10, neighbor_radius=0,
        total_pages=doc.total_pages,
    )

    ctx = {
        "doc_id": doc.doc_id,
        "page_texts": {p.page_num: p.markdown for p in doc.pages},
    }
    hits = retriever.retrieve(question, context=ctx)

    print(f"Retrieved {len(hits)} hits for: {question!r}\n")
    for i, h in enumerate(hits[:8], 1):
        page = h.meta.get("page", "?")
        print(f"  {i:>2}. p.{page:>3} score={h.score:.2f}  channel={h.channel}")
        snippet = h.text[:120].replace("\n", " ")
        print(f"      {snippet}")


if __name__ == "__main__":
    main()
