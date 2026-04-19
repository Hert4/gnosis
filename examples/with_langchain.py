"""Example: use framework retriever as a LangChain BaseRetriever.

Drop-in with any LangChain chain that accepts a retriever.

Requires: pip install langchain-core langchain

Usage:
    python examples/with_langchain.py path/to/doc.pdf "your question"
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from gnosis.indexers import PageBM25Indexer, TreeIndexIndexer
from gnosis.integrations.langchain import FrameworkRetrieverAdapter
from gnosis.parsers import PdfplumberParser, TableNormalizerParser
from gnosis.retrievers import (
    BmxMultiQueryChannel,
    HybridChatbotRetriever,
    TreeBm25Channel,
)


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python examples/with_langchain.py <pdf> <question>")
        sys.exit(1)
    pdf = sys.argv[1]
    question = sys.argv[2]

    # Build framework retriever
    doc = PdfplumberParser().parse(pdf)
    doc = TableNormalizerParser().parse(pdf, document=doc)
    bm25 = PageBM25Indexer(); bm25.build(doc)
    tree = TreeIndexIndexer(); tree.build(doc)

    retriever = HybridChatbotRetriever(
        channels=[TreeBm25Channel(tree_indexer=tree),
                  BmxMultiQueryChannel(bm25_indexer=bm25)],
        bm25_indexer=bm25,
        final_top_k=5,
        total_pages=doc.total_pages,
    )

    # Wrap for LangChain
    ctx = {"doc_id": doc.doc_id,
           "page_texts": {p.page_num: p.markdown for p in doc.pages}}
    lc_retriever = FrameworkRetrieverAdapter(retriever, context=ctx).as_retriever()

    # Use in a LangChain RetrievalQA (stub — user plugs their own LLM)
    docs = lc_retriever.invoke(question)
    print(f"LangChain retriever returned {len(docs)} Documents:")
    for d in docs[:3]:
        print(f"  - p.{d.metadata.get('page', '?')} score={d.metadata.get('score'):.2f}")
        print(f"    {d.page_content[:150]!r}")


if __name__ == "__main__":
    main()
