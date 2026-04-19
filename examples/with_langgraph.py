"""Example: compose framework components as LangGraph nodes.

Requires: pip install langgraph

Usage:
    python examples/with_langgraph.py path/to/doc.pdf "your question"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from gnosis.indexers import PageBM25Indexer
from gnosis.integrations.langgraph import (
    make_retrieval_node,
)
from gnosis.parsers import PdfplumberParser
from gnosis.retrievers import BmxMultiQueryChannel, HybridChatbotRetriever


def build_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        print("Install langgraph: pip install langgraph")
        sys.exit(1)
    return StateGraph(dict), START, END


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python examples/with_langgraph.py <pdf> <question>")
        sys.exit(1)
    pdf = sys.argv[1]
    question = sys.argv[2]

    doc = PdfplumberParser().parse(pdf)
    bm25 = PageBM25Indexer(); bm25.build(doc)
    retriever = HybridChatbotRetriever(
        channels=[BmxMultiQueryChannel(bm25_indexer=bm25)],
        bm25_indexer=bm25, final_top_k=5, total_pages=doc.total_pages,
    )
    ctx = {"doc_id": doc.doc_id,
           "page_texts": {p.page_num: p.markdown for p in doc.pages}}

    graph, START, END = build_graph()
    graph.add_node("retrieve",
                   make_retrieval_node(retriever, query_key="q",
                                       output_key="hits", context_key="ctx"))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", END)
    compiled = graph.compile()

    out = compiled.invoke({"q": question, "ctx": ctx})
    hits = out["hits"]
    print(f"Graph returned {len(hits)} hits")
    for h in hits[:3]:
        print(f"  p.{h.meta.get('page')} score={h.score:.2f}  {h.text[:100]!r}")


if __name__ == "__main__":
    main()
