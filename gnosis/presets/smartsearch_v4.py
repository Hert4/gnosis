"""Preset that reproduces smartsearch-v4 behavior via the framework.

Usage:
    from gnosis.presets import smartsearch_v4

    pipeline = smartsearch_v4.build(
        llm_client=openai_client,
        llm_model="gemini-2.5-flash",
    )
    pipeline.load_document("doc.pdf")
    answer = pipeline.query("câu hỏi")

Parsers: pdfplumber → ocr2 → table_normalizer → ellipsis_handler
         → element_classifier (→ multipage_stitcher optional)
Indexers: page_bm25 (bmx mode) + tree_index + entity_graph
Retriever: hybrid_chatbot with all 5 channels
Ranker: weighted_merge (0.4/0.6 done inside hybrid; this is pass-through top-N)
Synthesizer: chatbot_llm (Vietnamese prompt)
"""

from __future__ import annotations

from typing import Any

from gnosis.core.pipeline import Pipeline, PipelineBuilder
from gnosis.indexers import (
    EntityGraphIndexer,
    PageBM25Indexer,
    TreeIndexIndexer,
)
from gnosis.parsers import (
    ElementClassifierParser,
    EllipsisHandlerParser,
    MultipageStitcherParser,
    OCR2Parser,
    PdfplumberParser,
    TableNormalizerParser,
    TextExtractorParser,
)
from gnosis.rankers import WeightedMergeRanker
from gnosis.retrievers import (
    BmxMultiQueryChannel,
    Entity2HopChannel,
    HybridChatbotRetriever,
    IdentifierBoostChannel,
    LlmTreeNavChannel,
    TreeBm25Channel,
)
from gnosis.synthesizers import ChatbotLLMSynthesizer


def build(
    *,
    llm_client=None,
    llm_model: str | None = None,
    ocr2_api_base: str = "http://127.0.0.1:30000",
    render_dpi: int = 250,
    enable_ocr2: bool = True,
    enable_text_extractor: bool = True,
    enable_multipage_stitch: bool = False,
    bm25_mode: str = "bmx",
    final_top_k: int = 15,
    neighbor_radius: int = 1,
) -> Pipeline:
    """Build a Pipeline replicating smartsearch-v4 defaults.

    Args:
        llm_client: OpenAI-compatible client (used by tree-nav + query expansion + synth).
        llm_model: Model name (e.g., "gemini-2.5-flash").
        ocr2_api_base: OpenAI-compatible API endpoint for OCR2-3B (sglang/vLLM/...).
        render_dpi: OCR render resolution (don't change unless you know why).
        enable_ocr2: Include OCR2 parser (requires sglang running or local GPU).
        enable_text_extractor: Include pdftohtml TextExtractor parser.
        enable_multipage_stitch: Include multipage table stitcher (off by default).
        bm25_mode: "bm25" or "bmx".
        final_top_k: How many hits to return from hybrid retriever.
        neighbor_radius: Add ±N neighbor pages to retrieved set.

    Returns a Pipeline ready to accept ``load_document(pdf)`` and ``query(q)``.
    """
    # ── Indexers (shared instances — channels reference them) ──
    bm25_idx = PageBM25Indexer(mode=bm25_mode)
    tree_idx = TreeIndexIndexer(
        run_refiner=bool(llm_client),
        run_enricher=bool(llm_client),
        llm_client=llm_client,
        llm_model=llm_model,
    )
    entity_idx = EntityGraphIndexer(
        llm_client=llm_client,
        llm_model=llm_model,
        tree_indexer=tree_idx,
    )

    # ── Retrieval channels ──
    channels = [
        LlmTreeNavChannel(tree_indexer=tree_idx,
                          llm_client=llm_client, llm_model=llm_model),
        TreeBm25Channel(tree_indexer=tree_idx),
        IdentifierBoostChannel(tree_indexer=tree_idx),
        Entity2HopChannel(entity_graph_indexer=entity_idx,
                          tree_indexer=tree_idx),
        BmxMultiQueryChannel(bm25_indexer=bm25_idx,
                             llm_client=llm_client, llm_model=llm_model),
    ]

    retriever = HybridChatbotRetriever(
        channels=channels,
        bm25_indexer=bm25_idx,
        final_top_k=final_top_k,
        neighbor_radius=neighbor_radius,
    )

    # ── Build pipeline ──
    builder = PipelineBuilder()

    # Parsers — order matters (each enriches doc from previous)
    builder.parse(PdfplumberParser())
    if enable_text_extractor:
        builder.parse(TextExtractorParser())
    if enable_ocr2:
        builder.parse(OCR2Parser(
            api_base=ocr2_api_base,
            dpi=render_dpi,
        ))
    builder.parse(TableNormalizerParser())
    builder.parse(EllipsisHandlerParser())
    builder.parse(ElementClassifierParser())
    if enable_multipage_stitch:
        builder.parse(MultipageStitcherParser())

    # Indexers (use the shared instances)
    builder.index(bm25_idx)
    builder.index(tree_idx)
    builder.index(entity_idx)

    # Retrieve + rank + synth
    builder.retrieve(retriever)
    builder.rank(WeightedMergeRanker(top_k=final_top_k))
    builder.synthesize(ChatbotLLMSynthesizer(
        llm_client=llm_client,
        llm_model=llm_model,
    ))

    pipeline = builder.build()
    # Attach shared refs so shim / consumers can access them
    pipeline._bm25_idx = bm25_idx
    pipeline._tree_idx = tree_idx
    pipeline._entity_idx = entity_idx
    pipeline._retriever = retriever
    return pipeline
