"""Parsers — turn raw input (PDF) into Document, or enrich existing Document.

Each plugin wraps an existing implementation in agent-search/smartsearch-v4
so the framework doesn't re-implement OCR, table parsing, etc.

Source parsers: pdfplumber, ocr2_sglang, text_extractor (consume PDF path).
Enrichment parsers: table_normalizer, multipage_stitcher,
element_classifier, ellipsis_handler (consume Document, return enriched).
"""

# Import order matters — each module registers itself via @register at
# import time. The registry is populated as a side effect of importing here.
from gnosis.parsers import (  # noqa: F401
    pdfplumber_parser,
    ocr2_sglang_parser,
    text_extractor_parser,
    table_normalizer_parser,
    multipage_stitcher_parser,
    element_classifier_parser,
    ellipsis_handler_parser,
)
from gnosis.parsers.pdfplumber_parser import PdfplumberParser
from gnosis.parsers.ocr2_sglang_parser import OCR2SglangParser
from gnosis.parsers.text_extractor_parser import TextExtractorParser
from gnosis.parsers.table_normalizer_parser import TableNormalizerParser
from gnosis.parsers.multipage_stitcher_parser import MultipageStitcherParser
from gnosis.parsers.element_classifier_parser import ElementClassifierParser
from gnosis.parsers.ellipsis_handler_parser import EllipsisHandlerParser

__all__ = [
    "PdfplumberParser",
    "OCR2SglangParser",
    "TextExtractorParser",
    "TableNormalizerParser",
    "MultipageStitcherParser",
    "ElementClassifierParser",
    "EllipsisHandlerParser",
]
