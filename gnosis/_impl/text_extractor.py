"""
text_extractor.py — Structured text extraction from text-extractable PDFs.

Uses pdf_positional_parser (pdftohtml → layout detection → HTML) and
dom_tree (DOMTreeBuilder → structured text) to extract content that
preserves tables, headings, and paragraph structure.
"""

from __future__ import annotations

from pathlib import Path

from gnosis._impl.pdf_positional_parser import _run_pdftohtml, _page_to_html
from gnosis._impl.dom_tree import DOMTreeBuilder


class TextExtractor:
    """Extract structured text from PDF pages via positional HTML + DOM tree."""

    def __init__(self, storage_dir: str | Path | None = None):
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._xml_cache: dict[str, object] = {}  # pdf_path → XML root

    def _get_xml_root(self, pdf_path: str):
        """Parse PDF via pdftohtml, cache the XML root."""
        if pdf_path not in self._xml_cache:
            self._xml_cache[pdf_path] = _run_pdftohtml(pdf_path)
        return self._xml_cache[pdf_path]

    def extract_pages(
        self, pdf_path: str, pages: list[int]
    ) -> dict[int, str]:
        """Extract structured text from specific PDF pages.

        Args:
            pdf_path: Path to the PDF file.
            pages: 1-indexed page numbers to extract.

        Returns:
            {page_num: structured_text} for pages with content.
        """
        xml_root = self._get_xml_root(pdf_path)

        # Build page_num → page_element mapping
        page_elems = {}
        for page_elem in xml_root.findall("page"):
            num = int(page_elem.get("number", "0"))
            page_elems[num] = page_elem

        results: dict[int, str] = {}
        for page_num in pages:
            elem = page_elems.get(page_num)
            if elem is None:
                continue

            html_str = _page_to_html(elem, page_num)
            if not html_str:
                continue

            # Parse HTML through DOM tree builder
            builder = DOMTreeBuilder()
            builder.feed(html_str)

            # Find the page node and extract its full text
            for node in builder.all_nodes:
                if node.tag == "page" and node.page_num == page_num:
                    text = node.subtree_text(max_depth=99)
                    if text.strip():
                        results[page_num] = text.strip()
                    break

        return results
