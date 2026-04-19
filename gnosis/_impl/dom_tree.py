"""
dom_tree.py — DOM Tree builder + BM25 index + context windows.

Pipeline:
  index.html (from pdf_positional_parser) → HTMLParser → DOMNode tree
  → Context enrichment (prepend headers/headings to each node)
  → BM25Index (multi-granularity: node + row + page level)
  → ContextWindow (breadcrumb, headers, siblings)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional


def _tokenize(text: str) -> list[str]:
    """Generic tokenizer: lowercase + split by non-word chars."""
    return re.findall(r'\w+', text.lower())


@dataclass
class DOMNode:
    """A node in the document DOM tree."""

    node_id: int
    tag: str

    text: str = ""
    page_num: int = 0
    page_type: str = ""

    parent: Optional[DOMNode] = field(default=None, repr=False)
    children: list[DOMNode] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    depth: int = 0

    @property
    def is_searchable(self) -> bool:
        return self.tag in ("td", "th", "p", "h2", "h3", "figure") and bool(self.text.strip())

    @property
    def breadcrumb(self) -> str:
        parts = []
        chain = []
        node = self
        while node is not None:
            chain.append(node)
            node = node.parent
        for anc in reversed(chain):
            if anc.tag == "document":
                continue
            elif anc.tag == "page":
                parts.append(f"Page {anc.page_num}")
            elif anc.tag == "table":
                parts.append("Table")
            elif anc.tag == "tr":
                if anc.parent:
                    idx = next((i for i, c in enumerate(anc.parent.children) if c is anc), 0)
                    parts.append(f"Row {idx + 1}")
            elif anc.tag in ("td", "th"):
                if anc.parent:
                    idx = next((i for i, c in enumerate(anc.parent.children) if c is anc), 0)
                    label = "Header" if anc.tag == "th" else "Cell"
                    parts.append(f"{label} {idx + 1}")
            elif anc.tag in ("h2", "h3"):
                parts.append(f"Heading: {anc.text[:50]}")
            elif anc.tag == "p":
                parts.append("Paragraph")
            elif anc.tag == "figure":
                parts.append("Figure")
        return " > ".join(parts)

    def subtree_text(self, max_depth: int = 99) -> str:
        if max_depth <= 0:
            return self.text
        parts = [self.text] if self.text.strip() else []
        for child in self.children:
            t = child.subtree_text(max_depth - 1)
            if t.strip():
                parts.append(t)
        return " ".join(parts)

    def preceding_siblings(self, n: int = 3) -> list[DOMNode]:
        if self.parent is None:
            return []
        idx = next((i for i, c in enumerate(self.parent.children) if c is self), 0)
        start = max(0, idx - n)
        return self.parent.children[start:idx]

    def following_siblings(self, n: int = 3) -> list[DOMNode]:
        if self.parent is None:
            return []
        idx = next((i for i, c in enumerate(self.parent.children) if c is self), 0)
        return self.parent.children[idx + 1 : idx + 1 + n]


class DOMTreeBuilder(HTMLParser):
    """Build DOMNode tree from HTML (pdf_positional_parser output)."""

    TRACKED_TAGS = {"section", "table", "tr", "td", "th", "h2", "h3", "p", "figure"}

    def __init__(self):
        super().__init__()
        self._counter = 0
        self._root = DOMNode(node_id=0, tag="document", depth=0)
        self._current = self._root
        self._page_num = 0
        self._page_type = ""
        self._all_nodes: list[DOMNode] = [self._root]

    def _next_id(self) -> int:
        self._counter += 1
        return self._counter

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self.TRACKED_TAGS:
            if tag == "img" and self._current.tag == "figure":
                attr_dict = dict(attrs)
                self._current.text = f"[img: {attr_dict.get('src', '')}]"
            return

        attr_dict = dict(attrs)
        node = DOMNode(
            node_id=self._next_id(),
            tag=tag,
            parent=self._current,
            depth=self._current.depth + 1,
        )

        if tag == "section":
            node.tag = "page"
            node.page_num = int(attr_dict.get("data-page", "0"))
            node.page_type = attr_dict.get("data-type", "")
            self._page_num = node.page_num
            self._page_type = node.page_type
        elif tag == "figure":
            node.page_num = int(attr_dict.get("data-page", str(self._page_num)))
            node.page_type = self._page_type
        else:
            node.page_num = self._page_num
            node.page_type = self._page_type

        self._current.children.append(node)
        self._current = node
        self._all_nodes.append(node)

    def handle_endtag(self, tag: str) -> None:
        if tag not in self.TRACKED_TAGS:
            return
        effective_tag = "page" if tag == "section" else tag
        if self._current.tag == effective_tag:
            if self._current.text.strip():
                self._current.tokens = _tokenize(self._current.text)
            if self._current.parent is not None:
                self._current = self._current.parent

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            if self._current.text:
                self._current.text += " " + text
            else:
                self._current.text = text

    @property
    def root(self) -> DOMNode:
        return self._root

    @property
    def all_nodes(self) -> list[DOMNode]:
        return self._all_nodes


def build_dom_tree(html_path: Path) -> tuple[DOMNode, list[DOMNode]]:
    """Build DOM tree from index.html file."""
    html_content = html_path.read_text(encoding="utf-8")
    builder = DOMTreeBuilder()
    builder.feed(html_content)
    return builder.root, builder.all_nodes
