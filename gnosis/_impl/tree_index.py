"""
tree_index.py — Hierarchical document tree index built from OCR markdown.

Multi-strategy cascade (adapted from NanoIndex):
  1. Dense markdown headings (>= 15% page coverage) → nested tree
  2. Sparse headings (< 15% coverage) → section-grouped tree
  3. No/few headings → page-based tree with heading children

Post-processing:
  - Running header detection (frequency-based, no hardcoded patterns)
  - Orphan page recovery
  - Full page text reassignment for leaf nodes
  - Parent/sibling text deduplication
  - Title disambiguation
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from nanoindex.models import DocumentTree, TreeNode
from nanoindex.utils.markdown import HeadingNode, parse_markdown_headings
from nanoindex.utils.tree_ops import (
    assign_node_ids,
    collect_text,
    find_node,
    find_siblings,
    iter_nodes,
    tree_to_outline,
)

from gnosis._impl.index import _tok

logger = logging.getLogger(__name__)

# Minimum tokens for a node to be considered meaningful
_MIN_NODE_TOKENS = 20
# Maximum tokens before splitting a node
_MAX_NODE_TOKENS = 4000

_PAGE_MARKER_RE = re.compile(r"<!--\s*nanoindex:page:(\d+)\s*-->")


# ── Helper functions ────────────────────────────────────────


def _split_markdown_by_page(combined: str, total_pages: int) -> dict[int, str]:
    """Split merged markdown into per-page text using page markers."""
    page_texts: dict[int, list[str]] = {p: [] for p in range(1, total_pages + 1)}
    current_page = 1

    for line in combined.split("\n"):
        m = _PAGE_MARKER_RE.search(line)
        if m:
            current_page = int(m.group(1))
            continue
        page_texts.setdefault(current_page, []).append(line)

    return {p: "\n".join(lines).strip() for p, lines in page_texts.items()}


def _best_page_title(headings_on_page: list[HeadingNode], text: str, page_num: int) -> str:
    """Pick the best title for a page node — first meaningful line."""
    for h in headings_on_page:
        t = h.title.strip()
        if t and len(t) > 3:
            return t
    for line in text.split("\n"):
        cleaned = line.strip().lstrip("#").strip().strip("*").strip()
        if cleaned and len(cleaned) > 3:
            return cleaned[:100]
    return f"Page {page_num}"


def _iter_all(nodes: list[TreeNode]):
    """Depth-first iteration over all nodes."""
    for n in nodes:
        yield n
        yield from _iter_all(n.nodes)


# ── Running header detection (frequency-based, generic) ─────


def _find_running_headers(headings: list[HeadingNode]) -> set[str]:
    """Detect running headers: titles appearing on >30% of headings."""
    if len(headings) < 4:
        return set()

    title_counts = Counter(h.title.strip() for h in headings)
    threshold = len(headings) * 0.3
    return {title for title, count in title_counts.items() if count >= threshold}


def _strip_boilerplate(
    headings: list[HeadingNode], to_remove: set[str],
) -> list[HeadingNode]:
    """Remove boilerplate headings."""
    return [h for h in headings if h.title.strip() not in to_remove]


# ── Tree building strategies ────────────────────────────────


def _nest_flat_nodes(flat: list[HeadingNode]) -> list[TreeNode]:
    """Convert a flat list of headings into a nested tree using a stack."""
    root: list[TreeNode] = []
    stack: list[tuple[int, TreeNode]] = []

    for h in flat:
        node = TreeNode(
            title=h.title,
            level=h.level,
            text=h.text_content or None,
            start_index=h.page,
            end_index=h.page,
        )
        while stack and stack[-1][0] >= h.level:
            stack.pop()
        if stack:
            stack[-1][1].nodes.append(node)
        else:
            root.append(node)
        stack.append((h.level, node))

    return root


def _build_section_grouped_tree(
    combined: str, total_pages: int, real_headings: list[HeadingNode],
) -> list[TreeNode]:
    """Group pages into sections defined by sparse headings.

    Each heading → parent node. Pages from that heading until the next
    heading become children. Handles preamble (pages before first heading).
    """
    page_texts = _split_markdown_by_page(combined, total_pages)
    headings_sorted = sorted(real_headings, key=lambda h: h.page)

    sections: list[TreeNode] = []
    for i, heading in enumerate(headings_sorted):
        start_page = heading.page
        end_page = (headings_sorted[i + 1].page - 1) if i + 1 < len(headings_sorted) else total_pages
        end_page = max(end_page, start_page)

        section_text_parts = [page_texts.get(p, "") for p in range(start_page, end_page + 1)]
        section_text = "\n\n".join(t for t in section_text_parts if t).strip()

        section_node = TreeNode(
            title=heading.title,
            level=heading.level,
            text=section_text or None,
            start_index=start_page,
            end_index=end_page,
        )

        # If section spans multiple pages, create child per page
        span = end_page - start_page + 1
        if span > 1:
            heading_by_page: dict[int, list[HeadingNode]] = {}
            for h in real_headings:
                if start_page <= h.page <= end_page:
                    heading_by_page.setdefault(h.page, []).append(h)

            for pg in range(start_page, end_page + 1):
                pg_text = page_texts.get(pg, "").strip()
                if not pg_text:
                    continue
                pg_title = _best_page_title(heading_by_page.get(pg, []), pg_text, pg)
                section_node.nodes.append(TreeNode(
                    title=pg_title,
                    level=heading.level + 1,
                    text=pg_text,
                    start_index=pg,
                    end_index=pg,
                ))
            if section_node.nodes:
                section_node.text = None  # Children carry the text

        sections.append(section_node)

    # Handle pages before the first heading
    if headings_sorted and headings_sorted[0].page > 1:
        pre_pages = list(range(1, headings_sorted[0].page))
        pre_parts = [page_texts.get(p, "") for p in pre_pages]
        pre_text = "\n\n".join(t for t in pre_parts if t).strip()
        if pre_text:
            min_level = min(h.level for h in headings_sorted)
            preamble = TreeNode(
                title="Preamble",
                level=min_level,
                text=pre_text,
                start_index=1,
                end_index=headings_sorted[0].page - 1,
            )
            sections.insert(0, preamble)

    return _nest_flat_nodes_from_tree(sections)


def _nest_flat_nodes_from_tree(sections: list[TreeNode]) -> list[TreeNode]:
    """Re-nest flat TreeNode list using heading levels (stack algorithm)."""
    root: list[TreeNode] = []
    stack: list[TreeNode] = []

    for node in sections:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            stack[-1].nodes.append(node)
        else:
            root.append(node)
        stack.append(node)

    return root


def _build_page_based_tree(
    combined: str, total_pages: int, heading_nodes: list[HeadingNode],
) -> list[TreeNode]:
    """One node per page — headings on each page become children."""
    page_texts = _split_markdown_by_page(combined, total_pages)

    heading_by_page: dict[int, list[HeadingNode]] = {}
    for h in heading_nodes:
        heading_by_page.setdefault(h.page, []).append(h)

    nodes: list[TreeNode] = []
    for pg in range(1, total_pages + 1):
        text = page_texts.get(pg, "").strip()
        if not text:
            continue

        headings_on_page = heading_by_page.get(pg, [])
        title = _best_page_title(headings_on_page, text, pg)

        node = TreeNode(
            title=title,
            level=1,
            text=text,
            start_index=pg,
            end_index=pg,
        )

        # Additional headings on the page become children
        for h in headings_on_page[1:]:
            node.nodes.append(TreeNode(
                title=h.title,
                level=2,
                text=h.text_content or None,
                start_index=pg,
                end_index=pg,
            ))

        nodes.append(node)

    return nodes


# ── Post-processing ─────────────────────────────────────────


def _propagate_page_ranges(nodes: list[TreeNode], total_pages: int) -> None:
    """Fill in page ranges bottom-up: children inform parents."""
    for node in nodes:
        if node.nodes:
            _propagate_page_ranges(node.nodes, total_pages)
            child_starts = [c.start_index for c in node.nodes if c.start_index]
            child_ends = [c.end_index for c in node.nodes if c.end_index]
            if child_starts and not node.start_index:
                node.start_index = min(child_starts)
            if child_ends:
                node.end_index = max(child_ends)
        if not node.end_index:
            node.end_index = node.start_index or total_pages


def _split_oversized(nodes: list[TreeNode]) -> list[TreeNode]:
    """Split nodes whose text exceeds _MAX_NODE_TOKENS into sub-nodes."""
    result: list[TreeNode] = []
    for node in nodes:
        node.nodes = _split_oversized(node.nodes)
        if node.text and len(_tok(node.text)) > _MAX_NODE_TOKENS and not node.nodes:
            paragraphs = node.text.split("\n\n")
            chunk: list[str] = []
            chunk_tokens = 0
            part_idx = 0

            for para in paragraphs:
                para_tokens = len(_tok(para))
                if chunk_tokens + para_tokens > _MAX_NODE_TOKENS and chunk:
                    child = TreeNode(
                        title=f"{node.title} (part {part_idx + 1})",
                        level=node.level + 1,
                        text="\n\n".join(chunk),
                        start_index=node.start_index,
                        end_index=node.end_index,
                    )
                    node.nodes.append(child)
                    chunk = []
                    chunk_tokens = 0
                    part_idx += 1
                chunk.append(para)
                chunk_tokens += para_tokens

            if chunk:
                child = TreeNode(
                    title=f"{node.title} (part {part_idx + 1})",
                    level=node.level + 1,
                    text="\n\n".join(chunk),
                    start_index=node.start_index,
                    end_index=node.end_index,
                )
                node.nodes.append(child)

            if node.nodes:
                node.text = None

        result.append(node)
    return result


def _recover_orphan_pages(
    nodes: list[TreeNode], combined: str, total_pages: int,
) -> list[TreeNode]:
    """Create nodes for pages not covered by any existing tree node."""
    if total_pages <= 0:
        return nodes

    covered: set[int] = set()
    for node in _iter_all(nodes):
        if node.start_index and node.end_index:
            for p in range(node.start_index, node.end_index + 1):
                covered.add(p)

    orphan_pages = sorted(set(range(1, total_pages + 1)) - covered)
    if not orphan_pages:
        return nodes

    page_texts = _split_markdown_by_page(combined, total_pages)

    # Group contiguous orphan pages into ranges
    ranges: list[tuple[int, int]] = []
    start = orphan_pages[0]
    prev = start
    for p in orphan_pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = p
            prev = p
    ranges.append((start, prev))

    for rng_start, rng_end in ranges:
        parts = [page_texts.get(p, "") for p in range(rng_start, rng_end + 1)]
        text = "\n\n".join(t for t in parts if t).strip()
        if not text or len(text) < 20:
            continue

        # Try to extract a meaningful title from the first page
        first_text = page_texts.get(rng_start, "")
        title = f"Pages {rng_start}-{rng_end}" if rng_start != rng_end else f"Page {rng_start}"
        for line in first_text.split("\n"):
            cleaned = line.strip().lstrip("#").strip().strip("*").strip()
            if cleaned and len(cleaned) > 3:
                title = cleaned[:100]
                break

        nodes.append(TreeNode(
            title=title,
            level=1,
            text=text,
            start_index=rng_start,
            end_index=rng_end,
        ))

    if len(orphan_pages) > 0:
        logger.info("Recovered %d orphan pages as %d new nodes", len(orphan_pages), len(ranges))

    return nodes


def _reassign_page_text(
    nodes: list[TreeNode], combined: str, total_pages: int,
) -> None:
    """Replace heading-parser text with full per-page text for leaf nodes.

    The markdown heading parser only captures text between heading lines.
    Content not under any heading gets lost. This function reassigns each
    leaf node's text to the full markdown content of its page range.
    """
    page_texts = _split_markdown_by_page(combined, total_pages)

    for node in _iter_all(nodes):
        if node.nodes:
            continue  # Skip parent nodes
        if not node.start_index or node.start_index <= 0:
            continue
        parts = [page_texts.get(p, "") for p in range(node.start_index, (node.end_index or node.start_index) + 1)]
        full_text = "\n\n".join(t for t in parts if t).strip()
        if full_text:
            node.text = full_text


def _deduplicate_parent_text(nodes: list[TreeNode]) -> None:
    """Remove text from parent nodes that is duplicated in their children."""
    for node in nodes:
        if not node.nodes:
            continue
        _deduplicate_parent_text(node.nodes)

        if not node.text:
            continue

        first_child_text = node.nodes[0].text
        if first_child_text and first_child_text[:200] in node.text:
            idx = node.text.find(first_child_text[:200])
            prefix = node.text[:idx].strip() if idx > 0 else ""
            node.text = prefix if prefix else None


def _deduplicate_sibling_branches(nodes: list[TreeNode]) -> list[TreeNode]:
    """Remove duplicate top-level branches with high text overlap."""
    if len(nodes) <= 1:
        return nodes

    def _branch_text(node: TreeNode) -> str:
        parts = [node.text or ""]
        for child in node.nodes or []:
            parts.append(_branch_text(child))
        return "\n".join(parts)

    kept: list[TreeNode] = []
    kept_texts: list[str] = []

    for node in nodes:
        branch = _branch_text(node)
        if not branch.strip():
            kept.append(node)
            kept_texts.append("")
            continue

        is_dup = False
        for prev_text in kept_texts:
            if not prev_text:
                continue
            shorter = min(len(branch), len(prev_text))
            if shorter < 100:
                continue
            sample_len = min(shorter, 2000)
            overlap = sum(1 for a, b in zip(branch[:sample_len], prev_text[:sample_len]) if a == b)
            if overlap / sample_len > 0.80:
                is_dup = True
                break

        if not is_dup:
            kept.append(node)
            kept_texts.append(branch)

    if len(kept) < len(nodes):
        logger.info("Sibling dedup removed %d duplicate branches", len(nodes) - len(kept))
    return kept


def _disambiguate_titles(nodes: list[TreeNode]) -> None:
    """Make duplicate titles unique by appending distinguishing info."""
    # Collect all titles
    title_nodes: dict[str, list[TreeNode]] = {}
    for node in _iter_all(nodes):
        key = node.title.strip().lower()
        title_nodes.setdefault(key, []).append(node)

    for key, dupes in title_nodes.items():
        if len(dupes) < 2:
            continue

        for node in dupes:
            suffixes: list[str] = []

            # 1. Try child titles
            if node.nodes:
                child_titles = [c.title for c in node.nodes[:3] if c.title != node.title]
                if child_titles:
                    suffixes.append("; ".join(child_titles))

            # 2. Try first meaningful line of text
            if not suffixes and node.text:
                for line in node.text.split("\n"):
                    cleaned = line.strip().lstrip("#").strip().strip("*").strip()
                    if cleaned and cleaned.lower() != key and len(cleaned) > 5:
                        suffixes.append(cleaned[:60])
                        break

            # 3. Page number fallback
            if not suffixes and node.start_index:
                if node.start_index == node.end_index:
                    suffixes.append(f"p.{node.start_index}")
                else:
                    suffixes.append(f"p.{node.start_index}-{node.end_index}")

            if suffixes:
                node.title = f"{node.title} — {suffixes[0]}"


# ── Main class ──────────────────────────────────────────────


class DocumentTreeIndex:
    """Hierarchical document tree built from OCR2 markdown output.

    Provides section-level BM25 search and tree navigation for the agent.
    """

    def __init__(self):
        self._tree: DocumentTree | None = None
        self._node_index: dict[str, tuple[list[str], TreeNode]] = {}
        self._node_df: Counter = Counter()
        self._node_n: int = 0
        self._node_avgdl: float = 0.0

    @property
    def is_ready(self) -> bool:
        return self._tree is not None and len(self._tree.structure) > 0

    @property
    def tree(self) -> DocumentTree | None:
        return self._tree

    def build_from_pages(
        self, page_markdowns: dict[int, str], doc_name: str = "", total_pages: int = 0,
    ) -> None:
        """Build tree from per-page OCR2 markdown texts.

        Multi-strategy cascade:
        1. Merge pages with page markers
        2. Parse markdown headings
        3. Strip running headers (frequency-based)
        4. Choose strategy based on heading density
        5. Post-process: orphan recovery, page text, dedup, disambig
        6. Build section BM25 index
        """
        if not page_markdowns:
            return

        if not total_pages:
            total_pages = max(page_markdowns.keys()) if page_markdowns else 0

        # 1. Merge all page markdowns with page markers
        parts: list[str] = []
        for page_num in sorted(page_markdowns.keys()):
            parts.append(f"<!-- nanoindex:page:{page_num} -->")
            parts.append(page_markdowns[page_num])
        combined = "\n\n".join(parts)

        # 2. Parse markdown headings
        headings = parse_markdown_headings(combined)

        # 3. Strip running headers (frequency-based, no hardcoded patterns)
        boilerplate = _find_running_headers(headings)
        if boilerplate:
            real_headings = _strip_boilerplate(headings, boilerplate)
            logger.info(
                "Stripped %d running headers (%s) — %d real headings remain",
                len(headings) - len(real_headings),
                ", ".join(sorted(boilerplate)[:3]),
                len(real_headings),
            )
        else:
            real_headings = headings

        # 4. Multi-strategy cascade
        page_coverage = (
            len({h.page for h in real_headings}) / total_pages
            if total_pages and real_headings
            else 0
        )

        if real_headings and page_coverage >= 0.15:
            # Dense headings → nested tree
            logger.info(
                "Using markdown headings (%d headings, %.0f%% page coverage)",
                len(real_headings), page_coverage * 100,
            )
            structure = _nest_flat_nodes(real_headings)
        elif real_headings:
            # Sparse headings → section-grouped tree
            logger.info(
                "Sparse headings (%d, %.0f%% coverage) → section-grouped tree",
                len(real_headings), page_coverage * 100,
            )
            structure = _build_section_grouped_tree(combined, total_pages, real_headings)
        else:
            # No headings → page-based tree
            logger.info("No headings detected → page-based tree")
            structure = _build_page_based_tree(combined, total_pages, real_headings)

        # 5. Post-processing pipeline
        _propagate_page_ranges(structure, total_pages)
        structure = _split_oversized(structure)
        structure = _recover_orphan_pages(structure, combined, total_pages)
        _reassign_page_text(structure, combined, total_pages)
        _deduplicate_parent_text(structure)
        structure = _deduplicate_sibling_branches(structure)
        _disambiguate_titles(structure)

        # 6. Assign IDs
        assign_node_ids(structure)

        self._tree = DocumentTree(
            doc_name=doc_name or "document",
            structure=structure,
        )

        # 7. Build section BM25
        self._build_section_bm25()

        total_nodes = sum(1 for _ in _iter_all(structure))
        leaves = sum(1 for n in _iter_all(structure) if not n.nodes)
        logger.info("Tree built: %d nodes (%d leaves)", total_nodes, leaves)

    def _build_section_bm25(self) -> None:
        """Build BM25 index over tree node texts for section-level search."""
        self._node_index = {}
        self._node_df = Counter()

        for node in iter_nodes(self._tree.structure):
            text = collect_text(node)
            # Include summary in search index if available
            if node.summary:
                text = f"{node.summary}\n\n{text}" if text else node.summary
            if not text:
                continue
            tokens = _tok(text)
            if len(tokens) < _MIN_NODE_TOKENS:
                continue
            self._node_index[node.node_id] = (tokens, node)
            for t in set(tokens):
                self._node_df[t] += 1

        self._node_n = len(self._node_index)
        total_len = sum(len(toks) for toks, _ in self._node_index.values())
        self._node_avgdl = total_len / max(self._node_n, 1)

    def get_outline(self, max_depth: int = 3) -> str:
        """Return human-readable tree outline for agent consumption."""
        if not self._tree:
            return "(tree not built)"

        def _outline(nodes: list[TreeNode], depth: int, indent: int) -> list[str]:
            if depth <= 0:
                return []
            lines: list[str] = []
            prefix = "  " * indent
            for node in nodes:
                line = f"{prefix}- [{node.node_id}] {node.title}"
                if node.start_index and node.end_index:
                    if node.start_index == node.end_index:
                        line += f" (p.{node.start_index})"
                    else:
                        line += f" (p.{node.start_index}-{node.end_index})"
                n_children = len(node.nodes)
                if n_children and depth == 1:
                    line += f" [{n_children} sub]"
                lines.append(line)
                if node.nodes:
                    lines.extend(_outline(node.nodes, depth - 1, indent + 1))
            return lines

        lines = _outline(self._tree.structure, max_depth, 0)
        return "\n".join(lines)

    def search_tree(
        self, query: str, top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """BM25 search across tree node texts.

        Returns [(node_id, title, score)] sorted by relevance.
        """
        if not self._node_index:
            return []

        query_tokens = _tok(query)
        if not query_tokens:
            return []

        import math
        k1, b = 1.5, 0.75
        scores: list[tuple[str, str, float]] = []

        for node_id, (doc_tokens, node) in self._node_index.items():
            dl = len(doc_tokens)
            tf = Counter(doc_tokens)
            score = 0.0
            for qt in query_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                df = self._node_df.get(qt, 0)
                idf = max(0, math.log((self._node_n - df + 0.5) / (df + 0.5) + 1.0))
                score += idf * (f * (k1 + 1)) / (
                    f + k1 * (1 - b + b * dl / self._node_avgdl)
                )
            if score > 0:
                scores.append((node_id, node.title, score))

        scores.sort(key=lambda x: -x[2])
        return scores[:top_k]

    def get_node_content(self, node_id: str) -> str | None:
        """Get full text of a specific node (including children)."""
        if not self._tree:
            return None
        node = find_node(self._tree.structure, node_id)
        if not node:
            return None
        return collect_text(node)

    def get_section_context(self, node_id: str) -> str | None:
        """Get node content + sibling titles for surrounding awareness."""
        if not self._tree:
            return None
        node = find_node(self._tree.structure, node_id)
        if not node:
            return None

        parts: list[str] = []

        siblings = find_siblings(self._tree.structure, node_id, max_each_side=1)
        if siblings:
            sib_titles = [f"[{s.node_id}] {s.title}" for s in siblings]
            parts.append(f"Nearby sections: {', '.join(sib_titles)}")
            parts.append("")

        page_info = ""
        if node.start_index and node.end_index:
            if node.start_index == node.end_index:
                page_info = f" (p.{node.start_index})"
            else:
                page_info = f" (p.{node.start_index}-{node.end_index})"
        parts.append(f"## [{node.node_id}] {node.title}{page_info}")
        parts.append("")

        content = collect_text(node)
        if content:
            parts.append(content)

        return "\n".join(parts)

    def save(self, path: Path) -> None:
        """Serialize tree to JSON for caching."""
        if not self._tree:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._tree.model_dump(exclude_none=True),
                f, indent=2, ensure_ascii=False,
            )

    def load(self, path: Path) -> bool:
        """Load cached tree. Returns True if loaded successfully."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._tree = DocumentTree.model_validate(data)
            self._build_section_bm25()
            return True
        except Exception:
            return False
