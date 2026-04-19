"""
refiner.py — Smart splitting for oversized tree nodes.

Three-phase strategy (adapted from NanoIndex):
  Phase A: Heuristic heading split (zero LLM cost)
  Phase B: LLM-assisted subsection identification
  Phase C: Paragraph-boundary chunking (fallback)
"""

from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from nanoindex.models import DocumentTree, TreeNode
from nanoindex.utils.tree_ops import assign_node_ids, iter_nodes

from gnosis._impl.index import _tok

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

_SPLIT_PROMPT = """\
You are a document structure expert. The following text is from a section \
titled "{title}". Break it into 3-8 logical subsections.

Return ONLY a JSON array of subsection titles in document order:
["<title 1>", "<title 2>", ...]

Text (first {char_limit} characters):
{text}

JSON:"""

_MAX_REFINE_PASSES = 4


class TreeRefiner:
    """Smart splitting for oversized tree nodes."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_tokens: int = 4000,
    ):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def refine(self, tree: DocumentTree, verbose: bool = False) -> int:
        """Split oversized nodes. Modifies tree in-place. Returns number of nodes split."""
        total_split = 0
        for iteration in range(_MAX_REFINE_PASSES):
            oversized = [
                n for n in iter_nodes(tree.structure)
                if not n.nodes and n.text and len(_tok(n.text)) > self._max_tokens
            ]
            if not oversized:
                break

            split_this_pass = 0
            for node in oversized:
                if _try_heading_split(node, self._max_tokens):
                    split_this_pass += 1
                    if verbose:
                        print(f"  [refiner] Heading-split '{node.title[:50]}' → {len(node.nodes)} children")
                    continue
                if self._try_llm_split(node):
                    split_this_pass += 1
                    if verbose:
                        print(f"  [refiner] LLM-split '{node.title[:50]}' → {len(node.nodes)} children")
                    continue
                _paragraph_split(node, self._max_tokens)
                split_this_pass += 1
                if verbose:
                    print(f"  [refiner] Paragraph-split '{node.title[:50]}' → {len(node.nodes)} children")

            total_split += split_this_pass
            if split_this_pass == 0:
                break

        if total_split > 0:
            assign_node_ids(tree.structure)
            logger.info("Refiner split %d oversized nodes in %d passes", total_split, iteration + 1)

        return total_split

    def _try_llm_split(self, node: TreeNode) -> bool:
        """Phase B: LLM suggests subsection titles → split at boundaries."""
        text = node.text or ""
        if len(text) < 500:
            return False

        char_limit = min(len(text), 40000)
        prompt = _SPLIT_PROMPT.format(
            title=node.title,
            char_limit=char_limit,
            text=text[:char_limit],
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            # Extract JSON array
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            titles = json.loads(content)
            if not isinstance(titles, list) or len(titles) < 2:
                return False
        except Exception:
            return False

        return _split_at_titles(node, titles)


def _try_heading_split(node: TreeNode, max_tokens: int) -> bool:
    """Phase A: Split on sub-headings already present in the node's text."""
    text = node.text or ""
    matches = list(_HEADING_RE.finditer(text))
    if len(matches) < 2:
        return False

    # Split text at heading boundaries
    parts: list[tuple[str, str]] = []  # (title, content)

    # Text before first heading
    prefix = text[:matches[0].start()].strip()
    if prefix:
        parts.append((node.title, prefix))

    for i, m in enumerate(matches):
        heading_title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            parts.append((heading_title, content))

    if len(parts) < 2:
        return False

    for title, content in parts:
        node.nodes.append(TreeNode(
            title=title,
            level=node.level + 1,
            text=content,
            start_index=node.start_index,
            end_index=node.end_index,
        ))

    if node.nodes:
        node.text = None
        _estimate_child_pages(node)
    return True


def _split_at_titles(node: TreeNode, titles: list[str]) -> bool:
    """Split node text at LLM-suggested title boundaries."""
    text = node.text or ""
    text_lower = text.lower()

    # Find approximate positions for each title
    positions: list[tuple[int, str]] = []
    for title in titles:
        idx = text_lower.find(title.lower()[:30])
        if idx >= 0:
            positions.append((idx, title))

    if len(positions) < 2:
        # Can't find titles → even split by paragraph count
        paragraphs = text.split("\n\n")
        n_chunks = min(len(titles), max(2, len(paragraphs) // 3))
        chunk_size = max(1, len(paragraphs) // n_chunks)
        for i in range(0, len(paragraphs), chunk_size):
            chunk_paras = paragraphs[i:i + chunk_size]
            chunk_text = "\n\n".join(chunk_paras).strip()
            if chunk_text:
                title_idx = min(i // chunk_size, len(titles) - 1)
                node.nodes.append(TreeNode(
                    title=titles[title_idx] if title_idx < len(titles) else f"{node.title} (part {i // chunk_size + 1})",
                    level=node.level + 1,
                    text=chunk_text,
                    start_index=node.start_index,
                    end_index=node.end_index,
                ))
    else:
        positions.sort(key=lambda x: x[0])
        for i, (pos, title) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            content = text[pos:end].strip()
            if content:
                node.nodes.append(TreeNode(
                    title=title,
                    level=node.level + 1,
                    text=content,
                    start_index=node.start_index,
                    end_index=node.end_index,
                ))

    if node.nodes:
        node.text = None
        _estimate_child_pages(node)
        return True
    return False


def _paragraph_split(node: TreeNode, max_tokens: int) -> None:
    """Phase C: Last-resort paragraph-boundary chunking."""
    paragraphs = (node.text or "").split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        pt = len(_tok(para))
        if current_tokens + pt > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += pt

    if current:
        chunks.append("\n\n".join(current))

    for i, chunk in enumerate(chunks):
        node.nodes.append(TreeNode(
            title=f"{node.title} (part {i + 1})",
            level=node.level + 1,
            text=chunk,
            start_index=node.start_index,
            end_index=node.end_index,
        ))

    if node.nodes:
        node.text = None
        _estimate_child_pages(node)


def _estimate_child_pages(parent: TreeNode) -> None:
    """Distribute parent's page range across children proportionally."""
    if not parent.nodes or not parent.start_index:
        return
    total_chars = sum(len(c.text or "") for c in parent.nodes)
    if total_chars == 0:
        return

    page_span = (parent.end_index or parent.start_index) - parent.start_index + 1
    current_page = parent.start_index

    for child in parent.nodes:
        child_chars = len(child.text or "")
        child_pages = max(1, round(page_span * child_chars / total_chars))
        child.start_index = current_page
        child.end_index = min(current_page + child_pages - 1, parent.end_index or parent.start_index)
        current_page = child.end_index + 1

    parent.nodes[-1].end_index = parent.end_index or parent.start_index
