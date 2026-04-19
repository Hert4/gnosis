"""
enricher.py — LLM-based retrieval-optimized summary generation for tree nodes.

Adapted from NanoIndex's enricher: generates concise summaries that boost
search recall when indexed alongside node content in BM25.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from nanoindex.models import DocumentTree, TreeNode
from nanoindex.utils.tree_ops import iter_nodes

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
Write a retrieval-optimized summary (max 60 words). Rules:
- Lead with the MOST IMPORTANT fact, number, or conclusion — not a description.
- Include specific names, codes, dates, amounts, percentages.
- If there is a table, list every row/column label.
- NEVER start with "This section" or "The section".

BAD: "This section discusses revenue recognition principles."
GOOD: "Revenue from financial activities (Account 515) includes interest, dividends, \
foreign exchange gains. Recognition upon transfer of risks/rewards."

Title: {title}
Content (first 3000 chars):
{content}

Summary:"""

_PARENT_SUMMARY_PROMPT = """\
Write a retrieval-optimized summary (max 60 words) for this section \
based on its subsections. Lead with the key finding, not a description.

Section: {title}
Subsections:
{children}

Summary:"""

_MAX_CONCURRENT = 20
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2
_MIN_LEAF_CHARS = 500  # skip short leaves — bumped 200→500 to cut LLM calls ~30-40%
_BATCH_SIZE = 8        # batch N leaf nodes per LLM call (~5-8x speedup vs per-node)


_BATCH_SUMMARY_PROMPT = """\
For EACH section below, write a retrieval-optimized summary (max 60 words each).
Lead with the MOST IMPORTANT fact/number — include names, codes, dates, amounts.
Never start with "This section" or "The section".

Output STRICT JSON array, one object per section, in the SAME ORDER as input:
[{"id": "<id>", "summary": "<60 words max>"}, ...]

Sections:
{items}

JSON:"""


class TreeEnricher:
    """Generate retrieval-optimized summaries for tree nodes."""

    def __init__(self, client: OpenAI, model: str, max_concurrent: int = _MAX_CONCURRENT):
        self._client = client
        self._model = model
        self._max_concurrent = max_concurrent

    def enrich(self, tree: DocumentTree, verbose: bool = False) -> int:
        """Generate summaries for all nodes. Modifies tree in-place.

        Strategy: leaf nodes first (batched LLM calls), then parents (no LLM).
        Returns number of summaries generated.
        """
        # Phase 1: Leaf nodes with text (skip short leaves)
        leaves = [
            n for n in iter_nodes(tree.structure)
            if not n.nodes and n.text and not n.summary and len(n.text) >= _MIN_LEAF_CHARS
        ]

        if verbose:
            print(f"  [enricher] Generating summaries for {len(leaves)} leaf nodes (batched, size={_BATCH_SIZE})...")

        generated = self._summarize_leaves_batched(leaves, verbose=verbose)

        if verbose:
            print(f"  [enricher] {generated} leaf summaries generated")

        # Phase 2: Parent nodes (synthesize from child summaries)
        parents = [
            n for n in iter_nodes(tree.structure)
            if n.nodes and not n.summary
        ]
        parent_count = 0
        for node in parents:
            if self._summarize_parent(node):
                parent_count += 1

        if verbose and parent_count:
            print(f"  [enricher] {parent_count} parent summaries synthesized")

        total = generated + parent_count
        if total > 0:
            logger.info("Enricher generated %d summaries (%d leaf, %d parent)", total, generated, parent_count)
        return total

    def _summarize_leaves_batched(self, leaves: list[TreeNode], verbose: bool = False) -> int:
        """Batch leaves into groups, one LLM call per group.

        On JSON parse failure, falls back to per-leaf calls for that batch.
        """
        import json as _json
        import re as _re

        if not leaves:
            return 0

        batches: list[list[TreeNode]] = [
            leaves[i:i + _BATCH_SIZE] for i in range(0, len(leaves), _BATCH_SIZE)
        ]

        generated = 0

        def _run_batch(batch: list[TreeNode]) -> int:
            items_str = "\n\n".join(
                f"ID: leaf_{idx}\nTitle: {n.title}\nContent: {(n.text or '')[:2500]}"
                for idx, n in enumerate(batch)
            )
            prompt = _BATCH_SUMMARY_PROMPT.format(items=items_str)

            for attempt in range(_MAX_RETRIES):
                try:
                    resp = self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150 * len(batch) + 100,
                        temperature=0.0,
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                    # Strip code fences
                    raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.IGNORECASE)
                    raw = _re.sub(r"\s*```\s*$", "", raw)
                    # Extract JSON array
                    m = _re.search(r"\[.*\]", raw, _re.DOTALL)
                    if not m:
                        raise ValueError("No JSON array in response")
                    arr = _json.loads(m.group())
                    if not isinstance(arr, list):
                        raise ValueError("Not a JSON array")

                    # Map by id → summary
                    by_id: dict[str, str] = {}
                    for item in arr:
                        if isinstance(item, dict) and "id" in item and "summary" in item:
                            by_id[str(item["id"])] = str(item["summary"])

                    count = 0
                    for idx, node in enumerate(batch):
                        key = f"leaf_{idx}"
                        summary = by_id.get(key, "").strip()
                        if summary and len(summary) > 10:
                            node.summary = summary
                            count += 1
                    return count
                except Exception as e:
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                    else:
                        logger.warning("Batch summarize failed (size=%d): %s — falling back per-leaf",
                                       len(batch), e)
                        # Fallback: per-leaf
                        fb_count = 0
                        for node in batch:
                            if self._summarize_leaf(node):
                                fb_count += 1
                        return fb_count
            return 0

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as pool:
            futures = [pool.submit(_run_batch, b) for b in batches]
            done_batches = 0
            for fut in as_completed(futures):
                try:
                    generated += fut.result()
                except Exception as e:
                    logger.warning("Batch future failed: %s", e)
                done_batches += 1
                if verbose and done_batches % 5 == 0:
                    print(f"  [enricher] batches {done_batches}/{len(batches)}, summaries so far: {generated}")

        return generated

    def _summarize_leaf(self, node: TreeNode) -> bool:
        """Generate summary for a single leaf node with retry."""
        content = (node.text or "")[:3000]
        if not content:
            return False

        prompt = _SUMMARY_PROMPT.format(title=node.title, content=content)

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.0,
                )
                summary = (resp.choices[0].message.content or "").strip()
                if summary and len(summary) > 10:
                    node.summary = summary
                    return True
                return False
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                else:
                    logger.warning("Summary failed for '%s': %s", node.title[:40], e)
                    return False
        return False

    def _summarize_parent(self, node: TreeNode) -> bool:
        """Synthesize parent summary from child summaries (no LLM call)."""
        child_summaries = [
            f"- {c.title}: {c.summary}"
            for c in node.nodes
            if c.summary and c.summary != c.title
        ]
        if not child_summaries:
            return False

        # Simple concatenation for parent — no LLM needed
        combined = "; ".join(
            c.summary for c in node.nodes[:5] if c.summary
        )
        if combined:
            node.summary = combined[:200]
            return True
        return False
