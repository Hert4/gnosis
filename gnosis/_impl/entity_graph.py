"""
entity_graph.py — Entity extraction + NetworkX graph for document search.

Extracts entities from tree nodes via LLM, builds a relationship graph,
and provides entity-based search.

Post-processing (adapted from NanoIndex):
  - Fuzzy entity resolution (exact, substring, Levenshtein)
  - Generic cross-reference detection (title matching, no hardcoded patterns)

Uses NanoIndex graph_builder for NetworkX operations.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from nanoindex.core.entity_resolver import resolve_entities
from nanoindex.core.graph_builder import (
    build_entity_to_nodes,
    build_nx_graph,
    entity_keyword_match,
    graph_expand,
)
from nanoindex.models import (
    DocumentGraph,
    DocumentTree,
    Entity,
    Relationship,
    TreeNode,
)
from nanoindex.utils.tree_ops import iter_nodes

logger = logging.getLogger(__name__)

# Generic entity types — works for any domain
_ENTITY_TYPES = (
    "Organization, Person, Document, Concept, Code, "
    "Process, Rule, Amount, Date, Location, Other"
)

_COMBINED_PROMPT = """\
Extract entities AND relationships from this document section in ONE response.

ENTITY TYPES: {entity_types}
COMMON RELATIONSHIP KEYWORDS: contains, references, related, part_of, requires, transfers_to, applies_to, depends_on, produces, inputs, outputs.

Output format (one item per line, pipe-separated). First output ALL entities, then all relationships between them:
ENTITY|name|type|short description
REL|source_entity|target_entity|keyword|short description

Minimum 3 entities. Minimum 2 relationships if ≥2 entities found.
End with DONE on its own line.

Section: {title}

{content}

Output:"""

_MAX_CONCURRENT = 25
_MIN_NODE_CHARS = 800   # bumped 300→800: skip short nodes, ~30-40% fewer LLM calls
_BATCH_SIZE = 4         # batch N nodes per LLM call — ~3-5x speedup with fallback


def _parse_response(
    text: str,
    node_id: str,
    entities: list[tuple[str, str, str, str]],
    relationships: list[tuple[str, str, str, str, str]],
) -> None:
    """Parse ENTITY|...|...|... and REL|...|...|...|... lines."""
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.upper() == "DONE":
            continue

        parts = [p.strip() for p in line.split("|")]

        if parts[0].upper() == "ENTITY" and len(parts) >= 4:
            name, etype, desc = parts[1], parts[2], parts[3]
            if name:
                entities.append((name, etype, desc, node_id))
        elif parts[0].upper() == "REL" and len(parts) >= 5:
            src, tgt, kw, desc = parts[1], parts[2], parts[3], parts[4]
            if src and tgt:
                relationships.append((src, tgt, kw, desc, node_id))


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _merge_entities(raw: list[tuple[str, str, str, str]]) -> list[Entity]:
    merged: dict[str, Entity] = {}
    for name, etype, desc, node_id in raw:
        key = _normalize_name(name)
        if key in merged:
            ent = merged[key]
            if node_id not in ent.source_node_ids:
                ent.source_node_ids.append(node_id)
            if len(desc) > len(ent.description):
                ent.description = desc
        else:
            merged[key] = Entity(
                name=name.strip(),
                entity_type=etype.strip(),
                description=desc.strip(),
                source_node_ids=[node_id],
            )
    return list(merged.values())


def _merge_relationships(raw: list[tuple[str, str, str, str, str]]) -> list[Relationship]:
    merged: dict[tuple[str, str], Relationship] = {}
    for src, tgt, kw, desc, node_id in raw:
        k1, k2 = _normalize_name(src), _normalize_name(tgt)
        key = (min(k1, k2), max(k1, k2))
        if key in merged:
            rel = merged[key]
            if node_id not in rel.source_node_ids:
                rel.source_node_ids.append(node_id)
            if len(desc) > len(rel.description):
                rel.description = desc
        else:
            merged[key] = Relationship(
                source=src.strip(),
                target=tgt.strip(),
                keywords=kw.strip(),
                description=desc.strip(),
                source_node_ids=[node_id],
            )
    return list(merged.values())


def _add_cross_references(
    tree: DocumentTree, graph_data: DocumentGraph,
) -> None:
    """Add cross-reference edges by matching node titles against other nodes' text.

    Generic approach — no hardcoded patterns. If node A's text mentions
    the title of node B, create a "references" relationship.
    """
    # Collect titles long enough to be meaningful (avoid noise)
    title_to_node: dict[str, str] = {}
    for node in iter_nodes(tree.structure):
        title = node.title.strip()
        if len(title) >= 10 and node.node_id:
            title_to_node[title] = node.node_id

    if not title_to_node:
        return

    new_rels: list[Relationship] = []
    existing_pairs = {
        (r.source.lower(), r.target.lower()) for r in graph_data.relationships
    }

    for node in iter_nodes(tree.structure):
        if not node.text or not node.node_id:
            continue
        text_lower = node.text.lower()
        for title, target_id in title_to_node.items():
            if target_id == node.node_id:
                continue  # skip self
            if title.lower() in text_lower:
                pair = (node.node_id.lower(), target_id.lower())
                reverse = (target_id.lower(), node.node_id.lower())
                if pair not in existing_pairs and reverse not in existing_pairs:
                    new_rels.append(Relationship(
                        source=node.node_id,
                        target=target_id,
                        keywords="references",
                        description=f"References '{title}'",
                        source_node_ids=[node.node_id],
                    ))
                    existing_pairs.add(pair)

    if new_rels:
        graph_data.relationships.extend(new_rels)
        logger.info("Added %d cross-reference edges", len(new_rels))


class DocumentEntityGraph:
    """Entity-relationship graph for document search.

    Extracts entities from tree nodes via LLM, builds NetworkX graph,
    provides entity-based search + BFS expansion.
    """

    def __init__(self):
        self._graph_data: DocumentGraph | None = None
        self._nx_graph = None
        self._entity_to_nodes: dict[str, set[str]] = {}

    @property
    def is_ready(self) -> bool:
        return self._graph_data is not None and len(self._graph_data.entities) > 0

    @property
    def num_entities(self) -> int:
        return len(self._graph_data.entities) if self._graph_data else 0

    @property
    def num_relationships(self) -> int:
        return len(self._graph_data.relationships) if self._graph_data else 0

    def build_from_tree(
        self,
        tree: DocumentTree,
        client: OpenAI,
        model: str,
        verbose: bool = False,
    ) -> None:
        """Extract entities from tree nodes via LLM, build graph.

        Steps:
        1. For each leaf node with text → LLM extract entities + relationships
        2. Merge duplicates across nodes
        3. Build NetworkX graph + inverted index
        """
        # Skip short nodes (ít entity giá trị, không đáng tốn LLM call)
        all_nodes = [n for n in iter_nodes(tree.structure) if n.text and len(n.text) >= _MIN_NODE_CHARS]

        if verbose:
            print(f"  [entity] Extracting entities from {len(all_nodes)} nodes (combined prompt, {_MAX_CONCURRENT} workers)...")

        raw_entities: list[tuple[str, str, str, str]] = []
        raw_relationships: list[tuple[str, str, str, str, str]] = []

        def _extract_one(node: TreeNode) -> None:
            content = (node.text or "")[:5000]
            # Combined prompt: entities + relationships in 1 LLM call
            prompt = _COMBINED_PROMPT.format(
                entity_types=_ENTITY_TYPES,
                title=node.title,
                content=content,
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3072,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content or ""
                node_entities: list[tuple[str, str, str, str]] = []
                _parse_response(text, node.node_id, node_entities, raw_relationships)
                raw_entities.extend(node_entities)
            except Exception as e:
                logger.warning("Entity extraction failed for '%s': %s", node.title, e)

        def _extract_batch(batch: list[TreeNode]) -> None:
            """Batch N nodes in 1 LLM call — falls back to per-node on parse fail."""
            sections = []
            for node in batch:
                content = (node.text or "")[:3000]
                sections.append(
                    f"=== SECTION id={node.node_id} ===\n"
                    f"Title: {node.title}\n"
                    f"Content:\n{content}"
                )
            batch_prompt = (
                f"Extract entities AND relationships for EACH section below.\n\n"
                f"ENTITY TYPES: {_ENTITY_TYPES}\n"
                f"COMMON RELATIONSHIP KEYWORDS: contains, references, related, part_of, "
                f"requires, transfers_to, applies_to, depends_on, produces, inputs, outputs.\n\n"
                "Output format — for each section emit its id header first, then entities + rels:\n"
                "BEGIN id=<id>\n"
                "ENTITY|name|type|description\n"
                "REL|source|target|keyword|description\n"
                "END\n\n"
                "Minimum 3 entities per section. Minimum 2 rels if >=2 entities.\n\n"
                + "\n\n".join(sections) + "\n\nOutput:"
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": batch_prompt}],
                    max_tokens=2048 * len(batch),
                    temperature=0.0,
                )
                text = resp.choices[0].message.content or ""

                # Split by BEGIN id=... / END markers and dispatch _parse_response
                import re as _re
                blocks = _re.split(r"BEGIN\s+id=([^\s\n]+)", text)
                # blocks: [preamble, id1, body1, id2, body2, ...]
                for i in range(1, len(blocks), 2):
                    node_id = blocks[i].strip()
                    body = blocks[i + 1] if i + 1 < len(blocks) else ""
                    # Stop at END
                    end_idx = body.find("END")
                    if end_idx != -1:
                        body = body[:end_idx]
                    node_entities: list[tuple[str, str, str, str]] = []
                    _parse_response(body, node_id, node_entities, raw_relationships)
                    raw_entities.extend(node_entities)
            except Exception as e:
                logger.warning("Entity batch failed (size=%d): %s — falling back per-node", len(batch), e)
                for node in batch:
                    _extract_one(node)

        # Extract in parallel batches
        batches: list[list[TreeNode]] = [
            all_nodes[i:i + _BATCH_SIZE] for i in range(0, len(all_nodes), _BATCH_SIZE)
        ]
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT) as pool:
            futures = {pool.submit(_extract_batch, b): b for b in batches}
            done = 0
            for future in as_completed(futures):
                done += 1
                if verbose and done % 10 == 0:
                    print(f"  [entity] batches {done}/{len(batches)} processed")
                future.result()

        entities = _merge_entities(raw_entities)
        relationships = _merge_relationships(raw_relationships)

        self._graph_data = DocumentGraph(
            doc_name=tree.doc_name,
            entities=entities,
            relationships=relationships,
        )

        # Fuzzy entity resolution (exact → suffix strip → substring → Levenshtein)
        pre_count = len(self._graph_data.entities)
        self._graph_data = resolve_entities(self._graph_data)
        post_count = len(self._graph_data.entities)
        if verbose and pre_count != post_count:
            print(f"  [entity] Resolved {pre_count} → {post_count} entities (merged {pre_count - post_count})")

        # Generic cross-reference edges (title matching)
        _add_cross_references(tree, self._graph_data)

        self._nx_graph = build_nx_graph(self._graph_data)
        self._entity_to_nodes = build_entity_to_nodes(self._graph_data)

        if verbose:
            print(
                f"  [entity] Done: {post_count} entities, "
                f"{len(self._graph_data.relationships)} relationships, "
                f"graph {self._nx_graph.number_of_nodes()} nodes / "
                f"{self._nx_graph.number_of_edges()} edges"
            )

    def find_nodes(self, query: str) -> list[tuple[str, str]]:
        """Entity keyword match → [(node_id, matched_entity_name)].

        Matches against entity names AND descriptions for better recall.
        """
        if not self._entity_to_nodes or not self._graph_data:
            return []

        # Standard name matching
        matched_ids = entity_keyword_match(query, self._entity_to_nodes)

        # Also match against entity descriptions (catches "doanh thu tài chính" → TK 515)
        query_lower = query.lower()
        for ent in self._graph_data.entities:
            desc = (ent.description or "").lower()
            name_lower = ent.name.lower()
            # Check if query words overlap with description
            if len(desc) >= 3 and desc in query_lower:
                for nid in ent.source_node_ids:
                    matched_ids.add(nid)
            # Check if description words overlap with query
            elif len(desc) >= 5:
                desc_words = set(desc.split())
                query_words = set(query_lower.split())
                overlap = desc_words & query_words
                if len(overlap) >= 2 and len(overlap) / len(desc_words) >= 0.5:
                    for nid in ent.source_node_ids:
                        matched_ids.add(nid)

        if not matched_ids:
            return []

        # Find which entities matched
        query_lower = query.lower()
        results: list[tuple[str, str]] = []
        seen: set[str] = set()
        for ent_name, node_ids in self._entity_to_nodes.items():
            if len(ent_name) >= 3 and ent_name in query_lower:
                for nid in node_ids:
                    if nid not in seen:
                        results.append((nid, ent_name))
                        seen.add(nid)

        return results

    def expand_nodes(self, seed_ids: set[str], hops: int = 1) -> set[str]:
        """BFS expand on entity graph → more tree node_ids."""
        if not self._nx_graph or not self._entity_to_nodes:
            return set()
        return graph_expand(self._nx_graph, seed_ids, self._entity_to_nodes, hops=hops)

    # ── Graph traversal methods (GraphWalks-inspired) ──

    def _fuzzy_match_entity(self, query: str) -> str | None:
        """Resolve a query string to an exact graph node name."""
        if not self._nx_graph:
            return None
        q = query.strip().lower()
        if not q:
            return None

        nodes = list(self._nx_graph.nodes())
        nodes_lower = {n: n.lower() for n in nodes}

        # 1. Exact match
        for n, nl in nodes_lower.items():
            if nl == q:
                return n

        # 2a. Query is substring of node name → prefer shortest (most specific)
        contain_best, contain_len = None, float("inf")
        for n, nl in nodes_lower.items():
            if q in nl and len(nl) < contain_len:
                contain_best, contain_len = n, len(nl)
        if contain_best:
            return contain_best

        # 2b. Node name is substring of query → prefer longest (most specific)
        inside_best, inside_len = None, 0
        for n, nl in nodes_lower.items():
            if len(nl) >= 3 and nl in q and len(nl) > inside_len:
                inside_best, inside_len = n, len(nl)
        if inside_best:
            return inside_best

        # 3. Token overlap: >50% of query tokens in node name
        q_tokens = set(q.split())
        if len(q_tokens) >= 2:
            for n, nl in nodes_lower.items():
                n_tokens = set(nl.split())
                overlap = q_tokens & n_tokens
                if len(overlap) >= 2 and len(overlap) / len(q_tokens) >= 0.5:
                    return n

        return None

    def find_entities(self, query: str) -> list[tuple[str, str, str]]:
        """Find entities matching query → [(name, type, description)]."""
        if not self._graph_data:
            return []
        q = query.strip().lower()
        if not q:
            return []

        results: list[tuple[str, str, str]] = []
        q_tokens = set(q.split())

        for ent in self._graph_data.entities:
            name_lower = ent.name.lower()
            desc_lower = (ent.description or "").lower()
            matched = False
            if len(name_lower) >= 3 and (q in name_lower or name_lower in q):
                matched = True
            elif len(desc_lower) >= 3 and (q in desc_lower or desc_lower in q):
                matched = True
            elif len(q_tokens) >= 2 and len(desc_lower) >= 5:
                d_tokens = set(desc_lower.split())
                overlap = q_tokens & d_tokens
                if len(overlap) >= 2 and len(overlap) / len(q_tokens) >= 0.5:
                    matched = True
            if matched:
                results.append((ent.name, ent.entity_type, ent.description))
        return results[:20]

    def neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        """BFS neighbors with edge metadata, grouped by distance."""
        if not self._nx_graph:
            return []
        hops = max(1, min(3, hops))
        start = self._fuzzy_match_entity(entity_name)
        if not start or not self._nx_graph.has_node(start):
            return []

        result: list[dict] = []
        visited = {start}
        frontier = {start}

        for depth in range(1, hops + 1):
            next_frontier: set[str] = set()
            for node in frontier:
                for nb in self._nx_graph.neighbors(node):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    next_frontier.add(nb)
                    edge_data = self._nx_graph[node][nb]
                    node_data = self._nx_graph.nodes.get(nb, {})
                    result.append({
                        "entity": nb,
                        "entity_type": node_data.get("entity_type", ""),
                        "edge_keywords": edge_data.get("keywords", ""),
                        "edge_description": edge_data.get("description", ""),
                        "distance": depth,
                        "via": node,
                        "source_node_ids": list(node_data.get("source_node_ids", set())),
                    })
            frontier = next_frontier
            if not frontier:
                break

        result.sort(key=lambda x: (x["distance"], x["entity"]))
        return result[:30]

    def paths(
        self, src_entity: str, dst_entity: str,
        max_hops: int = 3, max_paths: int = 5,
    ) -> list[list[dict]]:
        """Find simple paths between two entities with edge annotations."""
        if not self._nx_graph:
            return []
        max_hops = max(1, min(3, max_hops))
        src = self._fuzzy_match_entity(src_entity)
        dst = self._fuzzy_match_entity(dst_entity)
        if not src or not dst or src == dst:
            return []
        if not self._nx_graph.has_node(src) or not self._nx_graph.has_node(dst):
            return []

        import networkx as nx
        try:
            raw_paths = nx.all_simple_paths(self._nx_graph, src, dst, cutoff=max_hops)
            raw_paths = list(itertools.islice(raw_paths, max_paths))
        except (nx.NetworkXError, nx.NodeNotFound):
            return []

        raw_paths.sort(key=len)

        result: list[list[dict]] = []
        for path_nodes in raw_paths:
            annotated: list[dict] = []
            for i, node in enumerate(path_nodes):
                node_data = self._nx_graph.nodes.get(node, {})
                entry: dict = {
                    "entity": node,
                    "entity_type": node_data.get("entity_type", ""),
                    "source_node_ids": list(node_data.get("source_node_ids", set())),
                }
                if i < len(path_nodes) - 1:
                    next_node = path_nodes[i + 1]
                    edge_data = self._nx_graph[node][next_node]
                    entry["edge_keywords"] = edge_data.get("keywords", "")
                    entry["edge_description"] = edge_data.get("description", "")
                annotated.append(entry)
            result.append(annotated)
        return result

    def save(self, path: Path) -> None:
        """Serialize graph data to JSON."""
        if not self._graph_data:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._graph_data.model_dump(exclude_none=True),
                f, indent=2, ensure_ascii=False,
            )

    def load(self, path: Path) -> bool:
        """Load graph from cached JSON. Returns True if loaded."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._graph_data = DocumentGraph.model_validate(data)
            self._nx_graph = build_nx_graph(self._graph_data)
            self._entity_to_nodes = build_entity_to_nodes(self._graph_data)
            return True
        except Exception:
            return False
