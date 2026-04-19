"""
smartsearch/index.py — PageBM25: page-level BM25 Okapi / BMX ranking.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tok(text: str) -> list[str]:
    """Tokenize: extract lowercase \\w+ tokens."""
    return re.findall(r"\w+", text.lower())


class PageBM25:
    """
    BM25 Okapi ranking with optional BMX (Li et al. 2024) entropy weighting.

    Usage:
        bm25 = PageBM25(mode="bm25")   # or "bmx"
        bm25.build({1: "page one text", 2: "page two text", ...})
        results = bm25.query("keyword", top_k=5)  # [(page_num, score), ...]
    """

    def __init__(self, mode: str = "bm25", k1: float = 1.5, b: float = 0.75):
        self.mode = mode
        self.k1 = k1
        self.b = b
        self._docs: list[tuple[int, list[str]]] = []
        self._df: Counter = Counter()
        self._n: int = 0
        self._avgdl: float = 0.0
        # BMX extras
        self._entropy: dict[str, float] = {}
        self._alpha: float = 1.0
        self._beta: float = 0.5
        self._avg_entropy: float = 0.0

    def build(self, page_texts: dict[int, str]) -> None:
        self._docs = [(pn, _tok(text)) for pn, text in sorted(page_texts.items())]
        self._n = len(self._docs)
        self._df = Counter()
        total_len = 0
        for _, tokens in self._docs:
            total_len += len(tokens)
            for t in set(tokens):
                self._df[t] += 1
        self._avgdl = total_len / max(self._n, 1)
        if self.mode == "bmx":
            self._build_bmx_extras()

    def _build_bmx_extras(self) -> None:
        tf_all: dict[str, Counter] = {}
        for _, tokens in self._docs:
            c = Counter(tokens)
            for t, cnt in c.items():
                if t not in tf_all:
                    tf_all[t] = Counter()
                tf_all[t][_] = cnt  # type: ignore[index]

        self._entropy = {}
        for term, dist in tf_all.items():
            total = sum(dist.values())
            if total == 0:
                self._entropy[term] = 0.0
                continue
            ent = 0.0
            for cnt in dist.values():
                p = cnt / total
                if p > 0:
                    ent -= p * math.log2(p)
            self._entropy[term] = ent

        if self._entropy:
            self._avg_entropy = sum(self._entropy.values()) / len(self._entropy)
        else:
            self._avg_entropy = 0.0

        self._alpha = max(0.5, min(1.5, self._avgdl / 100))
        self._beta = 1 / math.log(self._n + 1) if self._n > 0 else 0.5

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def query(self, query_text: str, top_k: int = 5) -> list[tuple[int, float]]:
        if not self._n:
            return []
        tokens = _tok(query_text)
        if not tokens:
            return []
        scores = self._score_all(tokens)
        return scores[:top_k]

    def multi_query(
        self, queries: list[str], top_k: int = 5
    ) -> list[tuple[int, float]]:
        if not queries or not self._n:
            return []
        best: dict[int, tuple[int, float]] = {}
        for q in queries:
            for page_num, score in self._score_all(_tok(q)):
                if page_num not in best or score > best[page_num][1]:
                    best[page_num] = (page_num, score)
        scores = sorted(best.values(), key=lambda x: -x[1])
        return scores[:top_k]

    def _score_all(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        if self.mode == "bmx":
            return self._score_all_bmx(query_tokens)
        return self._score_all_bm25(query_tokens)

    def _score_all_bm25(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        scores: list[tuple[int, float]] = []
        for page_num, doc_tokens in self._docs:
            if not doc_tokens:
                continue
            dl = len(doc_tokens)
            tf = Counter(doc_tokens)
            score = 0.0
            for qt in query_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                idf = self._idf(qt)
                score += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
            if score > 0:
                scores.append((page_num, score))
        scores.sort(key=lambda x: -x[1])
        return scores

    def _score_all_bmx(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        scores: list[tuple[int, float]] = []
        for page_num, doc_tokens in self._docs:
            if not doc_tokens:
                continue
            dl = len(doc_tokens)
            rel_dl = dl / self._avgdl if self._avgdl > 0 else 1.0
            tf = Counter(doc_tokens)

            score = 0.0
            overlap = sum(1 for qt in query_tokens if qt in tf)
            s_qd = overlap / len(query_tokens) if query_tokens else 0.0

            for qt in query_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                idf = self._idf(qt)
                ent = self._entropy.get(qt, self._avg_entropy)

                bm25_part = idf * (f * (self._alpha + 1)) / (
                    f + self._alpha * rel_dl + self._alpha * self._avg_entropy
                )
                bmx_bonus = s_qd * ent * self._beta
                score += bm25_part + bmx_bonus

            if score > 0:
                scores.append((page_num, score))
        scores.sort(key=lambda x: -x[1])
        return scores
