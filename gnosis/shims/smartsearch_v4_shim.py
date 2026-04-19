"""SmartSearchV4Shim — wraps the framework preset in the legacy class shape.

Exposes the same attributes and methods as ``smartsearch.engine.SmartSearchV4``
so that ``smartsearch-api/server.py`` and other callers work unchanged:

    engine.load_document(pdf_path, force=False) -> dict
    engine.query(question, on_event=None, chat_history=None) -> str
    engine.readiness                                           -> int
    engine._tree_index / _entity_graph / _page_texts / _extracted_pages
    engine.export_structured()                                 -> dict
    engine.processing_state                                     -> dict

Not a drop-in for the visual agent_loop path (scan-only docs) — that
remains available via the legacy class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnosis.core.pipeline import Pipeline
from gnosis.core.schema import Document
from gnosis.presets import smartsearch_v4 as _preset


class SmartSearchV4Shim:
    """Framework-backed drop-in for ``smartsearch.engine.SmartSearchV4``."""

    def __init__(
        self,
        *,
        answer_base_url: str,
        answer_model: str,
        answer_api_key: str,
        chunking_base_url: str,
        chunking_model: str,
        chunking_api_key: str,
        storage_dir: str | Path,
        text_extractor=None,           # accepted for API compat; ignored
        ocr2_api_base: str = "http://127.0.0.1:30000",
        render_dpi: int = 250,
        max_steps: int = 25,
        search_mode: str = "bmx",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        from openai import OpenAI

        self._answer_client = OpenAI(base_url=answer_base_url, api_key=answer_api_key)
        self._chunking_client = OpenAI(base_url=chunking_base_url, api_key=chunking_api_key)
        self._answer_model = answer_model
        self._chunking_model = chunking_model
        self._storage_dir = Path(storage_dir)
        self._verbose = verbose

        self._pipeline: Pipeline = _preset.build(
            llm_client=self._answer_client,
            llm_model=answer_model,
            ocr2_api_base=ocr2_api_base,
            render_dpi=render_dpi,
            bm25_mode=search_mode,
        )

        # Legacy attributes populated after load_document
        self._page_texts: dict[int, str] = {}
        self._extracted_pages: set[int] = set()
        self._total_pages = 0
        self._track: str = ""
        self._doc_dir: Path = Path()
        self._readiness = 0

    # ─────────────── Legacy API — load ───────────────

    def load_document(self, pdf_path: str, force: bool = False) -> dict:
        import time
        t0 = time.time()

        doc = self._pipeline.load_document(pdf_path)
        self._page_texts = {p.page_num: p.markdown for p in doc.pages}
        self._extracted_pages = {p.page_num for p in doc.pages if p.markdown}
        self._total_pages = doc.total_pages
        n_text = sum(1 for p in doc.pages if p.page_type == "text")
        n_scan = self._total_pages - n_text
        self._track = "text" if n_text > self._total_pages / 2 else "scan"
        self._doc_dir = self._storage_dir / Path(pdf_path).stem
        self._doc_dir.mkdir(parents=True, exist_ok=True)
        self._readiness = 2  # framework pipeline runs all parsers/indexers synchronously

        return {
            "track": self._track,
            "total_pages": self._total_pages,
            "readiness": self._readiness,
            "text_pages": n_text,
            "scan_pages": n_scan,
            "load_time_ms": int((time.time() - t0) * 1000),
        }

    # ─────────────── Legacy API — query ───────────────

    def query(
        self,
        question: str,
        on_event=None,
        chat_history: list[dict] | None = None,
    ) -> str:
        # Pipeline.query needs page_texts in context for channels to build hits
        ctx_extras = {
            "doc_id": self._pipeline.document.doc_id if self._pipeline.document else "",
            "page_texts": self._page_texts,
        }
        # Patch the retriever's call path by passing context via the
        # retriever directly; Pipeline.query forwards context.state.
        # Simpler: invoke channels + retriever + synth by hand so we can thread ctx.
        retriever = self._pipeline._retriever  # set by preset
        ranker = self._pipeline.rankers[0] if self._pipeline.rankers else None
        synth = self._pipeline.synthesizers[0]

        hits = retriever.retrieve(question, context=ctx_extras)
        if ranker is not None:
            hits = ranker.rank(hits, query=question, context=ctx_extras)
        answer = synth.synthesize(question, hits, chat_history=chat_history,
                                  context=ctx_extras)
        return answer.text

    # ─────────────── Legacy attributes (read-only surface) ───────────────

    @property
    def readiness(self) -> int:
        return self._readiness

    @property
    def _tree_index(self):
        ti = getattr(self._pipeline, "_tree_idx", None)
        return ti.raw if ti else None

    @property
    def _entity_graph(self):
        eg = getattr(self._pipeline, "_entity_idx", None)
        return eg.raw if eg else None

    @property
    def _bm25(self):
        idx = getattr(self._pipeline, "_bm25_idx", None)
        return idx.raw if idx else None

    @property
    def processing_state(self) -> dict[str, Any]:
        doc = self._pipeline.document
        return {
            "readiness": self._readiness,
            "extracted_pages": len(self._extracted_pages),
            "total_pages": self._total_pages,
            "tree_ready": bool(self._tree_index and self._tree_index.is_ready),
            "graph_ready": bool(self._entity_graph and self._entity_graph.is_ready),
            "errors": doc.meta.get("text_extractor_errors", []) if doc else [],
        }

    # ─────────────── export ───────────────

    def export_structured(self) -> dict:
        doc = self._pipeline.document
        if not doc:
            return {}
        return {
            "doc_name": doc.name,
            "total_pages": doc.total_pages,
            "sections": doc.sections,
            "tables": [
                {
                    "pages": t.pages,
                    "n_rows": t.n_rows,
                    "n_cols": t.n_cols,
                    "flat_headers": t.flat_headers,
                    "body_rows": t.body_rows,
                }
                for t in doc.tables
            ],
            "entities": doc.entities,
            "ellipsis_pages": doc.meta.get("ellipsis_pages", {}),
        }
