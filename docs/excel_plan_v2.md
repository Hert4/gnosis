# Gnosis Excel Intelligence — Implementation Plan v2

**Author:** Đức (Hert4) — MISA Agent Platform Team
**Date:** April 2026
**Status:** Planning — **supersedes v1**
**Changes vs v1:** fixes 5 architectural mismatches + resolves Sprint 1 blocker

---

## 0. Diff vs v1 (changes rationale)

v1 có 5 điểm conflict với gnosis core protocols hoặc với 3 design decisions đã chốt. v2 sửa chúng và thay đổi Sprint 1 scope để deliverable thực sự chạy được.

| # | v1 (incorrect) | v2 (fixed) | Reason |
|---|---|---|---|
| 1 | `page.metadata = {...}`, `page.dataframe` typed attr | `page.meta["dataframe"]`, `page.meta["headers"]`, ... | `Page` dataclass chỉ có `meta: dict[str, Any]` ([schema.py:22](../gnosis/core/schema.py)) |
| 2 | Glossary modules dùng `@register("glossary", ...)` | Plain Python package `gnosis/glossary/`, inject qua constructor | Đã chốt "plain module". PluginRegistry không có layer "glossary" ([registry.py:12-15](../gnosis/core/registry.py)) |
| 3 | Tiered DuckDB (in-memory → file → parquet theo số files) | 1 shared DuckDB connection, mỗi doc là schema riêng | Đã chốt "shared DuckDB cho nhiều docs" |
| 4 | `Text2SQLRetriever` trả `SQLHits` dataclass | Trả `list[Hit]` (1 summary + N row Hits, hybrid) | Đã chốt "hybrid". Retriever protocol bắt buộc `list[Hit]` ([protocols.py:78-86](../gnosis/core/protocols.py)) |
| 5 | `pipeline.get_glossary_draft()`, `pipeline.confirm_glossary()`, `pipeline.record_feedback()` | Glossary/Feedback có API riêng (không phải method của Pipeline) | Pipeline class chỉ có `load_document()`, `query()` ([pipeline.py:35-118](../gnosis/core/pipeline.py)) |

**Scope change**: Sprint 1 bổ sung abbreviation-dict-only glossary (không LLM call) để deliverable "upload → query tiếng Việt → answer" thực sự chạy được.

---

## 1. Mục tiêu (unchanged from v1)

Bổ sung vào gnosis khả năng parse, index, và query Excel files phức tạp (multi-sheet, multi-table, merged cells, ký hiệu chuyên biệt doanh nghiệp). Thiết kế cho self-service: doanh nghiệp tự onboard, tự tạo knowledge base, không cần MISA engineer can thiệp.

### Nguyên tắc thiết kế

1. **Let Excel be Excel, let LLM be LLM** — LLM không đọc raw data. LLM hiểu intent → sinh SQL/code → engine execute chính xác.
2. **Self-service first** — Mọi knowledge (glossary, template, abbreviation) phải auto-infer được, human chỉ review/confirm.
3. **Progressive complexity** — Heuristic trước, LLM khi cần, human khi heuristic+LLM fail.
4. **Plugin architecture** — Parser/Indexer/Retriever/Router đăng ký qua `@register`. **Glossary là plain module, KHÔNG register** (xem diff #2).

---

## 2. Architecture tổng quan (updated)

```
User uploads Excel
    ↓
┌─────────────────────────────────────────────────┐
│  INGESTION PIPELINE (Pipeline class)             │
│                                                   │
│  ExcelNativeParser              @register         │
│    → AdaptiveTableDetector (Tier 1→2→3→4)         │
│    → Output: Document                             │
│        .pages[i].meta["dataframe"]                │
│        .pages[i].meta["headers"]                  │
│        .tables[i] (Table objects, canonical)      │
│                                                   │
│  DuckDBSQLIndex                 @register         │
│    → 1 SHARED con, mỗi doc = schema "doc_{id}"    │
│    → Register DataFrames as doc_{id}.{table}      │
│    → Build schema registry (stored in Document)   │
│                                                   │
│  GlossaryManager (plain module, injected)         │
│    → TemplateRegistry.match() first               │
│    → AutoInferGlossary.infer() nếu no match       │
│    → Store in GlossaryStore                       │
│    → UI review (API riêng, không qua Pipeline)    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  QUERY PIPELINE (Pipeline.query)                 │
│                                                   │
│  ExcelQueryRouter               @register         │
│    ├→ Text2SQLRetriever (70%)   @register         │
│    │   ← injected: DuckDBSQLIndex, GlossaryMgr    │
│    ├→ ExcelAgentRetriever (25%) @register         │
│    └→ Existing BM25 (5%)                          │
│                                                   │
│  Synthesizer (existing ChatbotLLMSynthesizer)     │
│  → Answer                                          │
│                                                   │
│  FeedbackStore (plain module, separate API)       │
│    .record(query_id, feedback) — called by UI     │
└─────────────────────────────────────────────────┘
```

Key changes từ v1:
- Glossary/FeedbackStore **không phải** là Pipeline stages
- DuckDB shared 1 connection duy nhất, namespace qua schema
- `page.meta["dataframe"]` thay cho `page.dataframe`

---

## 3. Module Specifications

### 3.1 `gnosis/parsers/excel_native.py`

**Protocol:** `ParserProtocol` ([protocols.py:26](../gnosis/core/protocols.py))
**Registry:** `@register("parser", "excel_native")`

**Signature:**
```python
def parse(self, source: Any, *,
          document: Document | None = None,
          **kwargs) -> Document:
    ...
```

**Output shape (fixed vs v1):**
```python
Document(
    doc_id="<sha256 of file>",
    name="bang_luong_t3.xlsx",
    total_pages=<number of sub-tables detected>,
    pages=[
        Page(
            page_num=0,
            raw_text="<markdown table>",
            markdown="<HTML table>",
            page_type="table",
            meta={                              # ← dict, not typed attrs
                "sheet": "BangLuong_T3",
                "table_idx": 0,
                "headers": ["MaNV", "HoTen", "LuongCB", ...],
                "dtypes": {"MaNV": "string", "LuongCB": "float64", ...},
                "row_count": 150,
                "dataframe": <pd.DataFrame>,    # ← in meta, not typed
                "region": {                     # ← detection output
                    "row_start": 3, "row_end": 152,
                    "col_start": 0, "col_end": 6,
                    "header_rows": [2, 3],
                    "detection_tier": 1,
                    "confidence": 0.92
                },
                "cross_refs": ["Sheet2!B5", ...]
            }
        ),
        # ... 1 Page per sub-table
    ],
    tables=[                                    # canonical Table[]
        Table(
            pages=[0],
            n_rows=150, n_cols=6,
            flat_headers=["MaNV", "HoTen", ...],
            body_rows=[["NV001", "Nguyễn Văn A", ...], ...],
            title="Bảng lương tháng 3",
            meta={"sheet": "BangLuong_T3", "table_idx": 0}
        ),
        # ... 1 Table per sub-table
    ],
    meta={
        "source_type": "excel",
        "workbook_sheets": ["BangLuong_T3", "ChamCong_T3", ...],
        "sheets_skipped": ["Chart1", "EmptySheet"]
    }
)
```

**Implementation notes:**
- `openpyxl.load_workbook(source, data_only=True)` cho values
- `openpyxl.load_workbook(source, data_only=False)` cho formulas
- Merged cells: iterate `ws.merged_cells.ranges`, unmerge + fill constituent cells
- Hierarchical headers: concat với `" | "` separator
- `Document.tables` và `Document.pages` đều populate (tables = canonical, pages = index-friendly)

**Edge cases:** (unchanged from v1)

**Effort:** 3-4 days

---

### 3.2 `gnosis/parsers/_excel_adaptive_detector.py`

**Protocol:** Internal helper (underscore prefix, NOT a plugin)
**No `@register`** — called only by ExcelNativeParser

v1 register `"excel_adaptive_detector"` dưới layer unclear — detector không phải parser top-level, nên là internal module.

**Public API:**
```python
def detect_tables(ws: openpyxl.Worksheet,
                  llm_client: Any = None,
                  tier_limit: int = 4) -> list[TableRegion]:
    """
    Returns detected table regions on a single worksheet.
    Escalates Tier 1 → 2 → 3 if confidence < 0.7.
    Tier 4 (human) is not triggered here — it's a separate UI flow.
    """

@dataclass
class TableRegion:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    header_rows: list[int]
    confidence: float       # 0-1
    detection_tier: int     # 1, 2, or 3
    signals: dict[str, Any] # which signals fired
```

**Tier logic:** (unchanged from v1 — Tier 1 heuristic, Tier 2 DBSCAN, Tier 3 LLM)

**Effort:** 5-6 days (Tier 1+2 in Sprint 2, Tier 3 in Sprint 4)

---

### 3.3 `gnosis/indexers/duckdb_sql.py`

**Protocol:** `IndexerProtocol` ([protocols.py:44](../gnosis/core/protocols.py))
**Registry:** `@register("indexer", "duckdb_sql")`

**Signature:**
```python
def build(self, document: Document, **kwargs) -> None: ...
def update(self, document: Document, **kwargs) -> None: ...
```

**State model (fixed vs v1 — shared connection):**
```python
class DuckDBSQLIndex:
    """
    1 shared in-memory DuckDB connection for all documents.
    Each document = 1 schema: "doc_{doc_id_prefix_8chars}"
    Tables within doc: "doc_xxx.{sheet_name}_tbl{idx}"

    Example:
      doc_f7a3bc12.BangLuong_T3_tbl0
      doc_f7a3bc12.ChamCong_T3_tbl0
      doc_9c1e8a4f.BHXH_Q1_tbl0
    """

    def __init__(self, persist_path: str | None = None):
        # In-memory by default. persist_path optional for session recovery.
        self.con = duckdb.connect(persist_path or ":memory:")
        self.schema_registry: dict[str, dict] = {}  # table_fqn → schema info

    def build(self, document: Document, **kwargs) -> None:
        schema = f"doc_{document.doc_id[:8]}"
        self.con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        for page in document.pages:
            if "dataframe" not in page.meta:
                continue
            df = page.meta["dataframe"]
            sheet = page.meta["sheet"]
            idx = page.meta["table_idx"]
            table_name = f"{sheet}_tbl{idx}"
            fqn = f"{schema}.{table_name}"
            # Overwrite if exists (idempotent build)
            self.con.execute(f"DROP TABLE IF EXISTS {fqn}")
            self.con.register("_tmp_df", df)
            self.con.execute(f"CREATE TABLE {fqn} AS SELECT * FROM _tmp_df")
            self.con.unregister("_tmp_df")
            self.schema_registry[fqn] = self._build_schema_info(df, page)

    def update(self, document: Document, **kwargs) -> None:
        # Incremental — same as build for a specific doc
        self.build(document, **kwargs)

    def execute(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def get_schema_prompt(self, doc_id: str | None = None,
                          max_tokens: int = 500) -> str:
        """
        Compact schema for LLM. If doc_id given → filter to that doc.
        Else → all docs (cross-doc queries allowed).
        """
```

**Cross-doc query example:**
```sql
SELECT a.MaNV, a.LuongCB, b.SoNgayCong
FROM doc_f7a3bc12.BangLuong_T3_tbl0 a
JOIN doc_f7a3bc12.ChamCong_T3_tbl0 b ON a.MaNV = b.MaNV
WHERE a.LuongCB > 20000000
```

**Multi-file capacity note:**
In-memory DuckDB handles ~1-5GB comfortably. For larger corpora, pass `persist_path="excel_corpus.duckdb"` → file-backed. Both modes use same API.

**Effort:** 2-3 days

---

### 3.4 `gnosis/glossary/` — Plain Python Package

**NOT a plugin layer.** No `@register` calls. Injected into retrievers via constructor.

```
gnosis/glossary/
├── __init__.py           # exports GlossaryManager, GlossaryStore
├── auto_infer.py         # AutoInferGlossary (LLM + abbreviation dict)
├── abbreviations.py      # VN_ACCOUNTING_ABBREVIATIONS dict (Sprint 1)
├── template_registry.py  # TemplateRegistry
├── store.py              # GlossaryStore (SQLite persistence)
├── manager.py            # GlossaryManager (coordinates above 3)
└── feedback.py           # FeedbackStore (separate concern, same package)
```

**GlossaryManager** — facade that retrievers consume:
```python
class GlossaryManager:
    def __init__(self,
                 store: GlossaryStore,
                 template_registry: TemplateRegistry | None = None,
                 auto_infer: AutoInferGlossary | None = None):
        ...

    # Ingestion-time API
    def ingest_schema(self, doc_id: str, schema_registry: dict) -> dict:
        """
        Run template match → fallback to auto_infer → store as 'pending'.
        Returns draft glossary for UI review.
        """

    def confirm(self, doc_id: str, reviewed_entries: list[GlossaryEntry],
                user: str) -> None:
        """Called after human review."""

    # Query-time API (used by Text2SQLRetriever)
    def get_prompt_fragment(self, doc_id: str) -> str:
        """Compact glossary text for SQL-gen prompt."""

    def get_code_map(self, doc_id: str, column: str) -> dict[str, str] | None:
        """{'T': 'Thử việc', ...}"""
```

**AutoInferGlossary** (Sprint 1 = abbreviation-only, Sprint 2 = + LLM):
```python
class AutoInferGlossary:
    def __init__(self, llm_client=None, llm_model=None,
                 abbreviation_dict: dict = VN_ACCOUNTING_ABBREVIATIONS):
        self.abbr = abbreviation_dict
        self.llm = llm_client  # optional

    def infer(self, schema_registry: dict) -> list[GlossaryEntry]:
        # Phase 1 (always runs, no LLM): abbreviation lookup per column name
        entries = [self._infer_from_abbr(col, info)
                   for col, info in schema_registry.items()]

        # Phase 2 (if self.llm is not None): fill gaps with LLM
        low_conf = [e for e in entries if e.confidence < 0.6]
        if low_conf and self.llm is not None:
            entries = self._llm_refine(entries, low_conf, schema_registry)
        return entries
```

Sprint 1 ships Phase 1 only (no LLM). Sprint 2 adds Phase 2.

**Effort:** Sprint 1 scope = 1 day. Sprint 2 LLM scope = 2-3 days.

---

### 3.5 Glossary store + template registry + feedback

Unchanged from v1 in content — just relocated out of `@register` and into plain modules. SQLite schemas identical to v1 §3.5, §3.6, §3.7.

**Effort:** 3-4 days each (Sprints 2-4)

---

### 3.6 `gnosis/retrievers/text2sql.py`

**Protocol:** `RetrieverProtocol` ([protocols.py:78](../gnosis/core/protocols.py))
**Registry:** `@register("retriever", "text2sql")`

**Signature (fixed vs v1 — returns `list[Hit]`, not SQLHits):**
```python
class Text2SQLRetriever:
    name = "text2sql"

    def __init__(self, sql_index: DuckDBSQLIndex,
                 glossary: GlossaryManager,
                 feedback: FeedbackStore | None,
                 llm_client, llm_model: str):
        # Dependencies injected explicitly — no magic lookup
        ...

    def retrieve(self, query: str, *, top_k: int = 30,
                 context: dict | None = None) -> list[Hit]:
        doc_id = (context or {}).get("doc_id", "")
        sql = self._generate_sql(query, doc_id)
        try:
            df = self.sql_index.execute(sql)
        except Exception as e:
            sql_fixed = self._self_correct(sql, str(e), doc_id)
            df = self.sql_index.execute(sql_fixed)
            sql = sql_fixed

        return self._df_to_hits(df, sql, query, doc_id, top_k)
```

**Hybrid Hit packing (fixed vs v1):**
```python
def _df_to_hits(self, df, sql, query, doc_id, top_k) -> list[Hit]:
    hits = []

    # Hit 0: SUMMARY — gives synthesizer full context
    summary_md = self._render_summary(df, sql)  # markdown: shape, sql, head(5)
    hits.append(Hit(
        chunk_id=f"{doc_id}:sql:summary:{hash_sql(sql)}",
        doc_id=doc_id,
        text=summary_md,
        score=1.0,
        channel="text2sql",
        meta={
            "kind": "sql_summary",
            "sql": sql,
            "row_count": len(df),
            "columns": list(df.columns),
            "dataframe": df,  # full df stashed here for downstream use
        }
    ))

    # Hit 1..N: ROW-LEVEL (up to top_k-1, for citation granularity)
    for i, row in df.head(top_k - 1).iterrows():
        row_md = self._render_row_markdown(row, df.columns)
        hits.append(Hit(
            chunk_id=f"{doc_id}:sql:row:{hash_sql(sql)}:{i}",
            doc_id=doc_id,
            text=row_md,
            score=1.0 - (i * 0.001),  # preserve order, all relevant
            channel="text2sql",
            meta={"kind": "sql_row", "sql": sql, "row_index": i}
        ))

    return hits
```

**Synthesizer compatibility:**
`ChatbotLLMSynthesizer` concat `.text` của Hits → prompt. Summary Hit ensure LLM có schema + SQL context, row Hits cho LLM cite rows cụ thể.

**Effort:** 2-3 days

---

### 3.7 `gnosis/retrievers/excel_agent.py`

Unchanged in spec from v1 §3.9, but trả `list[Hit]` theo cùng hybrid pattern §3.6 ở trên. Tools (`sql_query`, `python_analyze`, `inspect_data`) giữ nguyên.

**Effort:** 4-5 days

---

### 3.8 `gnosis/sandbox/secure_executor.py`

Unchanged from v1 §3.10. **Windows caveat**: `resource.setrlimit` không có trên Windows. Default dev path = subprocess + timeout (no memory limit). Production = E2B.dev hoặc Docker. Document trong docstring.

**Effort:** 3-4 days

---

### 3.9 `gnosis/routers/excel_query_router.py`

**Protocol:** `RouterProtocol` ([protocols.py:120](../gnosis/core/protocols.py))
**Registry:** `@register("router", "excel_query")`

**Caveat**: RouterProtocol requires `add_document(doc_meta, pipeline)`, `route(query)`, `query(query, ...)`. v1 only specified `route()`. v2 implements all three: `add_document` register per-doc pipeline; `route` picks retriever; `query` delegates to chosen retriever's pipeline.

Alternative: nếu chỉ cần pick retriever within single pipeline (không multi-doc routing), implement as a **pre-retrieval stage wrapper** thay vì RouterProtocol. **Decision**: Sprint 4 implement Router as retriever-selector wrapper inside Pipeline (not RouterProtocol). Đơn giản hơn cho MVP.

**Effort:** 1-2 days

---

### 3.10 `gnosis/presets/excel_qa.py`

**Returns plain `Pipeline`** — không có custom methods (`get_glossary_draft`, v.v.).

```python
def build(
    llm_client,
    llm_model: str = "gemini-2.5-flash",
    persist_path: str | None = None,
    template_registry_path: str | None = None,
    feedback_store_path: str | None = None,
    tenant_id: str = "default",
) -> tuple[Pipeline, GlossaryManager, FeedbackStore]:
    """
    Returns (pipeline, glossary_mgr, feedback_store).
    Caller uses pipeline for load/query, glossary_mgr for review UI,
    feedback_store for recording feedback.

    Usage:
        pipeline, glossary, feedback = excel_qa.build(llm_client=client)
        pipeline.load_document("file.xlsx")

        # UI review flow (separate API, NOT through pipeline)
        draft = glossary.ingest_schema(doc_id, schema)
        # ... user reviews ...
        glossary.confirm(doc_id, reviewed, user="admin")

        # Query
        answer = pipeline.query("Tổng lương phòng kế toán?")

        # Feedback (separate API)
        feedback.record(query="...", sql=answer.meta["sql"], correct=True)
    """
    # Shared instances
    sql_index = DuckDBSQLIndex(persist_path=persist_path)
    glossary_store = GlossaryStore(tenant_id=tenant_id)
    template_registry = TemplateRegistry(path=template_registry_path)
    auto_infer = AutoInferGlossary(llm_client=llm_client, llm_model=llm_model)
    glossary_mgr = GlossaryManager(
        store=glossary_store,
        template_registry=template_registry,
        auto_infer=auto_infer,
    )
    feedback_store = FeedbackStore(path=feedback_store_path, tenant_id=tenant_id)

    pipeline = (PipelineBuilder()
        .parse(ExcelNativeParser(glossary_mgr=glossary_mgr))
        .index(sql_index)
        .retrieve(Text2SQLRetriever(
            sql_index=sql_index,
            glossary=glossary_mgr,
            feedback=feedback_store,
            llm_client=llm_client,
            llm_model=llm_model,
        ))
        .synthesize(ChatbotLLMSynthesizer(
            llm_client=llm_client,
            llm_model=llm_model,
        ))
        .build())

    return pipeline, glossary_mgr, feedback_store
```

Returns tuple vì glossary/feedback không nên hide trong Pipeline. User code trong MISA web app gọi 3 objects riêng biệt theo đúng separation of concerns.

**Effort:** 1-2 days

---

## 4. Dependencies

Unchanged from v1 §4.

```toml
[project.optional-dependencies]
excel = ["openpyxl>=3.1.0", "duckdb>=1.0.0", "scikit-learn>=1.3.0"]
excel-full = [
    "openpyxl>=3.1.0", "duckdb>=1.0.0", "scikit-learn>=1.3.0",
    "sentence-transformers>=2.2.0",
]
```

Note: `sentence-transformers` nặng (~500MB). Alternative: dùng LLM embedding API (Gemini embedding, OpenAI text-embedding-3-small) — zero install footprint. Quyết định ở Sprint 2 khi implement FeedbackStore.

---

## 5. Test Plan

Unchanged from v1 §5 except:
- `test_excel_parser.py::test_page_meta_dataframe` — assert DataFrame in `page.meta["dataframe"]`, not `page.dataframe`
- `test_duckdb_index.py::test_schema_namespacing` — assert tables under `doc_{id}.{name}` schema
- `test_text2sql.py::test_hit_shape` — assert returns `list[Hit]` with 1 summary + N rows
- `test_glossary.py` — no registry lookup tests (it's not a plugin)

**Fixture need** (unblocker): MISA sample Excels (`misa_bangluong_sample.xlsx`, `misa_bhxh_sample.xlsx`) — **Đức cần provide 3-5 samples hoặc confirm dùng synthetic before Sprint 1 starts**.

---

## 6. Sprint Plan (REVISED)

### Sprint 1: Foundation + minimal glossary (Week 1-2) — **changed**

| Task | Module | Effort | Priority |
|---|---|---|---|
| ExcelNativeParser (basic: merged cells, blank-row detection only) | `parsers/excel_native.py` | 3d | P0 |
| DuckDBSQLIndex (shared con, namespaced schemas) | `indexers/duckdb_sql.py` | 2d | P0 |
| **AutoInferGlossary Phase 1 (abbreviation dict only)** | `glossary/auto_infer.py`, `glossary/abbreviations.py` | 1d | **P0** (added) |
| **GlossaryStore + GlossaryManager skeleton** | `glossary/store.py`, `glossary/manager.py` | 1d | **P0** (added) |
| Text2SQLRetriever (basic + hybrid Hit packing) | `retrievers/text2sql.py` | 2d | P0 |
| Minimal preset `excel_qa.build()` | `presets/excel_qa.py` | 1d | P0 |
| Unit tests + 3 synthetic fixtures | `tests/` | 2d | P0 |
| E2E: upload simple xlsx → VN query → correct answer | `tests/integration/` | 1d | P0 |

**Deliverable v2:** Upload 1 simple Excel với tên column viết tắt VN (LCB, BHXH, MaNV) → hỏi "Tổng lương cơ bản phòng kế toán?" → correct answer. No LLM-based glossary inference, no sub-table detection beyond blank rows.

### Sprint 2: Intelligence

| Task | Module | Effort |
|---|---|---|
| AutoInferGlossary Phase 2 (LLM refinement) | `glossary/auto_infer.py` | 2d |
| AdaptiveTableDetector Tier 1+2 | `parsers/_excel_adaptive_detector.py` | 4d |
| FeedbackStore (basic record + few-shot retrieval) | `glossary/feedback.py` | 3d |

### Sprint 3: Templates + Advanced retrieval

| Task | Module | Effort |
|---|---|---|
| TemplateRegistry | `glossary/template_registry.py` | 3d |
| ExcelAgentRetriever (ReAct) | `retrievers/excel_agent.py` | 4d |
| SecurePythonSandbox (subprocess baseline) | `sandbox/secure_executor.py` | 3d |

### Sprint 4: Collaboration + Routing

| Task | Module | Effort |
|---|---|---|
| CollaborativeGlossaryStore (roles + approval) | extension of `glossary/store.py` | 3d |
| ExcelQueryRouter (as retriever-selector) | `routers/excel_query_router.py` | 2d |
| AdaptiveTableDetector Tier 3 (LLM) | `parsers/_excel_adaptive_detector.py` | 2d |

### Sprint 5: Eval + Hardening (unchanged)

---

## 7. Success Metrics, 8. Risks, 9. Open Questions

Unchanged from v1 §7-§9.

---

## 10. Confirmation Checklist (signoff before Sprint 1)

- [ ] Provide 3-5 MISA Excel samples (hoặc OK với synthetic-only cho Sprint 1)
- [ ] Confirm embedding choice for FeedbackStore: local sentence-transformers vs LLM API
- [ ] Confirm VN_ACCOUNTING_ABBREVIATIONS dict content (v2 will draft 50+ entries; bạn review)
- [ ] Confirm doc_id scheme: sha256 of file bytes (default proposed) vs filename-based
- [ ] Confirm Sprint 1 scope đủ cho demo nội bộ

---

## 11. SQL Backend Abstraction (pluggable DB)

**Rationale:** gnosis không nên lock Excel pipeline vào DuckDB. Customer có thể đã có Postgres/MySQL/SQL Server. Abstract thành `SQLBackend` protocol — ship 2-3 common backends, user `@register` cái của họ.

Gnosis registry đã có layer `"backend"` ([registry.py:12-15](../gnosis/core/registry.py)) — không cần thêm gì core.

### 11.1 Protocol

```python
# gnosis/core/protocols.py — new protocol
@runtime_checkable
class SQLBackendProtocol(Protocol):
    name: str
    dialect: str  # "duckdb" | "postgres" | "sqlite" | "mysql" | "mssql"

    def connect(self, **kwargs) -> Any: ...

    def create_namespace(self, name: str) -> None:
        """Create schema/database for a document's tables."""

    def load_dataframe(self, df: pd.DataFrame,
                       namespace: str, table: str) -> None:
        """Load Excel sub-table into backend."""

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute SELECT, return results as DataFrame."""

    def get_schema_info(self, namespace: str | None = None) -> dict:
        """Introspect: {fqn: {columns, dtypes, row_count, sample}}"""

    def attach_external(self, conn_string: str, alias: str) -> None:
        """Optional: federate external DB (DuckDB ATTACH pattern)."""
```

### 11.2 Default implementations (shipped)

| Backend | File | Dialect | Use case |
|---|---|---|---|
| DuckDB | `gnosis/backends/duckdb.py` | `duckdb` | Default, local, fast, federation-capable |
| SQLite | `gnosis/backends/sqlite.py` | `sqlite` | File-based, legacy compat, zero-install |
| Postgres | `gnosis/backends/postgres.py` | `postgres` | Enterprise, production scale (SQLAlchemy) |

### 11.3 User-plug pattern

```python
# user code
from gnosis.core.registry import register
from gnosis.core.protocols import SQLBackendProtocol

@register("backend", "mysql")
class MySQLBackend:
    name = "mysql"
    dialect = "mysql"
    def __init__(self, host, user, password, database): ...
    def connect(self): return pymysql.connect(...)
    def execute(self, sql): ...
    # ... implement protocol
```

### 11.4 Integration with indexer + retriever

`DuckDBSQLIndex` renames thành generic `SQLIndex`, takes backend via constructor:

```python
@register("indexer", "sql")
class SQLIndex:
    def __init__(self, backend: SQLBackendProtocol):
        self.backend = backend
    def build(self, document: Document, **kw):
        self.backend.create_namespace(f"doc_{document.doc_id[:8]}")
        for page in document.pages:
            if "dataframe" in page.meta:
                self.backend.load_dataframe(
                    page.meta["dataframe"],
                    namespace=f"doc_{document.doc_id[:8]}",
                    table=f"{page.meta['sheet']}_tbl{page.meta['table_idx']}"
                )
```

`Text2SQLRetriever` reads `backend.dialect` để chọn prompt:

```python
def _build_prompt(self, query, schema):
    return f"Generate {self.index.backend.dialect} SQL. ..."
```

### 11.5 Sprint impact

| Sprint | Addition | Effort |
|---|---|---|
| Sprint 1 | Protocol + DuckDBBackend only | +1d (protocol design) |
| Sprint 2 | SQLiteBackend | 1d |
| Sprint 3 | PostgresBackend | 2d |
| Future | User-registered backends | 0 (just documentation) |

---

## 12. Framework Integration (standalone use + LangChain/LangGraph/any agent)

**Principle**: mọi Excel module **phải usable standalone**, không lock vào `Pipeline`. Gnosis đã có pattern ở [gnosis/integrations/langchain.py](../gnosis/integrations/langchain.py) và [langgraph.py](../gnosis/integrations/langgraph.py) cho Parser/Retriever/Synthesizer — v2 bổ sung cho SQL-specific tools và Glossary.

### 12.1 Standalone usability check

Every module must pass this test — usable **without** constructing a Pipeline:

```python
# Pure gnosis modules, no Pipeline, no preset
from gnosis.parsers.excel_native import ExcelNativeParser
from gnosis.backends.duckdb import DuckDBBackend
from gnosis.indexers.sql import SQLIndex
from gnosis.glossary import GlossaryManager, AutoInferGlossary, GlossaryStore
from gnosis.retrievers.text2sql import Text2SQLRetriever

parser = ExcelNativeParser()
doc = parser.parse("bang_luong.xlsx")

backend = DuckDBBackend()
index = SQLIndex(backend=backend)
index.build(doc)

glossary = GlossaryManager(
    store=GlossaryStore(),
    auto_infer=AutoInferGlossary(),  # no LLM = abbreviation dict only
)
glossary.ingest_schema(doc.doc_id, index.schema_registry)

retriever = Text2SQLRetriever(
    index=index, glossary=glossary,
    llm_client=my_client, llm_model="any-model",
)
hits = retriever.retrieve("Tổng lương phòng kế toán?")
```

Anything blocking this == architectural bug.

### 12.2 Extended `gnosis/integrations/langchain.py`

Existing: `FrameworkRetrieverAdapter`, `FrameworkDocumentLoader`. Add:

```python
# New: expose SQL index as LangChain Tool for external agents
class SQLIndexTool:
    """Wrap SQLIndex.execute() as LangChain @tool.

    Usage:
        from gnosis.integrations.langchain import SQLIndexTool
        tools = SQLIndexTool(index=my_sql_index).as_tools()
        # returns: [sql_query_tool, list_tables_tool, inspect_table_tool]
        agent = create_react_agent(llm, tools)
    """

    def __init__(self, index: SQLIndex, glossary: GlossaryManager | None = None):
        self.index = index
        self.glossary = glossary

    def as_tools(self) -> list:
        """Return list of LangChain StructuredTool objects."""
        from langchain_core.tools import tool
        # Returns 3-4 tools: sql_query, list_tables, inspect_table, glossary_lookup

    def as_openai_functions(self) -> list[dict]:
        """Return OpenAI function-calling schema (for OpenAI SDK direct use)."""

    def as_anthropic_tools(self) -> list[dict]:
        """Return Anthropic tool-use schema (for Anthropic SDK direct use)."""


class GlossaryTool:
    """Wrap GlossaryManager as LangChain @tool for term lookup."""
    # similar pattern
```

### 12.3 Extended `gnosis/integrations/langgraph.py`

Existing: `make_parse_node`, `make_retrieval_node`, `make_synthesis_node`. Add:

```python
def make_excel_toolbelt_node(
    index: SQLIndex,
    glossary: GlossaryManager | None = None,
    tools_key: str = "excel_tools",
) -> Callable[[dict], dict]:
    """Inject Excel tools into LangGraph state.

    Usage:
        graph.add_node("setup_tools", make_excel_toolbelt_node(index, glossary))
        # downstream node reads state["excel_tools"]
    """


def make_sql_execute_node(
    index: SQLIndex,
    sql_key: str = "sql",
    output_key: str = "sql_result",
) -> Callable[[dict], dict]:
    """Node that executes SQL from state."""
```

### 12.4 Framework-agnostic tool definitions

Beyond LangChain/LangGraph, expose **plain Python callables** + standard schemas so users can plug into any framework (CrewAI, AutoGen, LlamaIndex, raw OpenAI/Anthropic SDK):

```python
# gnosis/integrations/tools.py — new file, zero framework deps

class ExcelToolset:
    """Framework-agnostic tool definitions.
    Each tool is (callable, JSON schema) — adaptable to any framework.
    """
    def __init__(self, index: SQLIndex, glossary: GlossaryManager | None = None):
        ...

    @property
    def tools(self) -> list[dict]:
        """Return OpenAI-style function definitions + Python callables.
        Schema: [{"name": str, "description": str, "parameters": dict, "fn": callable}]
        """
        return [
            {"name": "sql_query", "description": "...", "parameters": {...},
             "fn": self._sql_query},
            {"name": "list_tables", ...},
            {"name": "inspect_table", ...},
            {"name": "glossary_lookup", ...},
        ]
```

Users of any framework can then iterate `toolset.tools` và adapt theo convention của framework họ dùng.

### 12.5 Hit.meta serialization rule (critical)

Text2SQLRetriever's summary Hit `meta["dataframe"]` chứa `pd.DataFrame` — **KHÔNG JSON-serializable**. Quy tắc:

- `Hit.meta["dataframe_markdown"]: str` — markdown rendering, safe for serialization
- `Hit.meta["dataframe_fqn"]: str` — fully-qualified table name, consumer re-query backend
- `Hit.meta["_dataframe"]: pd.DataFrame` — underscore prefix = "framework adapters may drop this"
- LangChain adapter ([langchain.py:66](../gnosis/integrations/langchain.py)) passes `**h.meta` into `LCDocument.metadata` — filter out `_`-prefixed keys before spread

Impact: §3.6 Text2SQLRetriever spec cập nhật meta field names theo quy tắc này.

### 12.6 Sprint impact

| Sprint | Addition | Effort |
|---|---|---|
| Sprint 1 | Test standalone usability (no Pipeline) | 0.5d |
| Sprint 4 | `SQLIndexTool`, `GlossaryTool` in langchain.py | 1d |
| Sprint 4 | LangGraph node factories for Excel | 1d |
| Sprint 4 | `ExcelToolset` framework-agnostic | 1d |
| Sprint 5 | Example: `examples/with_excel_agent.py` (LangGraph), `examples/with_openai_agent.py` | 1d |
