# gnosis

Modular retrieval/QA framework — pick individual components (OCR, indexing,
retrieval, synthesis, routing) or wire them into a full pipeline.
Standalone; optional adapters for LangChain and LangGraph.

**Status:** functional, all phases implemented. 58 tests passing.

## Install

```bash
cd gnosis
pip install -e .                              # core only
pip install -e ".[langchain]"                 # + LangChain adapters
pip install -e ".[langgraph]"                 # + LangGraph adapters
```

No runtime dependency on agent-search. Heavy implementations (OCR2Engine,
TableNormalizer, TreeIndex, EntityGraph, etc.) live in ``gnosis/_impl/``.

## Layers

```
  [Parsers]     PDF → Document (pages + tables)
  [Indexers]    Document → BM25 / tree / entity-graph indexes
  [Retrievers]  query → Hits (via channels + composer)
  [Rankers]     Hits → reordered / truncated Hits
  [Synthesizers] Hits + query → Answer
  [Routers]     corpus of Documents → pick which doc(s) + delegate
```

Each layer has a Protocol (PEP 544) and plugins registered via `@register`.

## Quick start

### Use individual components

```python
from gnosis.parsers import PdfplumberParser, TableNormalizerParser

doc = PdfplumberParser().parse("doc.pdf")
doc = TableNormalizerParser().parse("doc.pdf", document=doc)
print(doc.pages[0].markdown)
```

### Build a full pipeline (smartsearch_v4 preset)

```python
from openai import OpenAI
from gnosis.presets import smartsearch_v4

client = OpenAI(base_url="...", api_key="...")
pipeline = smartsearch_v4.build(llm_client=client, llm_model="gemini-2.5-flash")
pipeline.load_document("doc.pdf")
answer = pipeline.query("câu hỏi?")
print(answer.text)
```

### Drop into LangChain

```python
from gnosis.integrations.langchain import FrameworkRetrieverAdapter

lc_retriever = FrameworkRetrieverAdapter(my_framework_retriever).as_retriever()
# Use lc_retriever with any LangChain chain
```

### Drop into LangGraph

```python
from gnosis.integrations.langgraph import make_retrieval_node

graph.add_node("retrieve", make_retrieval_node(my_retriever))
```

### Multi-document router

```python
from gnosis.routers import LLMFlatRouter
from gnosis.core.schema import DocMeta

router = LLMFlatRouter(llm_client=client, llm_model="gemini-2.5-flash")
router.add_document(DocMeta(doc_id="d1", name="doc1.pdf", ...), pipeline1)
router.add_document(DocMeta(doc_id="d2", name="doc2.pdf", ...), pipeline2)
answer = router.query("question")
```

## Directory

```
gnosis/
├── core/           # protocols, schema, pipeline, registry, config, events
├── parsers/        # pdfplumber, ocr2, text_extractor, table_normalizer,
│                   # multipage_stitcher, element_classifier, ellipsis_handler
├── indexers/       # page_bm25, tree_index, entity_graph
├── retrievers/     # 5 channels + hybrid_chatbot composer + agent_loop
├── rankers/        # weighted_merge, rrf, smart_truncate
├── synthesizers/   # chatbot_llm
├── routers/        # llm_flat_router
├── integrations/   # langchain, langgraph (optional extras)
├── presets/        # smartsearch_v4
└── shims/          # SmartSearchV4Shim
```

## Tests

```bash
python tests/unit/test_core.py            # 12/12
python tests/unit/test_parsers.py         # 7/7
python tests/unit/test_indexers.py        # 6/6
python tests/unit/test_retrievers.py      # 9/9
python tests/unit/test_rankers_synth.py   # 9/9
python tests/unit/test_preset_shim.py     # 3/3
python tests/unit/test_integrations.py    # 5/5
python tests/unit/test_routers.py         # 6/6
python tests/integration/test_pipeline_e2e.py  # 1/1
```

## Migration from smartsearch-v4

No code change needed if you use the legacy `SmartSearchV4` class — it's
now backed by the framework via `SmartSearchV4Shim`. To opt in, replace:

```python
from smartsearch.engine import SmartSearchV4
```

with:

```python
from gnosis.shims import SmartSearchV4Shim as SmartSearchV4
```

Public API (`load_document`, `query`, `readiness`, `_tree_index`,
`_entity_graph`, `_page_texts`, `_extracted_pages`, `export_structured`)
is preserved.

## Adding a new plugin

```python
from gnosis.core.registry import register
from gnosis.core.schema import Document

@register("parser", "my_parser")
class MyParser:
    def parse(self, source, document=None, **kw) -> Document:
        ...
```

Discoverable via `PluginRegistry.list("parser")`. Usable by name in YAML
or Python pipeline configs.
