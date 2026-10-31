# FoodSafe · Food Safety Knowledge Base System

An enterprise RAG-powered Q&A platform for the catering / food industry. It unifies six knowledge
bases — food-safety regulations, SOP procedures, incident-handling cases, nutrition references,
supplier qualifications, and high-frequency FAQs — to deliver second-level regulatory lookup,
SOP guidance, and incident-handling support for QC staff, store managers, and frontline operators.

- Requirements baseline: [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) (parsed from scenario doc 1.47)
- Getting started: [`RUN.md`](RUN.md)
- Chinese README: [`README.md`](README.md)

---

## Directory Layout

```
FoodSafetySystem/
├── app.py                      # FastAPI entrypoint (HTTP + WebSocket + static hosting)
├── build_kb.py                 # Knowledge-base build entry (load → split → embed → index)
├── run_cli.py                  # CLI Q&A entrypoint (verify RAG quality)
├── qa_system.py                # QA assembly layer (LLM + vector store + session history)
├── config.ini / .env           # Central configuration
├── requirements.txt
│
├── base/                       # Infrastructure
│   ├── config.py               #   Centralized config (models / Milvus / paths / retrieval)
│   └── logger.py               #   Logging
│
├── core/                       # RAG core logic
│   ├── text_splitter.py        #   Chinese recursive splitter + per-KB separators
│   ├── document_processor.py   #   Document loading & parent/child splitting
│   ├── vector_store.py         #   Embedding, storage, hybrid search, rerank
│   ├── prompts.py              #   Prompt templates (RAG / HyDE / SubQuery)
│   └── rag_system.py           #   Main RAG flow (routing → retrieval → assembly → generation)
│
├── data/                       # Knowledge data (directory name = KB type)
│   ├── regulation_data/  sop_data/  case_data/
│   ├── nutrition_data/   supplier_data/  faq_data/
│
├── static/index.html           # Q&A web UI
├── models/                     # Local models (bge-m3 / bge-reranker-large)
└── docs/REQUIREMENTS.md        # Requirements baseline
```

## Module Responsibilities

| Module | Responsibility | Key design |
|--------|----------------|------------|
| `base/config.py` | Single source of truth for config | Priority: `.env` > `config.ini` > code defaults |
| `core/text_splitter.py` | Chinese recursive splitting | Dedicated separators per KB type; preserves clause / step boundaries |
| `core/document_processor.py` | Loading + two-level splitting | KB type inferred from directory name; parent = context, child = retrieval precision |
| `core/vector_store.py` | Embedding, indexing, retrieval | BGE-M3 dense + sparse; single Milvus collection discriminated by `kb_type`; hybrid search + reranker |
| `core/prompts.py` | Prompt templates | Main RAG template enforces citation and refusal; HyDE / SubQuery for retrieval enhancement |
| `core/rag_system.py` | RAG orchestration | Rule-based strategy routing → retrieval → threshold check → context assembly → streaming generation |
| `qa_system.py` | Assembly layer | Wires LLM / vector store / RAG flow / session history + greeting short-circuit; exposes `ask` & `ask_stream` |
| `build_kb.py` | Offline indexing | Incremental, full rebuild, or single-KB import |
| `app.py` | Service entrypoint (single) | Assembly & forwarding only; `POST /api/query` (non-stream), `WS /api/stream` (stream) |

## Alignment with the reference project (D:/heima/RAG)

Same flat layering (`base/` + `core/` + root entry scripts), same `config.ini` + `.env` dual-track
configuration, same stack (Python + LangChain splitting + Milvus + BGE-M3 + BGE-Reranker +
DeepSeek OpenAI-compatible API + FastAPI), same Chinese-comment code style.

**Deliberate scope cut**: MySQL/Redis/BM25 three-tier cache, OCR loaders, BERT intent classifier,
and RAGAS evaluation are not included — the single-collection + `kb_type` filter replaces the
reference project's multi-collection physical isolation.

## Quick Start

See [`RUN.md`](RUN.md):

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
mkdir -p models
cp -r D:/heima/RAG/rag_qa/models/bge-m3 models/
cp -r D:/heima/RAG/rag_qa/models/bge-reranker-large models/
python build_kb.py
python run_cli.py -q "这批鸡胸肉到货温度8℃还能用吗"
python app.py          # open http://127.0.0.1:8080
```

> On PowerShell 5.1, `&&` chaining is unsupported — run commands one at a time.
> See [`RUN.md`](RUN.md) for both Git Bash and PowerShell variants.
