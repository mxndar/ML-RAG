# 🤖 ML-RAG — PDF Agentic RAG with ML Query-Intent Router

An enterprise-grade **Agentic Retrieval-Augmented Generation** system optimized for querying large PDF documents. This is NOT a traditional vector-database RAG or a simple chatbot — it's an intelligent agent system with a **trained ML router** that classifies queries by intent, routes them to specialized tools, validates its own evidence, and retries when results are incomplete.

## Architecture

```
Query → Exact Cache → Semantic Cache → ML Router (TF-IDF classifier)
                                           │
                         confident (≥75%)   │   uncertain (<75%)
                       ┌────────────────┤────────────────┐
                       ↓                                   ↓
                   [Direct Tool]              Planner (LLM) → [Tools]
                       │                                   │
                       └────────────────┬────────────────┘
                                       ↓
                            Reranker → Critic → Generator (LLM)
                                         ↑        │
                                         └─ retry ┘
```

### What Makes It Agentic

| Feature | Description |
|---------|-------------|
| **ML Query Router** | Trained TF-IDF + LogReg classifier routes queries by intent (Page/Section/Table/General) — skips the LLM planner when confident |
| **4 Specialized Tools** | BM25 Search, Page Retriever, Section Retriever, Table Extractor |
| **Planner Agent** | LLM decides which tools to use (fallback when router is uncertain) |
| **Self-Validating Critic** | Rule-based validator checks evidence completeness |
| **Retry Loop** | Critic routes back to planner with feedback when evidence is incomplete |
| **Dual Caching** | Exact (SHA256) + Semantic (FAISS) caching for 0 LLM calls on repeat queries |

### LLM Call Budget

| Scenario | LLM Calls |
|----------|:---------:|
| Exact cache hit | **0** |
| Semantic cache hit (fresh) | **0** |
| Semantic cache hit (stale) | **1** |
| Cache miss, **ML router confident**, critic passes | **1** ← 🆕 |
| Cache miss, ML router uncertain, critic passes | **2** |
| Cache miss + 1 retry | **3** |
| Cache miss + 2 retries (max) | **4** |

### Key Design Decisions

- **ML Router** replaces the LLM planner call for high-confidence queries (saves ~500ms + 1 LLM call per query)
- **No vector database** for document retrieval — uses BM25 (keyword) search
- **Embeddings used ONLY** in the semantic cache layer (FAISS + bge-small-en-v1.5)
- **No LLM-based reranking** — uses cross-encoder (ms-marco-MiniLM) for zero token cost
- **Rule-based critic** — no LLM needed for evidence validation
- **Stop words preserved** during text cleaning (important for BM25)

## Quick Start

### Prerequisites

1. **Python 3.12**
2. **Groq API Key** (Get one at [Groq Console](https://console.groq.com/keys))

### Setup

```bash
# Clone and enter the project
cd RAG_MODIFIED

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file and add your API key
cp .env.example .env
# Edit .env and replace your_gemini_api_key_here with your actual key!

# Run the app
python -m app.main
```

### Usage

1. Open `http://localhost:7860` in your browser
2. **Ingest tab**: Upload your PDF → click "Process & Index"
3. **Search tab**: Ask questions about your document
4. **Analytics tab**: View cache statistics

## Project Structure

```
app/
├── main.py                  # Gradio UI
├── config.py                # Settings (.env)
├── schemas.py               # Pydantic models
├── router/                  # 🆕 ML Query-Intent Router
│   ├── query_router.py      # Unified predict() interface
│   ├── baseline_model.py    # TF-IDF + LogReg training
│   ├── distilbert_model.py  # DistilBERT fine-tuning
│   └── saved_models/        # Trained model artifacts
├── ingestion/               # PDF → Chunks → Index
│   ├── loader.py            # PDF text extraction
│   ├── cleaner.py           # Header/footer removal
│   ├── chunker.py           # Semantic chunking + BM25 indexing
│   └── table_parser.py      # Table extraction (pdfplumber)
├── cache/
│   ├── exact_cache.py       # In-memory SHA256 cache (24h TTL)
│   └── semantic_cache.py    # FAISS + bge-small-en-v1.5 (0.95 threshold)
├── tools/                   # 4 retrieval tools
│   ├── bm25_tool.py         # BM25 keyword search
│   ├── page_tool.py         # Fetch by page number
│   ├── section_tool.py      # Fetch by section name (fuzzy match)
│   └── table_tool.py        # Extract parsed tables
├── retrieval/
│   ├── planner.py           # LLM tool router (fallback)
│   ├── reranker.py          # Cross-encoder reranking (top 2)
│   ├── evidence_merger.py   # Merge + truncate to ≤300 tokens
│   └── critic.py            # Rule-based validation
├── generation/
│   └── generator.py         # LLM answer generation
└── orchestration/
    └── graph.py             # LangGraph StateGraph workflow

data/query_intent/           # 🆕 ML Router Dataset
├── raw.csv                  # 1,600 labelled queries (400/class)
├── train.csv / val.csv / test.csv
├── hard_test.csv            # 100 hand-written queries
├── generate_dataset.py      # Reproducible generation script
├── validate_dataset.py      # Quality checks
└── dataset_card.md          # Full documentation
```

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | `your_groq_api_key_here` | Your Groq API key |
| `PLANNER_MODEL` | `llama3-8b-8192` | Planner LLM |
| `GENERATOR_MODEL` | `llama-3.3-70b-versatile` | Generator LLM |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Semantic cache embeddings |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `SEMANTIC_THRESHOLD` | `0.95` | Semantic cache match threshold |
| `MAX_EVIDENCE_TOKENS` | `300` | Max tokens in merged evidence |
| `BM25_TOP_K` | `10` | BM25 candidates before reranking |
| `RERANKER_TOP_K` | `2` | Final chunks after reranking |
| `MAX_CRITIC_RETRIES` | `2` | Max planner retry attempts |
| `ROUTER_ENABLED` | `true` | Enable/disable ML router |
| `ROUTER_MODEL_TYPE` | `tfidf_logreg` | `tfidf_logreg` or `distilbert` |
| `ROUTER_MODEL_PATH` | `app/router/saved_models` | Path to trained models |
| `ROUTER_CONFIDENCE_THRESHOLD` | `0.75` | Min confidence for direct routing |

## Tests

```bash
python -m pytest tests/ -v
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph |
| **ML Router** | **TF-IDF + LogReg (scikit-learn)** |
| LLM (Planner) | Llama 3 8B via Groq |
| LLM (Generator) | Llama 3.3 70B via Groq |
| Document Retrieval | BM25 (rank_bm25) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Semantic Cache | FAISS + BAAI/bge-small-en-v1.5 |
| Exact Cache | In-memory SHA256 dict |
| PDF Processing | pypdf + pdfplumber |
| Chunking | Semantic (sentence-transformers) |
| UI | Gradio |
| Schemas | Pydantic v2 |
