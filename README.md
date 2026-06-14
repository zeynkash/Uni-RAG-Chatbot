# 🎓 Istanbul 50-University RAG Chatbot

An AI-powered bilingual chatbot covering **50 Istanbul universities**, built with FastAPI, FAISS, and OpenAI. Supports Turkish and English queries with HyDE query expansion, ONNX cross-encoder reranking, and anti-hallucination prompts.

---

## ✨ Features

- 🌍 **Bilingual** — Handles Turkish & English queries automatically
- 🔍 **HyDE** — Hypothetical Document Embeddings for better semantic retrieval
- ⚡ **FAISS** — Fast cosine-similarity vector search over 50-university knowledge base
- 🔄 **Query Translation** — Auto-translates queries to retrieve from both languages
- 🏫 **Per-University Filtering** — Scope queries to a single university
- 🛡️ **Anti-Hallucination Prompts** — Strictly grounded answers from source documents
- 📄 **API Docs** — Interactive Swagger UI at `/docs`

---

## 📁 Project Structure

```
50_uni_rag/
├── app/
│   ├── main.py                  # FastAPI app factory & startup
│   ├── controllers/
│   │   └── chat_router.py       # API routes (/chat, /health)
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   └── services/
│       ├── rag_service.py       # Core RAG logic (retrieval + generation)
│       ├── reranker.py          # Async reranker wrapper
│       └── onnx_reranker.py     # ONNX INT8 cross-encoder reranker
├── scripts/
│   ├── chunking.py              # Text chunking pipeline
│   ├── embedding.py             # OpenAI embedding + checkpoint saving
│   └── build_faiss.py           # FAISS index builder
├── data/                        # (git-ignored) Generated data files
│   ├── chunks.json
│   ├── faiss_index.bin
│   ├── embeddings.npy
│   └── university_registry.json
├── models/                      # (git-ignored) ONNX reranker weights
│   └── reranker_onnx_quantized/
├── frontend/
│   └── index.html               # Browser-based chat UI
├── .env                         # API keys (never commit this)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & create virtual environment

```bash
git clone <repo-url>
cd 50_uni_rag
python3 -m venv ~/rag-venv
source ~/rag-venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

---

## 🗄️ Data Preparation

Run the scripts **in order** to build the data pipeline from raw scraped data:

```bash
# Step 1 — Merge raw university data into a unified JSON
python3 scripts/01_merge_raw_data.py

# Step 2 — Chunk documents into retrieval-ready pieces
python3 scripts/chunking.py

# Step 3 — Generate OpenAI embeddings (with checkpointing)
python3 scripts/embedding.py

# Step 4 — Build the FAISS index
python3 scripts/build_faiss.py
```

After these steps, the `data/` directory will contain:
- `chunks.json` — chunked university documents
- `embeddings.npy` — embedding vectors
- `faiss_index.bin` — FAISS index
- `university_registry.json` — university name/ID mapping

---

## 🚀 Running the Server

```bash
source ~/rag-venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8002 --log-level info --reload
```

| Endpoint | URL |
|---|---|
| API Server | http://localhost:8002 |
| Swagger Docs | http://localhost:8002/docs |
| Frontend UI | Open `frontend/index.html` in your browser |

---

## 🔌 API Reference

### `POST /chat`

Send a question and receive a grounded answer.

**Request body:**
```json
{
  "query": "Hangi fakülteler var?",
  "university_filter": "istanbul_universitesi",
  "language_filter": null
}
```

**Response:**
```json
{
  "answer": "İstanbul Üniversitesi bünyesinde şu fakülteler bulunmaktadır: ...",
  "sources": [...],
  "language": "tr",
  "latency_ms": 1240
}
```

### `GET /health`

Returns server and RAG system status.

---

## 🧠 RAG Pipeline

```
User Query
    │
    ▼
Language Detection (TR / EN)
    │
    ▼
Query Translation (TR ↔ EN via GPT-4o-mini)
    │
    ▼
HyDE — Generate hypothetical answer passage
    │
    ▼
Embed all 3 queries (original + translated + HyDE)
    │
    ▼
FAISS Search (top-32 candidates each, merged by best score)
    │
    ▼
University Filter  →  Language Filter
    │
    ▼
FAISS Score Ranking (top-8 returned)
    │
    ▼
GPT-4o-mini Answer Generation (anti-hallucination system prompt)
    │
    ▼
Response
```

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt). Key dependencies:

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `openai` | Embeddings & chat completions |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Embedding model support |
| `optimum[onnxruntime]` | ONNX cross-encoder reranker |
| `torch` | Tensor ops for reranker |
| `tiktoken` | Token counting for chunking |
| `python-dotenv` | `.env` loading |

---

## 📝 Notes

- The `data/` and `models/` directories are **git-ignored** — you must generate them locally.
- The ONNX reranker requires model weights in `models/reranker_onnx_quantized/`. If absent, the system falls back to FAISS score ranking.
- Embeddings use OpenAI's `text-embedding-3-small` model. Costs apply per API call.
- Answer generation uses `gpt-4o-mini` for cost-efficiency.
