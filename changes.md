# Changelog

All structural changes to this project are documented here in order.

---

## [2026-06-14] — Full Project Restructure (MVC Layout)

### Added
- `changes.md` — this file; documents all changes going forward
- `app/` — new application package (MVC split of the old flat structure)
  - `app/__init__.py`
  - `app/models/__init__.py`
  - `app/models/schemas.py` — Pydantic request/response schemas
  - `app/controllers/__init__.py`
  - `app/controllers/chat_router.py` — all HTTP route handlers
  - `app/services/__init__.py`
  - `app/services/rag_service.py` — RAG business logic (load, retrieve, generate)
  - `app/services/reranker.py` — ProductionReranker with cache + async support
  - `app/services/onnx_reranker.py` — FastReranker ONNX INT8 wrapper
  - `app/main.py` — FastAPI app factory and startup event
- `frontend/index.html` — chatbot UI (moved from root)
- `models/reranker_onnx_quantized/` — ONNX model weights (moved from root)
- `start.sh` — startup script (renamed + updated)
- `.gitignore` — proper gitignore (was missing entirely)

### Renamed / Moved
| Old Path | New Path |
|----------|----------|
| `chatbot_api_50unis.py` | Split into `app/main.py`, `app/controllers/chat_router.py`, `app/services/rag_service.py`, `app/models/schemas.py` |
| `fast_reranker.py` | `app/services/onnx_reranker.py` |
| `full_reranker.py` | `app/services/reranker.py` |
| `chatbot_ui_50unis.html` | `frontend/index.html` |
| `start_chatbot.sh` | `start.sh` |
| `scripts/02_chunk_50unis.py` | `scripts/02_chunk_data.py` |
| `scripts/03_embed_50unis.py` | `scripts/03_embed_data.py` |
| `reranker_onnx_quantized/` | `models/reranker_onnx_quantized/` |
| `data/chunks_50unis.json` | `data/chunks.json` |
| `data/embeddings_50unis.npy` | `data/embeddings.npy` |
| `data/faiss_index_50unis.bin` | `data/faiss_index.bin` |

### Deleted
| File | Reason |
|------|--------|
| `chatbot_api_50unis.py.bak` | Leftover backup, no longer needed |
| `chatbot_api_50unis.py.bak2` | Leftover backup, no longer needed |
| `chatbot_api_50unis.py` | Replaced by MVC split under `app/` |
| `fast_reranker.py` | Moved to `app/services/onnx_reranker.py` |
| `full_reranker.py` | Moved to `app/services/reranker.py` |
| `__pycache__/` | Python bytecode cache, should never be committed |
| `.vscode/` | Editor-specific IDE artifact |
| `scripts/rescue_embeddings.py` | One-off recovery script, no longer needed |

### Code Reference Updates
- All file paths updated: `chunks_50unis.json` → `chunks.json`, `faiss_index_50unis.bin` → `faiss_index.bin`, `embeddings_50unis.npy` → `embeddings.npy`
- Reranker model path updated: `reranker_onnx_quantized/` → `models/reranker_onnx_quantized/`
- Import: `from full_reranker import ProductionReranker` → `from app.services.reranker import ProductionReranker`
- Import: `from fast_reranker import FastReranker` → `from app.services.onnx_reranker import FastReranker`
- Uvicorn module string: `chatbot_api_50unis:app` → `app.main:app`
- Scripts updated to reference renamed data files
