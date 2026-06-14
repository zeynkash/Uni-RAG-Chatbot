"""
app/main.py
────────────
FastAPI application factory and startup event.
Entry point: uvicorn app.main:app
"""

import logging
import sys
import os

# Add the project root to Python's path so "app" is recognized as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.chat_router import router
from app.services.rag_service import load_rag_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── App factory ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Istanbul 50-University RAG Chatbot API",
    description=(
        "AI-powered chatbot covering 50 Istanbul universities — "
        "bilingual retrieval, HyDE query expansion, ONNX cross-encoder reranking, "
        "anti-hallucination prompts, and per-university + per-language metadata filtering."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    try:
        load_rag_system()
    except Exception as e:
        logger.error(f"Failed to load RAG system: {e}")
        raise


# ── Direct run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("ISTANBUL 50-UNIVERSITY RAG CHATBOT — BILINGUAL + HyDE + RERANKER")
    print("=" * 70)
    print("\nServer: http://localhost:8002")
    print("Docs:   http://localhost:8002/docs")
    print("=" * 70 + "\n")

    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, log_level="info", reload=True)
