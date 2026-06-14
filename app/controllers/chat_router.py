"""
app/controllers/chat_router.py
───────────────────────────────
HTTP route handlers for the RAG Chatbot API.
All business logic is delegated to app.services.rag_service.
"""

import time
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services import rag_service
from app.services.rag_service import (
    TOP_N,
    CANDIDATE_K,
    detect_language,
    retrieve_chunks,
    generate_answer,
    is_fallback_answer,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Root ───────────────────────────────────────────────────────────────────────
@router.get("/", response_model=dict)
async def root():
    return {
        "message": "Istanbul 50-University RAG Chatbot API",
        "version": "1.0.0",
        "status":  "running",
        "endpoints": {
            "chat":         "/chat",
            "universities": "/universities",
            "health":       "/health",
            "stats":        "/stats",
            "docs":         "/docs",
        },
    }


# ── Universities ───────────────────────────────────────────────────────────────
@router.get("/universities", response_model=dict)
async def list_universities():
    """Return the university registry for the UI dropdown."""
    return {"universities": rag_service.university_registry}


# ── Health ─────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if (
            rag_service.chunks and rag_service.faiss_index and rag_service.reranker
        ) else "not ready",
        chunks_loaded=len(rag_service.chunks) if rag_service.chunks else 0,
        index_size=rag_service.faiss_index.ntotal if rag_service.faiss_index else 0,
        universities=len(rag_service.university_registry) - 1,  # subtract "all"
        reranker_loaded=rag_service.reranker is not None,
        timestamp=datetime.now().isoformat(),
    )


# ── Chat ───────────────────────────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    - Sequential Search: Always queries Turkish pages first. If no answer is found,
      queries English pages.
    - Bilingual: auto-detects TR/EN, searches in both.
    - HyDE: generates hypothetical answer for better retrieval.
    - Reranker: ONNX INT8 cross-encoder for precision.
    - Anti-hallucination: source-only strict prompts in both languages.
    """
    start = time.time()

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(request.message) > 500:
        raise HTTPException(status_code=400, detail="Message too long (max 500 characters)")

    uni_filter = request.university_filter
    if uni_filter == "all":
        uni_filter = None

    # Resolve target output language
    if request.language and request.language in ("tr", "en"):
        lang = request.language
    else:
        lang = detect_language(request.message)

    logger.info(
        f"Query: '{request.message[:80]}' | "
        f"filter={uni_filter or 'ALL'} | target_lang={lang}"
    )

    # Phase 1: Search Turkish pages first
    logger.info("Sequential Search — Phase 1: Retrieving from Turkish pages")
    retrieved = retrieve_chunks(
        request.message,
        top_n=TOP_N,
        university_filter=uni_filter,
        language_filter="tr",
    )

    answer = generate_answer(
        request.message,
        retrieved,
        language=lang,
        university_filter=uni_filter,
    )

    # Phase 2 Fallback: If Turkish search had no chunks or returned fallback answer
    if not retrieved or is_fallback_answer(answer, lang):
        logger.info(
            "Sequential Search — Phase 1 yielded no results/fallback. "
            "Phase 2: Retrieving from English pages..."
        )
        retrieved_en = retrieve_chunks(
            request.message,
            top_n=TOP_N,
            university_filter=uni_filter,
            language_filter="en",
        )
        if retrieved_en:
            answer_en = generate_answer(
                request.message,
                retrieved_en,
                language=lang,
                university_filter=uni_filter,
            )
            if not is_fallback_answer(answer_en, lang):
                retrieved = retrieved_en
                answer    = answer_en
                logger.info("Sequential Search — Phase 2 found results in English pages.")
            else:
                logger.info(
                    "Sequential Search — Phase 2 also returned fallback answer. "
                    "Keeping original fallback."
                )
        else:
            logger.info(
                "Sequential Search — Phase 2 yielded no English candidates. "
                "Keeping original fallback."
            )

    response_time = (time.time() - start) * 1000

    sources = [
        {
            "title":           chunk["metadata"].get("title", "Unknown"),
            "url":             chunk["metadata"].get("url", ""),
            "university":      chunk["metadata"].get("university", ""),
            "university_name": chunk["metadata"].get("university_name", ""),
            "language":        chunk["metadata"].get("language", ""),
            "category":        chunk["metadata"].get("category", ""),
            "score":           round(chunk["score"], 4),
            "rerank_score":    round(chunk["rerank_score"], 4),
            "snippet":         chunk["content"][:200] + "…",
        }
        for chunk in retrieved
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        response_time_ms=response_time,
        conversation_id=request.conversation_id or f"conv_{int(time.time())}",
        university_filter=request.university_filter,
        detected_language=lang,
    )


# ── Stats ──────────────────────────────────────────────────────────────────────
@router.get("/stats", response_model=dict)
async def get_stats():
    return {
        "total_chunks":               len(rag_service.chunks) if rag_service.chunks else 0,
        "universities":               len(rag_service.university_registry) - 1,
        "index_type":                 "FAISS IndexFlatIP (cosine)",
        "embedding_model":            "text-embedding-3-small",
        "rerank_model":               "ONNX mmarco-mMiniLMv2-L12-H384-v1 (INT8)",
        "llm_model":                  "gpt-4o-mini",
        "embedding_dimension":        1536,
        "candidate_k":                CANDIDATE_K,
        "top_n":                      TOP_N,
        "bilingual_search":           True,
        "hyde_enabled":               True,
        "reranker_enabled":           True,
        "university_filter":          True,
        "language_filter":            True,
        "anti_hallucination_prompts": True,
        "chunking_strategy":          "priority-adaptive (400/600/800 tokens)",
    }
