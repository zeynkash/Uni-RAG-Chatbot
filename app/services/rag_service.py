"""
app/services/rag_service.py
────────────────────────────
Core RAG business logic:
  - System startup / data loading
  - Language detection & query translation
  - HyDE (Hypothetical Document Embeddings)
  - FAISS retrieval with university + language filters
  - Answer generation with anti-hallucination prompts
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional

import faiss
import numpy as np
import openai
from dotenv import load_dotenv

from app.services.reranker import ProductionReranker

logger = logging.getLogger(__name__)

# ── Environment ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))

load_dotenv(os.path.join(_root, ".env"))
load_dotenv(os.path.join(_root, "..", ".env"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── File paths ─────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(_root, "data")
CHUNKS_PATH     = os.path.join(DATA_DIR, "chunks.json")
INDEX_PATH      = os.path.join(DATA_DIR, "faiss_index.bin")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
REGISTRY_PATH   = os.path.join(DATA_DIR, "university_registry.json")

# ── Reranker config ────────────────────────────────────────────────────────────
_RERANK_MODEL_PATH     = os.path.join(_root, "models", "reranker_onnx_quantized")
TOP_N                  = 8
CANDIDATE_K            = 32
RERANK_SCORE_THRESHOLD = -10.0

# ── Global state ───────────────────────────────────────────────────────────────
chunks: list             = []
faiss_index              = None
embeddings               = None
reranker: ProductionReranker | None = None
university_registry: dict = {}


# ── Startup ────────────────────────────────────────────────────────────────────
def load_rag_system() -> None:
    global chunks, faiss_index, embeddings, reranker, university_registry

    logger.info("Loading RAG system ...")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"✓ Loaded {len(chunks):,} chunks")

    faiss_index = faiss.read_index(INDEX_PATH)
    logger.info(f"✓ FAISS index: {faiss_index.ntotal:,} vectors")

    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            university_registry = json.load(f)
        logger.info(f"✓ University registry: {len(university_registry)} entries")

    logger.info(f"Loading ONNX reranker from: {_RERANK_MODEL_PATH}")
    reranker = ProductionReranker(model_path=_RERANK_MODEL_PATH)
    logger.info("✓ Reranker ready (ONNX + INT8)")
    logger.info("✓ RAG system ready!")


# ── Language detection ─────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """
    Detect language from query text.
    Turkish special chars are a hard signal.
    Otherwise use keyword scoring.
    """
    if any(c in text for c in "çğıöşüÇĞİÖŞÜ"):
        return "tr"
    tr_words = {
        "nedir", "nasıl", "ne", "hangi", "kaç", "kim", "nerede",
        "var", "mı", "mi", "mu", "mü", "için", "ile", "üniversite",
        "bölüm", "fakülte", "lisans", "yüksek",
    }
    en_words = {
        "what", "how", "where", "when", "which", "who", "why",
        "faculty", "department", "the", "is", "are", "can", "does",
        "do", "university", "bachelor", "master", "program",
    }
    lower = text.lower()
    tr_h  = sum(1 for w in tr_words if w in lower)
    en_h  = sum(1 for w in en_words if w in lower)
    return "en" if en_h > tr_h else "tr"


# ── Query translation ──────────────────────────────────────────────────────────
def translate_query(query: str, source_lang: str) -> str:
    target = "en" if source_lang == "tr" else "tr"
    names  = {"tr": "Turkish", "en": "English"}
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"Translate the following {names[source_lang]} query to {names[target]}. "
                f"Return ONLY the translated query.\n\nQuery: {query}"}],
            temperature=0.1, max_tokens=150, seed=42,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return query


# ── HyDE ──────────────────────────────────────────────────────────────────────
_hyde_cache: dict = {}


def generate_hypothetical_document(query: str, src_lang: str, uni_name: str = "") -> str:
    lang_inst = "Cevabı Türkçe yaz." if src_lang == "tr" else "Write the answer in English."
    uni_ctx   = f" for {uni_name}" if uni_name else " for an Istanbul university"
    cache_key = (query.lower().strip(), src_lang, uni_name)
    if cache_key in _hyde_cache:
        return _hyde_cache[cache_key]
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"You are an official university website{uni_ctx}. "
                f"Write a short (2-4 sentence) factual passage that directly answers "
                f"the following question. Do NOT say this is hypothetical. {lang_inst}\n\n"
                f"Question: {query}"}],
            temperature=0.4, max_tokens=200, seed=42,
        )
        hyp = resp.choices[0].message.content.strip()
        _hyde_cache[cache_key] = hyp
        return hyp
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return query


# ── Embedding ──────────────────────────────────────────────────────────────────
_embed_cache: dict = {}


def get_embedding(text: str) -> list[float]:
    key = text.strip()[:500]
    if key in _embed_cache:
        return _embed_cache[key]
    resp = openai.embeddings.create(
        input=[text.replace("\n", " ")],
        model="text-embedding-3-small",
    )
    vec = resp.data[0].embedding
    _embed_cache[key] = vec
    return vec


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve_chunks(
    query: str,
    top_n: int = TOP_N,
    university_filter: Optional[str] = None,
    language_filter: Optional[str] = None,
    use_hyde: bool = True,
) -> list[dict]:
    """
    Full retrieval pipeline:
    1. Language detection + query translation
    2. HyDE hypothetical document generation
    3. Embed all three queries + FAISS search
    4. Merge results (highest score wins)
    5. University metadata filter (Filter 1)
    6. Language soft-filter (Filter 2) — prefer matching language
    7. ONNX cross-encoder reranking
    8. Return top_n results above threshold
    """
    # ── 1. Language + translation ──────────────────────────────────────────────
    src_lang   = detect_language(query)
    translated = translate_query(query, source_lang=src_lang)

    # ── 2. HyDE ───────────────────────────────────────────────────────────────
    uni_name = ""
    if university_filter and university_filter != "all":
        uni_name = university_registry.get(university_filter, university_filter)
    # Always generate HyDE in Turkish — university data lives mostly in Turkish pages,
    # so a Turkish hypothetical doc embeds closer to those chunks for better recall.
    hyde_lang = "tr"
    hyp_doc = generate_hypothetical_document(query, hyde_lang, uni_name) if use_hyde else query

    # ── 3. Embed + search ─────────────────────────────────────────────────────
    def embed_and_search(q: str) -> dict:
        vec = np.array([get_embedding(q)], dtype="float32")
        faiss.normalize_L2(vec)
        scores, indices = faiss_index.search(vec, CANDIDATE_K)
        return {
            int(idx): float(score)
            for idx, score in zip(indices[0], scores[0])
            if 0 <= int(idx) < len(chunks)
        }

    hits_orig  = embed_and_search(query)
    hits_trans = embed_and_search(translated)
    hits_hyde  = embed_and_search(hyp_doc)

    # ── 4. Merge — highest FAISS score wins ───────────────────────────────────
    merged: dict = {}
    for hits in (hits_trans, hits_hyde, hits_orig):
        for idx, score in hits.items():
            if idx not in merged or score > merged[idx]:
                merged[idx] = score

    logger.info(
        f"FAISS — orig:{len(hits_orig)} trans:{len(hits_trans)} "
        f"hyde:{len(hits_hyde)} → merged:{len(merged)}"
    )

    # ── 5a. University metadata filter (Filter 1) ──────────────────────────────
    if university_filter and university_filter != "all":
        before = len(merged)
        merged = {
            idx: score for idx, score in merged.items()
            if chunks[idx]["metadata"].get("university") == university_filter
        }
        logger.info(
            f"University filter '{university_filter}': "
            f"{before} → {len(merged)} candidates"
        )

    # ── 5b. Language filter (Filter 2) ────────────────────────────────────────
    # Filter strictly by the requested language if "tr" or "en" is provided.
    # Enables sequential Phase 1 (Turkish only) and Phase 2 (English fallback).
    if language_filter in ("tr", "en"):
        merged = {
            idx: score for idx, score in merged.items()
            if chunks[idx]["metadata"].get("language") == language_filter
        }
        logger.info(
            f"Language filter '{language_filter}' (HARD): {len(merged)} candidates"
        )
    else:
        logger.info(
            f"Language filter skipped for '{language_filter}' — retrieving from all languages"
        )

    if not merged:
        logger.warning("No candidates after filters — returning empty")
        return []

    # ── 6b. FAISS-score fallback (reranker disabled) ───────────────────────────
    # Returns top_n chunks sorted purely by FAISS cosine similarity score.
    # To re-enable the ONNX reranker, replace this block with the cross-encoder
    # rerank call using: reranker.rerank(query, doc_texts, top_n=top_n)
    sorted_candidates = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results = [
        {
            "content":      chunks[idx]["content"],
            "metadata":     chunks[idx]["metadata"],
            "score":        score,
            "rerank_score": 0.0,   # placeholder — reranker is disabled
        }
        for idx, score in sorted_candidates
    ]
    logger.info(
        f"Retrieve [{src_lang}] — {len(merged)} candidates → "
        f"{len(results)} returned by FAISS score (reranker DISABLED)"
    )
    return results


# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__",   r"\1", text)
    text = re.sub(r"\*(.+?)\*",   r"\1", text)
    text = re.sub(r"_(.+?)_",     r"\1", text)
    text = re.sub(r"^[\-\*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"```.*?```",   "", text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`",     r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_list_question(query: str) -> bool:
    """Detect questions that ask for an enumeration of items."""
    q = query.lower()

    tr_keywords = [
        "hangi", "neler", "listele", "hangileri",
        "tüm", "bütün", "hepsi", "sayın", "kaç tane",
    ]
    if any(kw in q for kw in tr_keywords):
        return True

    en_keywords = [
        "list", "enumerate", "name all", "give me all",
        "all of", "how many",
    ]
    if any(kw in q for kw in en_keywords):
        return True

    list_patterns = [
        r"\bwhat\b.{0,40}\bare\b",
        r"\bwhat\b.{0,40}\bdoes\b",
        r"\bwhat\b.{0,40}\bdo\b",
        r"\bwhich\b.{0,40}\bare\b",
        r"\bwhich\b.{0,40}\bdoes\b",
        r"\btell me (all|about all)\b",
        r"\bshow me (all|the)\b",
    ]
    return any(re.search(pat, q) for pat in list_patterns)


def extract_list_items(text: str) -> list[str]:
    """Extract bullet/numbered items from chunk text."""
    bullet = re.compile(r"^\s*(?:\d+[.):*]\s+|[-•*–] ?\s+)(.*)")
    items  = []
    for line in text.split("\n"):
        m = bullet.match(line)
        if m:
            item = m.group(1).strip()
            if item:
                items.append(item)
    return items


def is_fallback_answer(answer: str, language: str) -> bool:
    """Return True if the answer is a standard 'not found' fallback."""
    clean_ans = answer.strip().strip(".").strip().lower()
    if language == "tr":
        return clean_ans in (
            "bu bilgiye ulaşılamadı",
            "bilgiye ulaşılamadı",
            "bu bilgi mevcut değil",
        )
    return clean_ans in (
        "this information is not available",
        "information not available",
        "the information is not available",
    )


# ── Answer generation ──────────────────────────────────────────────────────────
def generate_answer(
    query: str,
    retrieved: list[dict],
    language: str = "tr",
    university_filter: Optional[str] = None,
) -> str:
    """Generate a grounded answer from the retrieved chunks."""
    fallback = {
        "tr": "Bu bilgiye ulaşılamadı.",
        "en": "This information is not available.",
    }.get(language, "This information is not available.")

    if not retrieved:
        return fallback

    sources_used = []
    for i, chunk in enumerate(retrieved[:6], 1):
        meta    = chunk["metadata"]
        uni     = meta.get("university_name", meta.get("university", "Unknown University"))
        title   = meta.get("title", "")
        url     = meta.get("url", "")
        content = chunk["content"]

        header = f"[Source {i}] University: {uni}"
        if title:
            header += f" | Page: {title}"
        if url:
            header += f" | URL: {url}"
        sources_used.append(f"{header}\n{content}")

    context = "\n\n---\n\n".join(sources_used)

    list_hint = ""
    if is_list_question(query):
        items = []
        for chunk in retrieved[:6]:
            items.extend(extract_list_items(chunk["content"]))
        if items:
            list_hint = "\n\nNOTE: The sources contain a list. Include ALL relevant items."

    if language == "tr":
        system_prompt = (
            "Sen, İstanbul üniversitelerine ait resmi web sitelerinden alınan kaynaklara "
            "dayalı olarak soruları yanıtlayan bir yapay zeka asistansın.\n\n"
            "Kurallar:\n"
            "1. YALNIZCA aşağıdaki kaynaklardaki bilgileri kullan. Bilgi eksikse, "
            "'Bu bilgiye ulaşılamadı.' yaz — başka bir şey yazma.\n"
            "2. Hiçbir zaman tahmin etme veya genel bilgi kullanma.\n"
            "3. Kaynak metinde lisans programları, fakülteler veya bölümler listeleniyorsa, "
            "bu listeyi eksiksiz olarak sun.\n"
            "4. Liste soruları için (örn. 'fakülteler neler'), varsa tüm öğeleri dahil et.\n"
            "5. Her zaman Türkçe yanıt ver.\n"
            "6. Markdown biçimlendirmesi kullanma. Listeler için sayılar kullan.\n"
            "7. Giriş veya kapanış cümlesi ekleme — yalnızca soruyu yanıtla.\n"
            "8. Farklı üniversitelere ait bilgileri birbirine karıştırma."
        )
        user_prompt = f"Kaynaklara dayanarak yanıtla:\n\n{context}\n\nSORI: {query}{list_hint}"
    else:
        system_prompt = (
            "You are an AI assistant that answers questions based solely on official "
            "university website sources provided below.\n\n"
            "Rules:\n"
            "1. Use ONLY information from the sources below. If the information is missing, "
            "write 'This information is not available.' — nothing else.\n"
            "2. Never guess or use general knowledge.\n"
            "3. If the source lists faculties, departments, or programs, present the full list.\n"
            "4. For list questions (e.g. 'what are the faculties'), if the source contains a "
            "list that answers the question, include ALL items from that list. Ignore irrelevant "
            "lists (like website menus or footers).\n"
            "5. If the source text is in Turkish, translate the relevant parts accurately.\n"
            "6. ALWAYS respond in English.\n"
            "7. NEVER use markdown formatting. Write in plain text, use numbers for lists.\n"
            "8. Do not add introductory or closing sentences — just answer the question directly.\n"
            "9. Do not mix information between different universities — attribute each fact to "
            "its correct source."
        )
        user_prompt = f"Answer based on sources:\n\n{context}\n\nQUESTION: {query}{list_hint}"

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.05,
        max_tokens=600,
        top_p=0.9,
        seed=42,
    )
    return clean_markdown(resp.choices[0].message.content)
