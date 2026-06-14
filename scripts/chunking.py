#!/usr/bin/env python3
"""
Phase 1: Chunking for all_universities.json
============================================
Improvements over the IZU prototype:
  - Pre-filters docs with < 50 chars of content (nav stubs / junk)
  - Injects metadata prefix into each chunk text so the embedding is context-aware
  - Carries all 9 metadata fields into every child chunk
  - Adds chunk_index + total_chunks for reading-order awareness
  - Sentence-boundary aware splitting with token overlap
"""

import json
import re
import os
import hashlib
from tqdm import tqdm
import tiktoken

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
INPUT_FILE  = os.path.join(os.path.dirname(__file__), "../istanbul_uni_crawler/data/all_universities.json")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "chunks_all_universities.json")
OUTPUT_META = os.path.join(os.path.dirname(__file__), "chunks_all_universities_meta.json")

CHUNK_SIZE    = 800   # max tokens per chunk
CHUNK_OVERLAP = 150   # overlap tokens between consecutive chunks
MIN_CONTENT_CHARS = 50  # skip docs shorter than this

# ─────────────────────────────────────────────
# Tokeniser (same model as embeddings target)
# ─────────────────────────────────────────────
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text)) if text else 0


# ─────────────────────────────────────────────
# Sentence-boundary aware splitter
# ─────────────────────────────────────────────
def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into token-bounded chunks with sentence-level overlap."""
    if not text or not text.strip():
        return []

    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sents: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        # If a single sentence exceeds chunk_size, hard-split by tokens
        if sent_tokens > chunk_size:
            if current_sents:
                chunks.append(" ".join(current_sents))
                current_sents = []
                current_tokens = 0
            tokens = encoding.encode(sent)
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i : i + chunk_size]
                chunks.append(encoding.decode(chunk_tokens))
            continue

        # Would adding this sentence exceed the limit?
        if current_tokens + sent_tokens > chunk_size:
            if current_sents:
                chunks.append(" ".join(current_sents))

            # Build overlap from the tail of current_sents
            overlap_sents: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sents):
                st = count_tokens(s)
                if overlap_tokens + st <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_tokens += st
                else:
                    break
            current_sents  = overlap_sents
            current_tokens = overlap_tokens

        current_sents.append(sent)
        current_tokens += sent_tokens

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


# ─────────────────────────────────────────────
# Metadata prefix injected into the embedded text
# (makes the embedding context-aware)
# ─────────────────────────────────────────────
def build_embed_prefix(doc: dict) -> str:
    parts = []
    if doc.get("university"):
        parts.append(f"[University: {doc['university']}]")
    if doc.get("category"):
        parts.append(f"[Category: {doc['category']}]")
    if doc.get("language"):
        parts.append(f"[Language: {doc['language']}]")
    if doc.get("title"):
        parts.append(f"Title: {doc['title']}")
    return " ".join(parts)


# ─────────────────────────────────────────────
# Chunk a single document
# ─────────────────────────────────────────────
def chunk_document(doc: dict) -> list[dict]:
    content = doc.get("content", "")

    # --- Quality gate ---
    if len(content.strip()) < MIN_CONTENT_CHARS:
        return []

    prefix       = build_embed_prefix(doc)
    full_text    = f"{prefix}\n\n{content}" if prefix else content
    text_chunks  = split_into_chunks(full_text)

    if not text_chunks:
        return []

    base_id = doc.get("doc_id") or hashlib.sha256(doc.get("url","").encode()).hexdigest()[:16]

    chunk_objects = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_obj = {
            # Identity
            "chunk_id":     f"{base_id}_{i}",
            "doc_id":       base_id,
            "chunk_index":  i,
            "total_chunks": len(text_chunks),
            # Text (this is what gets embedded)
            "content":      chunk_text,
            "tokens":       count_tokens(chunk_text),
            # Full metadata — all 9 fields carried through
            "metadata": {
                "title":        doc.get("title", ""),
                "url":          doc.get("url", ""),
                "university":   doc.get("university", ""),
                "language":     doc.get("language", ""),
                "content_type": doc.get("content_type", ""),
                "section":      doc.get("section", ""),
                "category":     doc.get("category", ""),
            },
        }
        chunk_objects.append(chunk_obj)

    return chunk_objects


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("PHASE 1 — CHUNKING all_universities.json")
    print("=" * 65)

    # Load
    print(f"\nLoading {INPUT_FILE} ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data):,} documents")

    # Chunk
    print(f"\nChunking (size={CHUNK_SIZE} tok, overlap={CHUNK_OVERLAP} tok, min_chars={MIN_CONTENT_CHARS}) ...")
    all_chunks: list[dict] = []
    skipped = 0

    for doc in tqdm(data, desc="Progress"):
        if not isinstance(doc, dict):
            skipped += 1
            continue
        chunks = chunk_document(doc)
        if not chunks:
            skipped += 1
            continue
        all_chunks.extend(chunks)

    # Stats
    total_tokens  = sum(c["tokens"] for c in all_chunks)
    avg_tokens    = total_tokens / len(all_chunks) if all_chunks else 0
    cost_estimate = (total_tokens / 1_000_000) * 0.02  # text-embedding-3-small

    print(f"\n{'=' * 65}")
    print(f"Documents loaded  : {len(data):>10,}")
    print(f"Documents skipped : {skipped:>10,}  (too short / malformed)")
    print(f"Documents chunked : {len(data)-skipped:>10,}")
    print(f"Total chunks      : {len(all_chunks):>10,}")
    print(f"Avg chunks/doc    : {len(all_chunks)/(len(data)-skipped):.1f}")
    print(f"Avg tokens/chunk  : {avg_tokens:.0f}")
    print(f"Total tokens      : {total_tokens:>10,}")
    print(f"Embedding cost est: ${cost_estimate:.2f}  (text-embedding-3-small @ $0.02/1M)")

    # Language distribution
    lang_counts: dict[str, int] = {}
    cat_counts:  dict[str, int] = {}
    uni_counts:  dict[str, int] = {}
    for c in all_chunks:
        m = c["metadata"]
        lang_counts[m["language"]]   = lang_counts.get(m["language"], 0) + 1
        cat_counts[m["category"]]    = cat_counts.get(m["category"], 0) + 1
        uni_counts[m["university"]]  = uni_counts.get(m["university"], 0) + 1

    print("\nLanguage distribution:")
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang:<6} {cnt:>7,} chunks")

    print("\nTop 5 universities by chunk count:")
    for uni, cnt in sorted(uni_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {uni:<45} {cnt:>5,}")

    # Save
    print(f"\nSaving chunks → {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    size_mb = os.path.getsize(OUTPUT_JSON) / 1024 / 1024
    print(f"✓ Saved {len(all_chunks):,} chunks  ({size_mb:.1f} MB)")

    # Save metadata summary
    meta = {
        "total_documents":  len(data),
        "documents_skipped": skipped,
        "total_chunks":     len(all_chunks),
        "avg_chunks_per_doc": round(len(all_chunks) / max(len(data)-skipped, 1), 2),
        "chunk_size_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "avg_tokens_per_chunk": round(avg_tokens, 1),
        "total_tokens": total_tokens,
        "embedding_cost_estimate_usd": round(cost_estimate, 3),
        "languages": lang_counts,
        "top_categories": dict(sorted(cat_counts.items(), key=lambda x: -x[1])[:10]),
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved metadata → {OUTPUT_META}")

    print("\n" + "=" * 65)
    print("CHUNKING COMPLETE ✅")
    print("Next step: Phase 2 — generate embeddings for the chunks")
    print("=" * 65)


if __name__ == "__main__":
    main()
