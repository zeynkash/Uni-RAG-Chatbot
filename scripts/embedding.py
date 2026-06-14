#!/usr/bin/env python3
"""
Phase 2: Generate Embeddings for chunks_all_universities.json
=============================================================
- Model  : text-embedding-3-small (1536-dim)
- Batches: 100 chunks per API call
- Resume : saves a checkpoint every 10 batches — safe to Ctrl+C and re-run
- Output : embeddings_all_universities.npy  (float32, shape [N, 1536])
           embedding_all_universities_meta.json
"""

import json
import os
import time
import numpy as np
import openai
from tqdm import tqdm
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(__file__)
CHUNKS_FILE    = os.path.join(BASE_DIR, "chunks_all_universities.json")
OUT_NPY        = os.path.join(BASE_DIR, "embeddings_all_universities.npy")
OUT_META       = os.path.join(BASE_DIR, "embedding_all_universities_meta.json")
CHECKPOINT_NPY = os.path.join(BASE_DIR, "embeddings_checkpoint.npy")
CHECKPOINT_IDX = os.path.join(BASE_DIR, "embeddings_checkpoint_idx.txt")

EMBED_MODEL    = "text-embedding-3-small"
BATCH_SIZE     = 100    # chunks per API request
CHECKPOINT_EVERY = 10   # save checkpoint every N batches
MAX_RETRIES    = 5
RETRY_BASE_SEC = 2      # exponential backoff base

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
load_dotenv(os.path.join(BASE_DIR, ".env"))
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found — add it to chunking/.env")

client = openai.OpenAI()

# ─────────────────────────────────────────────
# Load chunks
# ─────────────────────────────────────────────
print("=" * 65)
print("PHASE 2 — EMBEDDING chunks_all_universities.json")
print("=" * 65)

print(f"\nLoading {CHUNKS_FILE} ...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"✓ Loaded {len(chunks):,} chunks")

# Cost preview
total_tokens = sum(c.get("tokens", 0) for c in chunks)
cost_est     = (total_tokens / 1_000_000) * 0.02
print(f"  Total tokens : {total_tokens:,}")
print(f"  Cost estimate: ${cost_est:.3f}")

# ─────────────────────────────────────────────
# Resume from checkpoint if available
# ─────────────────────────────────────────────
start_idx    = 0
all_embeddings: list = []

if os.path.exists(CHECKPOINT_NPY) and os.path.exists(CHECKPOINT_IDX):
    with open(CHECKPOINT_IDX, "r") as f:
        start_idx = int(f.read().strip())
    saved = np.load(CHECKPOINT_NPY)
    all_embeddings = list(saved)
    print(f"\n⚡ Resuming from checkpoint: {start_idx:,} chunks already embedded")
else:
    print("\nNo checkpoint found — starting fresh")

# ─────────────────────────────────────────────
# Embed in batches
# ─────────────────────────────────────────────
remaining_chunks = chunks[start_idx:]
batches = [
    remaining_chunks[i : i + BATCH_SIZE]
    for i in range(0, len(remaining_chunks), BATCH_SIZE)
]

print(f"\nEmbedding {len(remaining_chunks):,} chunks in {len(batches)} batches of {BATCH_SIZE} ...")
print()

errors = 0

for batch_num, batch in enumerate(tqdm(batches, desc="Batches")):
    texts = [c["content"].replace("\n", " ") for c in batch]

    # Retry loop with exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                input=texts,
                model=EMBED_MODEL,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            break  # success

        except openai.RateLimitError:
            wait = RETRY_BASE_SEC ** attempt
            tqdm.write(f"  Rate limit — waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)

        except openai.APIError as e:
            wait = RETRY_BASE_SEC ** attempt
            tqdm.write(f"  API error: {e} — waiting {wait}s")
            time.sleep(wait)

        except Exception as e:
            tqdm.write(f"  Unexpected error on batch {batch_num}: {e}")
            errors += 1
            break  # skip this batch

    # Save checkpoint every N batches
    if (batch_num + 1) % CHECKPOINT_EVERY == 0:
        checkpoint_arr = np.array(all_embeddings, dtype="float32")
        np.save(CHECKPOINT_NPY, checkpoint_arr)
        with open(CHECKPOINT_IDX, "w") as f:
            f.write(str(start_idx + len(all_embeddings)))

# ─────────────────────────────────────────────
# Save final output
# ─────────────────────────────────────────────
print(f"\nSaving final embeddings → {OUT_NPY}")
embeddings_arr = np.array(all_embeddings, dtype="float32")
np.save(OUT_NPY, embeddings_arr)
size_mb = os.path.getsize(OUT_NPY) / 1024 / 1024
print(f"✓ Shape: {embeddings_arr.shape}   Size: {size_mb:.1f} MB")

# Save metadata
meta = {
    "model":              EMBED_MODEL,
    "dimension":          embeddings_arr.shape[1] if len(embeddings_arr.shape) > 1 else 1536,
    "total_embeddings":   embeddings_arr.shape[0],
    "total_chunks":       len(chunks),
    "errors_skipped":     errors,
    "chunks_file":        CHUNKS_FILE,
}
with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=2)
print(f"✓ Metadata → {OUT_META}")

# Clean up checkpoint files
for cp in [CHECKPOINT_NPY, CHECKPOINT_IDX]:
    if os.path.exists(cp):
        os.remove(cp)
print("✓ Checkpoint files cleaned up")

print("\n" + "=" * 65)
print("EMBEDDING COMPLETE ✅")
print(f"  {embeddings_arr.shape[0]:,} embeddings  |  dim={embeddings_arr.shape[1]}  |  {size_mb:.1f} MB")
print("Next step: Phase 3 — build Qdrant index")
print("=" * 65)
