#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Start the 50-University RAG Chatbot
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "[ERROR] OPENAI_API_KEY not set."
    echo "        Create .env in the project root with: OPENAI_API_KEY=sk-..."
    exit 1
fi

echo ""
echo "======================================================================"
echo "  ISTANBUL 50-UNIVERSITY RAG CHATBOT"
echo "  Bilingual + HyDE + ONNX Reranker + Anti-Hallucination Prompts"
echo "======================================================================"
echo ""
echo "  Server  : http://localhost:8002"
echo "  API Docs: http://localhost:8002/docs"
echo "  Frontend: open frontend/index.html in your browser"
echo ""
echo "  Required data files in data/:"
echo "    - chunks.json"
echo "    - faiss_index.bin"
echo "    - embeddings.npy"
echo "    - university_registry.json"
echo ""
echo "  If data/ is missing, run scripts in order:"
echo "    python3 scripts/01_merge_raw_data.py"
echo "    python3 scripts/02_chunk_data.py"
echo "    python3 scripts/03_embed_data.py"
echo ""
echo "======================================================================"
echo ""

uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8002 \
    --log-level info \
    --reload
