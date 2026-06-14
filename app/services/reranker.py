"""
app/services/reranker.py
─────────────────────────
Production-ready reranker with caching and async support.
Previously: full_reranker.py at project root.
"""

import hashlib
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.services.onnx_reranker import FastReranker

logger = logging.getLogger(__name__)


class ProductionReranker:
    def __init__(self, model_path: str = "./models/reranker_onnx_quantized"):
        self.reranker  = FastReranker(model_path)
        self._cache    = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _cache_key(self, query: str, chunks: list[str]) -> str:
        content = query + "".join(chunks[:10])
        return hashlib.md5(content.encode()).hexdigest()

    def rerank(self, query: str, chunks: list[str], top_n: int = 5) -> list[dict]:
        key = self._cache_key(query, chunks)
        if key in self._cache:
            logger.info("[cache hit] reranker")
            return self._cache[key]

        # Pre-filter by keyword overlap to reduce candidates
        query_words   = set(query.lower().split())
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap     = len(query_words & chunk_words)
            scored_chunks.append((overlap, chunk))

        pre_filtered = [
            chunk for _, chunk in
            sorted(scored_chunks, key=lambda x: x[0], reverse=True)[:15]
        ]

        result           = self.reranker.rerank(query, pre_filtered, top_n=top_n)
        self._cache[key] = result
        return result

    async def rerank_async(self, query: str, chunks: list[str], top_n: int = 5) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.rerank(query, chunks, top_n),
        )
