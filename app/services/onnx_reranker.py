"""
app/services/onnx_reranker.py
──────────────────────────────
ONNX INT8 cross-encoder wrapper (FastReranker).
Previously: fast_reranker.py at project root.
"""

import os

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import torch

_BASE               = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
_DEFAULT_MODEL_PATH = os.path.join(_BASE, "reranker_onnx_quantized")
_TOKENIZER_NAME     = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class FastReranker:
    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
        self.model     = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            file_name="model_quantized.onnx",
            provider="CPUExecutionProvider",
        )

    def rerank(self, query: str, chunks: list[str], top_n: int = 5) -> list[dict]:
        if not chunks:
            return []

        truncated = [c[:800] for c in chunks]
        pairs     = [[query, chunk] for chunk in truncated]
        features  = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1)

        scores = scores.numpy()

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            {"score": float(score), "text": chunk}
            for score, chunk in ranked[:top_n]
        ]
