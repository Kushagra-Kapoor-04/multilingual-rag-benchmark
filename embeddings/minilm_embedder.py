"""
embeddings/minilm_embedder.py

Wraps sentence-transformers all-MiniLM-L6-v2 model for embedding
English text. Used as the English baseline throughout the benchmark.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings.base_embedder import BaseEmbedder


class MiniLMEmbedder(BaseEmbedder):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim = 384

    def __init__(self, device: str = None):
        print(f"[INFO] Loading MiniLM model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name, device=device)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim))

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
