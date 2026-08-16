"""
embeddings/base_embedder.py

Defines a common interface all embedding model wrappers must follow,
so the rest of the pipeline (vectorstore, evaluation) never needs to
know which specific model is being used underneath.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedding model wrappers."""

    model_name: str = "base"
    embedding_dim: int = 0

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
