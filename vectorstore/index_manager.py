"""
vectorstore/index_manager.py

Manages multiple FaissStore instances - one per language - so the
project can build, save, load, and query separate indices for English
and Hindi without them interfering with each other.
"""

import os
from typing import Dict, List, Tuple

import numpy as np

from embeddings.embedder_factory import get_embedder
from vectorstore.faiss_store import FaissStore

DEFAULT_INDEX_ROOT = "data/processed/indices"


class IndexManager:
    def __init__(self, index_root: str = DEFAULT_INDEX_ROOT):
        self.index_root = index_root
        self.stores: Dict[str, FaissStore] = {}

    def build_index(self, language: str, chunks: List[Dict]) -> None:
        embedder = get_embedder(language)
        texts = [chunk["text"] for chunk in chunks]

        print(f"[INFO] Embedding {len(texts)} chunk(s) for language '{language}'...")
        embeddings = embedder.embed_texts(texts)

        store = FaissStore(embedding_dim=embedder.embedding_dim)
        store.add(embeddings, chunks)
        self.stores[language] = store

        print(f"[INFO] Built index for '{language}': {store.index.ntotal} vectors.")

    def query(self, language: str, query_text: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if language not in self.stores:
            raise ValueError(
                f"No index built/loaded for language '{language}'. "
                f"Call build_index() or load_index() first."
            )

        embedder = get_embedder(language)
        query_embedding = embedder.embed_query(query_text)
        return self.stores[language].search(query_embedding, top_k=top_k)

    def save_index(self, language: str) -> None:
        if language not in self.stores:
            raise ValueError(f"No index for language '{language}' to save.")
        path = os.path.join(self.index_root, language)
        self.stores[language].save(path)

    def load_index(self, language: str) -> None:
        embedder = get_embedder(language)
        path = os.path.join(self.index_root, language)
        self.stores[language] = FaissStore.load(path, embedding_dim=embedder.embedding_dim)

    def save_all(self) -> None:
        for language in self.stores:
            self.save_index(language)
