"""
vectorstore/faiss_store.py

Wraps FAISS for building, saving, loading, and querying a vector index
of embedded document chunks. Kept language-agnostic — one FaissStore
instance handles exactly one index (e.g., one per language), matching
the project's need to compare retrieval quality across languages
separately rather than mixing them into one index.
"""

import json
import os
from typing import Dict, List, Tuple

import faiss
import numpy as np


class FaissStore:
    """A single FAISS index plus the chunk metadata needed to map results back to text."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # IndexFlatIP = exact inner-product search. Since embeddings are
        # L2-normalized (see embedders), inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunk_metadata: List[Dict] = []  # parallel list: index position -> chunk dict

    def add(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """
        Add a batch of embeddings and their corresponding chunk metadata
        (chunk_id, doc_id, text, source — as produced by ingestion/chunker.py).

        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            chunks: list of chunk dicts, same length as embeddings
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings but {len(chunks)} chunk records."
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: index expects {self.embedding_dim}, got {embeddings.shape[1]}."
            )

        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self.chunk_metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search the index for the top_k most similar chunks to a query embedding.

        Args:
            query_embedding: numpy array of shape (embedding_dim,)
            top_k: number of results to return

        Returns:
            List of (chunk_dict, similarity_score) tuples, best match first.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunk_metadata[idx], float(score)))

        return results

    def save(self, directory: str) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved index ({self.index.ntotal} vectors) to '{directory}'")

    @classmethod
    def load(cls, directory: str, embedding_dim: int) -> "FaissStore":
        """Load a previously saved FAISS index and its chunk metadata."""
        store = cls(embedding_dim=embedding_dim)
        store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.json"), "r", encoding="utf-8") as f:
            store.chunk_metadata = json.load(f)
        print(f"[INFO] Loaded index ({store.index.ntotal} vectors) from '{directory}'")
        return store


if __name__ == "__main__":
    # Manual sanity check with random vectors — run: python vectorstore/faiss_store.py
    dim = 8
    store = FaissStore(embedding_dim=dim)

    fake_embeddings = np.random.rand(3, dim).astype("float32")
    fake_embeddings /= np.linalg.norm(fake_embeddings, axis=1, keepdims=True)  # normalize

    fake_chunks = [
        {"chunk_id": "doc1_chunk0", "doc_id": "doc1", "text": "sample text A", "source": "test"},
        {"chunk_id": "doc2_chunk0", "doc_id": "doc2", "text": "sample text B", "source": "test"},
        {"chunk_id": "doc3_chunk0", "doc_id": "doc3", "text": "sample text C", "source": "test"},
    ]

    store.add(fake_embeddings, fake_chunks)
    print(f"Index size: {store.index.ntotal}")

    query = fake_embeddings[0]  # search using the first vector itself
    results = store.search(query, top_k=2)
    print("Top match should be doc1_chunk0 with score ~1.0:")
    for chunk, score in results:
        print(f"  {chunk['chunk_id']}: {score:.4f}")

    store.save("/tmp/faiss_test_index")
    reloaded = FaissStore.load("/tmp/faiss_test_index", embedding_dim=dim)
    assert reloaded.index.ntotal == 3
    print("Save/load round-trip successful.")