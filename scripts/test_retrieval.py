"""
scripts/test_retrieval.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.embedder_factory import get_embedder
from vectorstore.index_manager import IndexManager


def main():
    with open("data/processed/en_chunks.json", encoding="utf-8") as f:
        en_chunks = json.load(f)

    print(f"Loaded {len(en_chunks)} English chunks.")

    manager = IndexManager()
    manager.build_index("en", en_chunks)

    query = "What is artificial intelligence?"
    print(f"\nQuery: {query}\n")

    results = manager.query("en", query, top_k=3)
    for chunk, score in results:
        print(f"{score:.3f} | {chunk['chunk_id']} | {chunk['text'][:100]}...")


if __name__ == "__main__":
    main()
