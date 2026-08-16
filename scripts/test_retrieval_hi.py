"""
scripts/test_retrieval_hi.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.embedder_factory import get_embedder
from vectorstore.index_manager import IndexManager


def main():
    with open("data/processed/hi_chunks.json", encoding="utf-8") as f:
        hi_chunks = json.load(f)

    print(f"Loaded {len(hi_chunks)} Hindi chunks.")

    manager = IndexManager()
    manager.build_index("hi", hi_chunks)

    query = "\u0915\u0943\u0924\u094d\u0930\u093f\u092e \u092c\u0941\u0926\u094d\u0927\u093f\u092e\u0924\u094d\u0924\u093e \u0915\u094d\u092f\u093e \u0939\u0948?"
    print(f"\nQuery: {query}\n")

    results = manager.query("hi", query, top_k=3)
    for chunk, score in results:
        print(f"{score:.3f} | {chunk['chunk_id']} | {chunk['text'][:100]}...")


if __name__ == "__main__":
    main()
