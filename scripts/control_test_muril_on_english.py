"""
scripts/control_test_muril_on_english.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.muril_embedder import MurilEmbedder
from vectorstore.faiss_store import FaissStore

ENGLISH_QUERIES = [
    "What is artificial intelligence?",
    "How does photosynthesis work?",
    "What caused World War II?",
    "What is democracy?",
    "Tell me about the Taj Mahal.",
]

TOP_K = 5


def main():
    with open("data/processed/en_chunks.json", encoding="utf-8") as f:
        en_chunks = json.load(f)

    print("[INFO] Loading MuRIL and embedding the ENGLISH corpus (control test)...")
    embedder = MurilEmbedder()

    texts = [chunk["text"] for chunk in en_chunks]
    embeddings = embedder.embed_texts(texts)

    store = FaissStore(embedding_dim=embedder.embedding_dim)
    store.add(embeddings, en_chunks)
    print(f"[INFO] Built MuRIL-on-English index: {store.index.ntotal} vectors.\n")

    gaps = []
    print(f"{'='*70}\nMuRIL ON ENGLISH DATA (control)\n{'='*70}")

    for query in ENGLISH_QUERIES:
        query_embedding = embedder.embed_query(query)
        results = store.search(query_embedding, top_k=TOP_K)

        if not results:
            print(f"\nQuery: {query}\n  No results.")
            continue

        top_score = results[0][1]
        bottom_score = results[-1][1]
        gap = top_score - bottom_score
        gaps.append(gap)

        print(f"\nQuery: {query}")
        print(f"  Top-1 score:    {top_score:.4f}  ({results[0][0]['chunk_id']})")
        print(f"  Bottom-of-{TOP_K} score: {bottom_score:.4f}  ({results[-1][0]['chunk_id']})")
        print(f"  Score gap: {gap:.4f}")

    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    print(f"\n{'='*70}\nCONTROL TEST RESULT\n{'='*70}")
    print(f"Average score gap - MuRIL on ENGLISH data: {avg_gap:.4f}")
    print("\nCompare this to the earlier results:")
    print("  MiniLM on English: 0.1301")
    print("  MuRIL on Hindi:     0.0005")
    print()

    if avg_gap < 0.01:
        print("=> MuRIL shows compressed scores on English too.")
        print("   This suggests score compression is a property of MuRIL ITSELF,")
        print("   not something specific to Hindi as a language.")
    else:
        print("=> MuRIL shows normal/reasonable separation on English.")
        print("   This suggests the compression seen on Hindi data is likely")
        print("   specific to how MuRIL handles Hindi/Devanagari text,")
        print("   not a general weakness of the model architecture.")


if __name__ == "__main__":
    main()
