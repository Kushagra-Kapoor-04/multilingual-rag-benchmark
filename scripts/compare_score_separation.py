"""
scripts/compare_score_separation.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectorstore.index_manager import IndexManager

ENGLISH_QUERIES = [
    "What is artificial intelligence?",
    "How does photosynthesis work?",
    "What caused World War II?",
    "What is democracy?",
    "Tell me about the Taj Mahal.",
]

HINDI_QUERIES = [
    "\u0915\u0943\u0924\u094d\u0930\u093f\u092e \u092c\u0941\u0926\u094d\u0927\u093f\u092e\u0924\u094d\u0924\u093e \u0915\u094d\u092f\u093e \u0939\u0948?",
    "\u092a\u094d\u0930\u0915\u093e\u0936 \u0938\u0902\u0936\u094d\u0932\u0947\u0937\u0923 \u0915\u0948\u0938\u0947 \u0939\u094b\u0924\u093e \u0939\u0948?",
    "\u0926\u094d\u0935\u093f\u0924\u0940\u092f \u0935\u093f\u0936\u094d\u0935 \u092f\u0941\u0926\u094d\u0927 \u0915\u093e \u0915\u093e\u0930\u0923 \u0915\u094d\u092f\u093e \u0925\u093e?",
    "\u0932\u094b\u0915\u0924\u0902\u0924\u094d\u0930 \u0915\u094d\u092f\u093e \u0939\u0948?",
    "\u0924\u093e\u091c \u092e\u0939\u0932 \u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u092c\u0924\u093e\u090f\u0902\u0964",
]

TOP_K = 5


def run_comparison(manager, language, queries):
    gaps = []
    print(f"\n{'='*70}\n{language.upper()} RESULTS\n{'='*70}")

    for query in queries:
        results = manager.query(language, query, top_k=TOP_K)
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
        print(f"  Score gap (confidence spread): {gap:.4f}")

    return gaps


def main():
    with open("data/processed/en_chunks.json", encoding="utf-8") as f:
        en_chunks = json.load(f)
    with open("data/processed/hi_chunks.json", encoding="utf-8") as f:
        hi_chunks = json.load(f)

    manager = IndexManager()
    manager.build_index("en", en_chunks)
    manager.build_index("hi", hi_chunks)

    en_gaps = run_comparison(manager, "en", ENGLISH_QUERIES)
    hi_gaps = run_comparison(manager, "hi", HINDI_QUERIES)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    avg_en_gap = sum(en_gaps) / len(en_gaps) if en_gaps else 0
    avg_hi_gap = sum(hi_gaps) / len(hi_gaps) if hi_gaps else 0

    print(f"Average score gap (English/MiniLM): {avg_en_gap:.4f}")
    print(f"Average score gap (Hindi/MuRIL):    {avg_hi_gap:.4f}")

    if avg_hi_gap < avg_en_gap:
        diff_pct = ((avg_en_gap - avg_hi_gap) / avg_en_gap) * 100
        print(f"\nHindi shows {diff_pct:.1f}% LESS score separation than English on average.")
    else:
        print("\nHindi does NOT show less score separation on average.")


if __name__ == "__main__":
    main()
