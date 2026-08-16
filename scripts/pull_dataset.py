"""
scripts/pull_dataset.py

Pulls a small real English + Hindi dataset from Wikipedia, runs it through
the full ingestion pipeline (load -> clean -> chunk), and saves the
resulting chunks to data/processed/ as JSON — ready to be embedded and
indexed in the next step.

This is intentionally a SMALL starter set (~15-20 articles per language)
so you can validate the whole pipeline end-to-end quickly, before scaling
up to the full AI4Bharat IndicCorp / larger Wikipedia dump later.

Run from the project root:
    python scripts/pull_dataset.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.loader import load_wikipedia_articles
from ingestion.cleaner import clean_documents
from ingestion.chunker import chunk_documents

# Starter article lists — topically similar across languages so retrieval
# quality can be fairly compared later (both cover general knowledge domains,
# not deliberately overlapping topics, since IndicQA-style eval sets come next).
ENGLISH_TITLES = [
    "Artificial intelligence",
    "Machine learning",
    "Natural language processing",
    "Retrieval-augmented generation",
    "India",
    "Climate change",
    "Solar System",
    "World War II",
    "Photosynthesis",
    "Democracy",
    "Internet",
    "Computer science",
    "History of India",
    "Taj Mahal",
    "Cricket",
]

HINDI_TITLES = [
    "कृत्रिम बुद्धिमत्ता",       # Artificial intelligence
    "मशीन लर्निंग",              # Machine learning
    "भारत",                      # India
    "जलवायु परिवर्तन",           # Climate change
    "सौर मंडल",                  # Solar System
    "द्वितीय विश्व युद्ध",        # World War II
    "प्रकाश संश्लेषण",           # Photosynthesis
    "लोकतंत्र",                  # Democracy
    "इंटरनेट",                   # Internet
    "कंप्यूटर विज्ञान",          # Computer science
    "भारत का इतिहास",           # History of India
    "ताज महल",                   # Taj Mahal
    "क्रिकेट",                   # Cricket
]

OUTPUT_DIR = "data/processed"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def build_dataset(titles, lang, output_filename):
    print(f"\n=== Pulling {lang.upper()} dataset ({len(titles)} articles) ===")

    raw_docs = load_wikipedia_articles(titles, lang=lang)
    print(f"[INFO] Successfully fetched {len(raw_docs)}/{len(titles)} articles.")

    cleaned_docs = clean_documents(raw_docs, min_length=200)
    chunks = chunk_documents(cleaned_docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(chunks)} chunks to '{output_path}'")
    return chunks


if __name__ == "__main__":
    en_chunks = build_dataset(ENGLISH_TITLES, "en", "en_chunks.json")
    hi_chunks = build_dataset(HINDI_TITLES, "hi", "hi_chunks.json")

    print("\n=== Summary ===")
    print(f"English chunks: {len(en_chunks)}")
    print(f"Hindi chunks: {len(hi_chunks)}")
    print("\nNext step: embed and index these using vectorstore/index_manager.py")