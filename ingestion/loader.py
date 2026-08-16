"""
ingestion/loader.py

Loads raw text data for the RAG pipeline from local files or Wikipedia.
Supports plain text/JSON files (for AI4Bharat/IndicQA dumps) and live
Wikipedia article fetching (for quick English/Hindi corpus building).
"""

import json
import os
from pathlib import Path
from typing import List, Dict


def load_text_files(directory: str, extension: str = ".txt") -> List[Dict]:
    documents = []
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for file_path in sorted(directory_path.glob(f"*{extension}")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            documents.append({
                "id": file_path.stem,
                "text": text,
                "source": str(file_path)
            })

    return documents


def load_jsonl(file_path: str, text_field: str = "text") -> List[Dict]:
    documents = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(text_field, "").strip()
            if text:
                documents.append({
                    "id": record.get("id", f"doc_{i}"),
                    "text": text,
                    "source": file_path
                })

    return documents


def load_wikipedia_articles(titles: List[str], lang: str = "en") -> List[Dict]:
    import wikipediaapi

    wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent="multilingual-rag-benchmark/0.1 (student research project)"
    )

    documents = []
    for title in titles:
        page = wiki.page(title)
        if page.exists():
            documents.append({
                "id": title.replace(" ", "_"),
                "text": page.text,
                "source": f"wikipedia:{lang}:{title}"
            })
        else:
            print(f"[WARN] Wikipedia page not found: '{title}' ({lang})")

    return documents


if __name__ == "__main__":
    sample_titles_en = ["Retrieval-augmented generation", "Natural language processing"]
    docs = load_wikipedia_articles(sample_titles_en, lang="en")
    print(f"Loaded {len(docs)} English documents.")
    for d in docs:
        print(f"- {d['id']}: {len(d['text'])} characters")
