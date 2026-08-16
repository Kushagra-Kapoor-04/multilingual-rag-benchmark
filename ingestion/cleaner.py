"""
ingestion/cleaner.py

Cleans and normalizes raw document text before chunking/embedding.
Handles HTML stripping, whitespace normalization, and basic
Devanagari-aware cleanup for Hindi text.
"""

import re
from typing import Dict, List

from bs4 import BeautifulSoup


def strip_html(text: str) -> str:
    if "<" in text and ">" in text:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_wiki_artifacts(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"={2,}\s*.*?\s*={2,}", "", text)
    return text


def clean_document(text: str) -> str:
    text = strip_html(text)
    text = remove_wiki_artifacts(text)
    text = normalize_whitespace(text)
    return text


def clean_documents(documents: List[Dict], min_length: int = 50) -> List[Dict]:
    cleaned = []
    dropped = 0

    for doc in documents:
        cleaned_text = clean_document(doc["text"])
        if len(cleaned_text) >= min_length:
            new_doc = dict(doc)
            new_doc["text"] = cleaned_text
            cleaned.append(new_doc)
        else:
            dropped += 1

    if dropped:
        print(f"[INFO] Dropped {dropped} document(s) shorter than {min_length} characters after cleaning.")

    return cleaned


if __name__ == "__main__":
    sample = {
        "id": "test_doc",
        "text": "<p>Retrieval-Augmented Generation (RAG) improves LLM factuality.[1]</p>   \n\n== See also ==",
        "source": "manual_test"
    }
    result = clean_documents([sample])
    print(result)
