"""
ingestion/chunker.py

Splits cleaned documents into retrieval-sized chunks with configurable
size and overlap. Chunking is done by word count (language-agnostic,
avoids needing a language-specific tokenizer at this stage).
"""

from typing import Dict, List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks


def chunk_documents(documents: List[Dict], chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    all_chunks = []

    for doc in documents:
        text_chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, chunk_text_value in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"{doc['id']}_chunk{i}",
                "doc_id": doc["id"],
                "text": chunk_text_value,
                "source": doc.get("source", "unknown")
            })

    print(f"[INFO] Produced {len(all_chunks)} chunks from {len(documents)} document(s).")
    return all_chunks


if __name__ == "__main__":
    sample = {
        "id": "test_doc",
        "text": " ".join([f"word{i}" for i in range(1200)]),
        "source": "manual_test"
    }
    chunks = chunk_documents([sample], chunk_size=500, overlap=100)
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk word count: {len(chunks[0]['text'].split())}")
