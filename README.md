# Multilingual RAG Benchmark
### Benchmarking and Improving Retrieval-Augmented Generation for Low-Resource and Code-Mixed Indian Languages

[![Status](https://img.shields.io/badge/status-Minor%20Project-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Overview

Most Retrieval-Augmented Generation (RAG) systems are built and evaluated almost exclusively in English. When queried in Hindi, or in natural code-mixed Hinglish, retrieval quality and answer grounding degrade — but there is little rigorous, end-to-end benchmarking of *why* and *how much*.

This project builds a modular RAG pipeline that plugs in different embedding models (MuRIL, IndicBERT) and LLMs, then systematically benchmarks retrieval and generation quality across **English → Hindi → code-mixed** queries. The goal is a reusable evaluation framework and a set of concrete, publishable findings on where and why multilingual RAG breaks down for Indian languages — not another single-language chatbot.

> This project positions itself as an **evaluator/benchmarker** of existing embedding and language models (e.g. MuRIL, IndicBERT, and open Indic LLMs), not as an attempt to build a competing foundation model.

---

## Project Status

**Current Phase:** Minor Project (20 July 2026 – 9 October 2026)

- [x] Literature review & research gap finalized
- [x] Requirement analysis & system architecture drafted
- [x] Dataset collection & preprocessing (English + Hindi) — 343 English chunks, 152 Hindi chunks from Wikipedia
- [x] Embedding model & vector store setup (MuRIL, MiniLM, FAISS)
- [x] Retrieval pipeline (English) — verified working end-to-end
- [x] Retrieval pipeline (Hindi) — verified working end-to-end
- [ ] LLM integration for generation
- [ ] Baseline benchmarking (Recall@k, answer relevance)
- [ ] Minor project report & demo

**Major Project (October 2026 – January 2027)** will extend this to code-mixed query handling, multi-model comparative evaluation, failure-mode analysis, deployment, and an IEEE-format research paper.

---

## Preliminary Finding (Week 5)

Initial manual testing on real Wikipedia-derived data surfaced an early, genuinely useful result: **English retrieval (via MiniLM) shows clear, well-separated similarity scores between relevant and irrelevant chunks, while Hindi retrieval (via base MuRIL) shows heavily compressed scores** — top results across completely unrelated topics scored within 0.001 of each other. The correct answer still ranked first in both languages, but the *confidence gap* between right and wrong answers was far weaker for Hindi.

This is consistent with prior literature (see `docs/literature_review.md`) showing that base MuRIL is not always strong on semantic similarity tasks specifically, even when it performs well on classification tasks. See `docs/preliminary_findings.md` for full details, scores, and caveats. This single-query observation motivates — but does not yet prove — the project's core hypothesis, and will be tested rigorously with a full query set during Week 8 benchmarking.

---

## Architecture (High-Level)

```
Query → Language Detection → Embedding (MuRIL / MiniLM)
      → Vector Retrieval (FAISS) → LLM Generation (Llama 3.1 via Ollama)
      → Grounded Answer with Citations
```

See [`docs/architecture.md`](docs/architecture.md) for the full design, [`docs/requirement_analysis.md`](docs/requirement_analysis.md) for functional requirements, and [`docs/preliminary_findings.md`](docs/preliminary_findings.md) for early retrieval-quality observations.

---

## Repository Structure

```
multilingual-rag-benchmark/
├── data/
│   ├── raw/                  # (gitignored) raw downloads
│   ├── processed/             # en_chunks.json, hi_chunks.json — real processed data
│   └── eval_sets/              # evaluation query sets (Week 8+)
├── ingestion/          # loader.py, cleaner.py, chunker.py — tested, working
├── embeddings/          # MuRIL, MiniLM embedders + factory — tested, working
├── vectorstore/          # FAISS store + index manager — tested, working
├── llm/                    # LLM client and prompt templates (not yet built)
├── chains/                  # Core retrieve → generate pipeline logic (not yet built)
├── evaluation/                # Retrieval metrics and benchmark runner (Week 8)
├── services/                    # High-level query orchestration (Week 5-6)
├── scripts/                      # pull_dataset.py, test_retrieval.py, test_retrieval_hi.py
├── tests/                          # Unit tests
├── docs/                            # architecture, requirements, literature review, findings
└── config/                            # Central YAML configuration
```

---

## Tech Stack

- **Embeddings:** MuRIL (`google/muril-base-cased`) for Hindi · MiniLM (`all-MiniLM-L6-v2`) for English baseline
- **Vector Store:** FAISS (`IndexFlatIP`, cosine similarity via normalized embeddings)
- **LLM:** Llama 3.1 8B via Ollama (local inference) — planned, not yet integrated
- **Language:** Python 3.10+
- **Datasets:** Wikipedia (English + Hindi, 15 + 13 articles as starter corpus); AI4Bharat IndicCorp/IndicQA planned for full-scale benchmarking

---

## Getting Started

```bash
git clone https://github.com/<org-or-username>/multilingual-rag-benchmark.git
cd multilingual-rag-benchmark
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Pull the starter dataset
```bash
python scripts/pull_dataset.py
```
Downloads ~15 English and ~13 Hindi Wikipedia articles, cleans and chunks them, and saves to `data/processed/en_chunks.json` and `hi_chunks.json`.

### Test retrieval end-to-end
```bash
python scripts/test_retrieval.py       # English (MiniLM)
python scripts/test_retrieval_hi.py    # Hindi (MuRIL)
```
Builds a real FAISS index from the processed chunks and runs a sample query, printing the top-3 retrieved chunks with similarity scores.

---

## Research Motivation

Existing multilingual embedding models are rarely benchmarked *within a full RAG pipeline* — most evaluation stops at embedding-level similarity or classification tasks. This project aims to close that gap by measuring end-to-end retrieval and generation quality on a self-curated benchmark spanning English, Hindi, and code-mixed queries, and by characterizing specific failure modes (transliteration mismatches, script-mixing, ambiguous romanization, score compression) that general-purpose evaluations miss.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Built as part of the NTCC Minor/Major Project at Amity School of Engineering & Technology, Greater Noida. Pipeline architecture inspired by common open-source RAG implementation patterns; extended substantially with multilingual embedding support and a dedicated evaluation framework.
