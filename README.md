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
- [ ] Dataset collection & preprocessing (English + Hindi)
- [ ] Embedding model & vector store setup
- [ ] Retrieval pipeline (English)
- [ ] Retrieval pipeline (Hindi)
- [ ] LLM integration for generation
- [ ] Baseline benchmarking (Recall@k, answer relevance)
- [ ] Minor project report & demo

**Major Project (October 2026 – January 2027)** will extend this to code-mixed query handling, multi-model comparative evaluation, failure-mode analysis, deployment, and an IEEE-format research paper.

---

## Team

| Member | Branch | Role |
|---|---|---|
| Student 1 | AI | Technical Lead — embeddings, retrieval design, evaluation methodology |
| Student 2 | CSE | Backend/Pipeline Engineer — RAG orchestration, API layer |
| Student 3 | CSE | Data & Evaluation Engineer — dataset handling, benchmarking |
| Student 4 | IT | Infrastructure & Documentation Lead — vector DB/deployment, reports |

**Faculty Guide:** Dr Amar Deep Gupta [90037]

---

## Architecture (High-Level)

```
Query → Language Detection → Embedding (MuRIL / MiniLM)
      → Vector Retrieval (FAISS) → LLM Generation (Llama 3.1 via Ollama)
      → Grounded Answer with Citations
```

See [`docs/architecture.md`](docs/architecture.md) for the full design and [`docs/requirement_analysis.md`](docs/requirement_analysis.md) for functional requirements and success criteria.

---

## Repository Structure

```
multilingual-rag-benchmark/
├── data/            # Raw and processed corpora, evaluation query sets
├── ingestion/        # Data loading, cleaning, chunking
├── embeddings/        # Embedding model wrappers (MuRIL, MiniLM, IndicBERT)
├── vectorstore/        # FAISS index build/query/management
├── llm/                 # LLM client and prompt templates
├── chains/               # Core retrieve → generate pipeline logic
├── evaluation/            # Retrieval metrics and benchmark runner
├── services/               # High-level query orchestration
├── scripts/                 # CLI entry points (ingest, index, benchmark)
├── tests/                     # Unit tests
├── docs/                       # Architecture, requirements, literature review
└── config/                      # Central YAML configuration
```

---

## Tech Stack

- **Embeddings:** MuRIL, IndicBERT (Hindi) · MiniLM (English baseline)
- **Vector Store:** FAISS
- **LLM:** Llama 3.1 8B via Ollama (local inference)
- **Language:** Python 3.10+
- **Datasets:** AI4Bharat IndicCorp, IndicQA, Wikipedia (English/Hindi)

---

## Getting Started

```bash
git clone https://github.com/<org-or-username>/multilingual-rag-benchmark.git
cd multilingual-rag-benchmark
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Setup instructions for dataset download, indexing, and running benchmarks will be added here as those components are built (Week 3 onward).

---

## Research Motivation

Existing multilingual embedding models are rarely benchmarked *within a full RAG pipeline* — most evaluation stops at embedding-level similarity or classification tasks. This project aims to close that gap by measuring end-to-end retrieval and generation quality on a self-curated benchmark spanning English, Hindi, and code-mixed queries, and by characterizing specific failure modes (transliteration mismatches, script-mixing, ambiguous romanization) that general-purpose evaluations miss.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Built as part of the NTCC Minor/Major Project at Amity School of Engineering & Technology, Greater Noida. Pipeline architecture inspired by common open-source RAG implementation patterns; extended substantially with multilingual embedding support and a dedicated evaluation framework.