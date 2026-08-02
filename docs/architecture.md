# System Architecture

## Project
Benchmarking and Improving Retrieval-Augmented Generation for Low-Resource and Code-Mixed Indian Languages

## Overview
The system is a modular Retrieval-Augmented Generation (RAG) pipeline designed to be language-agnostic at each stage, allowing embedding models, vector stores, and LLMs to be swapped independently. This is essential to the project's core goal: benchmarking retrieval and generation quality across English, Hindi, and (in the Major Project phase) code-mixed queries — not just building a single working chatbot.

## High-Level Flow

```
User Query
   │
   ▼
Language Detection / Tagging
   │
   ▼
Embedding Model Selection (MuRIL for Hindi / MiniLM for English)
   │
   ▼
Vector Retrieval (FAISS) — top-k relevant chunks
   │
   ▼
LLM Generation (Llama 3.1 via Ollama) — answer with citations
   │
   ▼
Response returned to user
```

## Component Breakdown

### 1. Ingestion Layer (`ingestion/`)
Responsible for loading raw text (Wikipedia dumps, AI4Bharat corpora), cleaning it (removing HTML/markup), and chunking it into retrieval-sized passages (~500 tokens, configurable overlap).

### 2. Embedding Layer (`embeddings/`)
A factory pattern selects the embedding model based on detected query/document language:
- **MiniLM (`all-MiniLM-L6-v2`)** — English baseline
- **MuRIL** — Hindi and Indic-language embeddings

This is the core variable the project benchmarks — different embedding models will be swapped in and their retrieval quality compared.

### 3. Vector Store (`vectorstore/`)
FAISS is used to index and query embedded chunks. Separate indices are maintained per language/embedding-model combination so retrieval quality can be isolated and measured independently rather than mixed into one index.

### 4. LLM Layer (`llm/`)
Handles generation using a locally-hosted LLM via Ollama (Llama 3.1 8B for the Minor Project baseline). Prompt templates enforce citation formatting so generated answers can be traced back to retrieved source chunks.

### 5. RAG Chain (`chains/`)
Orchestrates the full retrieve → generate flow for a single query, tying together the embedding, vector store, and LLM layers.

### 6. Evaluation Layer (`evaluation/`)
Runs the benchmark: takes a fixed test query set (from `data/eval_sets/`), executes it through the pipeline for each language configuration, and computes retrieval metrics (Recall@k) and answer-quality scores. This is the project's core research output for the Minor Project — a quantified comparison of English vs Hindi RAG performance.

### 7. Services Layer (`services/`)
High-level orchestration exposed to the rest of the application (and eventually a simple UI) — takes a raw user query and returns a final grounded answer by coordinating all the layers above.

## Design Principles

- **Modularity over completeness (for now).** Each component is built as a swappable module because the entire point of the project is comparison, not a single fixed pipeline.
- **Language as a first-class parameter**, not an afterthought — every layer (embedding, retrieval, generation) is designed to accept a language/model configuration rather than assuming English.
- **Reproducibility.** All benchmark runs are driven by config files (`config/config.yaml`) and fixed evaluation sets, so results can be regenerated and compared across model swaps.

## Minor Project Scope of This Architecture
For the Minor Project, only the English and Hindi paths are implemented and benchmarked using one embedding model per language and one LLM. Code-mixed (Hinglish) query handling, multi-model comparative benchmarking, and failure-mode analysis are explicitly out of scope for the Minor Project and are planned for the Major Project phase (see project roadmap).

## Future Extensions (Major Project)
- Code-mixed query handling via a language-detection/normalization layer before embedding
- Comparative evaluation across multiple embedding models (MuRIL vs IndicBERT vs LaBSE) and multiple LLMs
- A verifier/grounding-check agent to detect and correct unsupported claims in generated answers
- Failure-mode analysis module to categorize *why* retrieval fails for certain query types
