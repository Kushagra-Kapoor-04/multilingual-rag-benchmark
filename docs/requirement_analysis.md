# Requirement Analysis

## Project
Benchmarking and Improving Retrieval-Augmented Generation for Low-Resource and Code-Mixed Indian Languages

## 1. Problem Statement
Retrieval-Augmented Generation (RAG) systems are predominantly designed, tuned, and evaluated in English. When queried in Hindi — or in natural code-mixed Hinglish — retrieval quality and answer grounding degrade, but there is limited rigorous, end-to-end benchmarking of this gap at the full pipeline level (as opposed to isolated embedding-model evaluations).

## 2. Objectives

### Minor Project Objectives
1. Build a modular RAG pipeline supporting both English and Hindi queries.
2. Integrate an Indic-tuned embedding model (MuRIL) alongside an English baseline (MiniLM).
3. Establish a reproducible benchmarking methodology (Recall@k, answer relevance).
4. Produce a quantified baseline comparison of English vs Hindi RAG performance.

### Major Project Objectives (forward-looking, not in current scope)
1. Extend the pipeline to handle code-mixed (Hinglish) queries.
2. Run comparative evaluation across multiple embedding models and LLMs.
3. Perform failure-mode analysis to categorize retrieval breakdowns.
4. Introduce a grounding-verification agent to mitigate hallucination.
5. Deploy a public-facing demo and produce an IEEE-format research paper.

## 3. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | The system shall accept a natural language query in English or Hindi. |
| FR2 | The system shall detect or accept the query language as an explicit parameter. |
| FR3 | The system shall generate embeddings for the query using a language-appropriate embedding model. |
| FR4 | The system shall retrieve the top-k most relevant document chunks from a FAISS vector index. |
| FR5 | The system shall generate a natural language answer grounded in the retrieved chunks, using a local LLM. |
| FR6 | The system shall include source citations in generated answers. |
| FR7 | The system shall support running a fixed evaluation query set and computing retrieval/answer-quality metrics automatically. |
| FR8 | The system shall log latency, retrieval scores, and generation outputs for each benchmark run. |

## 4. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR1 | The system shall run entirely on free/open-source tools (no paid API dependencies required for core functionality). |
| NFR2 | The system shall be usable on consumer GPUs (RTX 3060/4060 class) or free cloud tiers (Colab/Kaggle). |
| NFR3 | The codebase shall be modular, allowing embedding models, vector stores, and LLMs to be swapped via configuration rather than code changes. |
| NFR4 | Benchmark results shall be reproducible given the same configuration and dataset version. |
| NFR5 | The system shall handle a single query end-to-end (embedding → retrieval → generation) within a reasonable latency for live demonstration (target: under 15 seconds on available hardware). |

## 5. Success Criteria (Minor Project)
- A working RAG pipeline that can answer queries in both English and Hindi.
- A benchmark report showing Recall@k and answer-relevance scores for English vs Hindi on a fixed test set of at least 50 queries per language.
- Documented, reproducible evaluation methodology that can be directly extended in the Major Project.

## 6. Constraints
- **Budget:** Software-based only; free/open-source tools, public datasets, free cloud tiers. No paid subscriptions or hardware purchases expected.
- **Compute:** Limited to two team members' personal GPUs (RTX 3060/4060) and free-tier Colab/Kaggle sessions — no dedicated server or paid cloud compute.
- **Timeline:** Minor Project must be complete by 9 October 2026.
- **Team bandwidth:** Variable weekly availability (5–20+ hours/week combined) due to concurrent placement activities.

## 7. Assumptions
- Publicly available datasets (AI4Bharat IndicCorp, IndicQA, Wikipedia dumps) are sufficient for building a representative evaluation set.
- A locally-hosted open LLM (Llama 3.1 8B via Ollama) provides adequate generation quality for baseline benchmarking without requiring paid API access.
- Team members without prior ML/NLP background will be brought up to working proficiency through guided onboarding from the AI-specialization team member.

## 8. Out of Scope (Minor Project)
- Code-mixed/Hinglish query handling
- Support for languages beyond English and Hindi
- Public deployment or hosted demo
- Comparative benchmarking across multiple embedding models/LLMs (single model per language only)
- Agentic/multi-agent components (planned for Major Project)
