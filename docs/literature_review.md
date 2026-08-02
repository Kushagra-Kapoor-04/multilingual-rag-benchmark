# Literature Review

## Project
Benchmarking and Improving Retrieval-Augmented Generation for Low-Resource and Code-Mixed Indian Languages

## Purpose
This review summarizes existing work in three areas directly relevant to the project: (1) multilingual RAG systems and their known limitations, (2) embedding models for Indian languages, and (3) code-mixed (Hindi-English) NLP. It is a working document — update it as more papers are reviewed through the Minor Project phase.

---

## 1. Multilingual RAG: Known Gaps

**Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Tasks** (Ranaldi et al., 2025) evaluates several strategies for extending RAG beyond English, including translating queries into English before retrieval versus retrieving directly across multiple languages, and finds that dedicated multilingual retrieval strategies improve performance for both high- and low-resource languages compared to naive approaches.

**Multilingual RAG for Culturally-Sensitive Tasks** (Li et al., 2024/2025) provides one of the more detailed cross-lingual RAG robustness studies. It finds that retrievers show a systematic preference for retrieving documents in the query language or in high-resource languages, and that citation behavior in generated answers varies far more for low-resource languages than for high-resource ones — suggesting the grounding problem is not uniform across languages, motivating dedicated per-language evaluation rather than a single aggregate score.

A broader synthesis of the multilingual RAG literature confirms two recurring themes: (1) retrieval systems are biased toward high-resource, Indo-European languages due to corpus prevalence and encoder pretraining, and (2) proposed mitigations include corpus upsampling, domain-adaptive pretraining, and better native-language benchmarking rather than translation-based workarounds.

**Relevance to this project:** These findings directly motivate the core research question — none of these works specifically isolate and benchmark Hindi or code-mixed Indian-language queries within a full retrieve-then-generate pipeline; most multilingual RAG evaluation to date has focused on higher-resource non-English languages or broad multilingual averages rather than Indian-language-specific, pipeline-level analysis.

---

## 2. Embedding Models for Indian Languages

**MuRIL (Multilingual Representations for Indian Languages)**, released by Google, is a BERT-based model trained specifically on 17 Indian languages and their transliterated forms, extending standard multilingual BERT training with India-specific data.

**IndicBERT**, released by AI4Bharat, is an ALBERT-based multilingual model pretrained on 12 major Indian languages as part of the broader IndicNLPSuite ecosystem.

A comparative study evaluating embedding models specifically on Hindi-English code-mixed queries (university domain chatbot data) found that MuRIL consistently outperformed IndicBERT, XLM-RoBERTa, and mBERT on intent classification and entity recognition, with the performance gap widening as code-mixing intensity increased.

Separately, work on Hindi/Marathi sentence embeddings found that base MuRIL embeddings are not automatically superior for sentence-similarity tasks — LaBSE performed competitively or better in several zero-shot settings, and specialized fine-tuned sentence-embedding models (e.g., IndicSBERT-STS) outperformed both MuRIL and LaBSE once fine-tuned specifically for semantic similarity.

A more recent line of work (**DeepRAG**, 2025) argues that general-purpose multilingual embeddings remain a bottleneck for Hindi RAG specifically, and proposes training a Hindi-dedicated embedding model from scratch, reporting improved retrieval precision over multilingual alternatives.

**Relevance to this project:** This is directly informative for the embedding-model choice in the Minor Project. MuRIL is a reasonable and well-supported starting point (strong track record specifically on Hindi and code-mixed tasks), but the literature is not unanimous — some studies show LaBSE or fine-tuned sentence-embedding variants outperforming it depending on the task. This supports the project's decision to keep the embedding layer swappable (via `embeddings/embedder_factory.py`) rather than hard-committing to one model, so this can be revisited empirically once baseline benchmarking begins.

---

## 3. Code-Mixed (Hindi-English / Hinglish) NLP

Code-mixed text handling remains a distinct challenge from standard multilingual NLP because it involves intra-sentence language switching, transliteration ambiguity (romanized Hindi has no single standard spelling), and script mixing — issues that don't arise in same-script or single-language-per-document settings.

Related work on code-switched retrieval in other language pairs (e.g., Tagalog-English) confirms this is not an Indian-language-specific problem: retrieval and generation quality for purely non-English or code-switched queries consistently underperforms compared to English queries, even when the underlying LLM is otherwise capable.

**Relevance to this project:** This confirms code-mixed query handling as a genuinely separate research problem from monolingual Hindi retrieval, and validates treating it as a distinct, later phase of the project (Major Project) rather than folding it into the Minor Project's English/Hindi baseline.

---

## 4. Summary of Research Gap

Across the reviewed literature, three consistent gaps emerge:

1. Most multilingual RAG evaluation happens at the level of broad multilingual benchmarks or high-resource non-English languages; **dedicated, pipeline-level (not just embedding-level) evaluation for Hindi and Hindi-English code-mixed queries is comparatively rare.**
2. Embedding model comparisons for Indian languages exist, but are **rarely evaluated inside a full RAG pipeline** (retrieval + generation together) — most compare embeddings in isolation (classification, similarity tasks) rather than measuring end-to-end answer quality and grounding.
3. Code-mixed query handling is acknowledged as a harder problem across multiple language pairs, but **there is limited work specifically benchmarking RAG systems on natural Hindi-English code-mixed queries** rather than clean monolingual Hindi or English.

This project's contribution is positioned directly at this intersection: an end-to-end, reproducible benchmark of RAG performance across English, Hindi, and (in the Major Project) code-mixed queries, evaluated with a consistent methodology and multiple embedding/LLM configurations — treating pipeline-level evaluation itself as the contribution, not a new foundation model.

---

## References

1. Ranaldi, L., Haddow, B., & Birch, A. (2025). *Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Task*. arXiv:2504.03616.
2. Li, B., et al. (2024/2025). *Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness*. arXiv:2410.01171.
3. Emergent Mind. *Multilingual Retrieval-Augmented Generation* (topic synthesis). https://www.emergentmind.com/topics/multilingual-retrieval-augmented-generation
4. Khanuja, S., et al. *MuRIL: Multilingual Representations for Indian Languages*. Google Research.
5. AI4Bharat. *IndicBERT and IndicNLPSuite*. https://github.com/AI4Bharat/indicnlp_catalog
6. Margaj, S., et al. (2025). *Comparative Analysis of Embedding Models for Hindi-English Code-Mixed University Related Queries*. Voice of Creative Research, Vol. 7, Issue 2.
7. Joshi, R., et al. *L3Cube-MahaSBERT and HindSBERT: Sentence Embeddings for Marathi and Hindi*. arXiv:2211.11187.
8. Deshpande, S., et al. *L3Cube-IndicSBERT: Cross-lingual Sentence Representations using Multilingual BERT*. arXiv:2304.11434.
9. Nandakishor, M. (2025). *DeepRAG: Building a Custom Hindi Embedding Model for Retrieval Augmented Generation from Scratch*. arXiv:2503.08213.
10. Castro, J. A. D. V., et al. *Benchmarking Open-Source Large Language Models on Code-Switched Tagalog-English Retrieval Augmented Generation*.

*(This list will grow as more papers are added through Weeks 1–3 of the Minor Project. Aim for 15–20 total entries before the literature review is finalized in the Minor Project report.)*
