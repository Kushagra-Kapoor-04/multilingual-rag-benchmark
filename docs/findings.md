# Preliminary Findings — Week 5

## Test Setup
- **English:** 343 chunks from 15 Wikipedia articles, embedded with `all-MiniLM-L6-v2`, indexed in FAISS.
- **Hindi:** 152 chunks from 13 Wikipedia articles, embedded with `google/muril-base-cased` (mean-pooled, L2-normalized), indexed in FAISS.
- Both indices built and queried using the same pipeline (`embeddings/embedder_factory.py` + `vectorstore/index_manager.py`), so results are directly comparable.

## Test 1: English Query
**Query:** "What is artificial intelligence?"

| Rank | Chunk | Score |
|---|---|---|
| 1 | `Artificial_intelligence_chunk0` | 0.683 |
| 2 | `Computer_science_chunk8` | 0.634 |
| 3 | `Artificial_intelligence_chunk31` | 0.554 |

**Observation:** Correct top result, and scores show a clear, sensible gradient — the AI-related chunks score meaningfully higher than the next-most-related chunk. Score separation between relevant and less-relevant results is roughly 0.05–0.13 per rank.

## Test 2: Hindi Query
**Query:** "कृत्रिम बुद्धिमत्ता क्या है?" (What is artificial intelligence?)

| Rank | Chunk | Score |
|---|---|---|
| 1 | `कृत्रिम_बुद्धिमत्ता_chunk0` | 0.995 |
| 2 | `लोकतंत्र_chunk8` (Democracy) | 0.994 |
| 3 | `लोकतंत्र_chunk7` (Democracy) | 0.994 |

**Observation:** Top result is correct, but ranks 2 and 3 are from a completely unrelated article (Democracy) and score within 0.001 of the correct result. All scores are compressed near 1.0, meaning MuRIL's raw mean-pooled embeddings show almost no meaningful separation between relevant and irrelevant content in this test.

## Interpretation

This is a genuine, useful early finding — not a pipeline bug. The same embedding→FAISS pipeline produces well-separated, interpretable similarity scores for English (via MiniLM) but poorly separated, compressed scores for Hindi (via base MuRIL). This is consistent with prior literature reviewed for this project (see `docs/literature_review.md`), which found that base MuRIL embeddings are not automatically strong for semantic similarity tasks, and that fine-tuned sentence-embedding variants (e.g., IndicSBERT-STS) outperform both MuRIL and LaBSE once specifically tuned for similarity rather than classification.

**This directly supports the project's core research motivation:** off-the-shelf multilingual embeddings do not necessarily transfer retrieval quality to Hindi at the same level as well-established English embedding models, even when correct results are technically retrieved at rank 1.

## Caveats
- Single-query test per language so far — not yet a statistically meaningful sample. Needs to be repeated across a larger query set (10+ queries per language minimum) before drawing firm conclusions.
- Small corpus (152–343 chunks) — score compression could partly be an artifact of a small, topically narrow index rather than purely an embedding-quality issue. Needs re-testing once corpus size increases.
- No Recall@k computed yet — this is qualitative/manual inspection only, not the formal benchmarking methodology planned for Week 8.

## Next Steps (does not change current week's plan, just noted for later)
- Repeat this test with 5–10 more Hindi and English queries to see if the score-compression pattern holds.
- Consider testing an alternative Hindi embedding model (e.g., IndicSBERT) as a comparison point once formal benchmarking (Week 8) begins.
- Formal Recall@k benchmarking (per the existing Minor Project plan) will quantify this gap properly rather than relying on manual score inspection.
