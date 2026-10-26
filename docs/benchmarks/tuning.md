# Retrieval tuning (Week 6)

**Run date:** 2026-10-22. **Corpus:** the BEIR paper (222 chunks). **K = 3**,
the number of chunks the generator receives.

To reproduce (no generation calls, 22 query embeddings):

```bash
docker exec -i ragflow_backend python - < scripts/tune_retrieval.py
```

**Method.** Settings are chosen on the 11 odd-numbered benchmark questions and
confirmed on the 11 even-numbered ones. A default changes only if the winner
also beats it on the held-out half. With 11 questions per half, one question
can move a half's mean nDCG@3 by up to 0.09, so small wins on one half are
treated as noise.

## 1. BM25 tokeniser fix

The Week 4 tokeniser was `text.lower().split()`. Punctuation stayed attached
("BM25," never matched "bm25"), and words like "what", "is" and "the" matched
nearly every chunk. It now keeps lowercase alphanumeric words and drops a
short stopword list (`_tokenize` in `backend/app/rag_pipelines.py`).

All 22 questions, default settings:

| Strategy | Tokeniser | Recall@3 | MRR | nDCG@3 |
|---|---|---:|---:|---:|
| BM25 | whitespace split (old) | 0.508 | 0.508 | 0.328 |
| BM25 | words, no stopwords (new) | **0.765** | **0.773** | **0.564** |
| Hybrid | whitespace split (old) | **0.848** | 0.803 | 0.625 |
| Hybrid | words, no stopwords (new) | 0.788 | 0.803 | **0.649** |
| Hybrid + CE | whitespace split (old) | **0.864** | 0.879 | **0.644** |
| Hybrid + CE | words, no stopwords (new) | 0.841 | 0.879 | 0.623 |

- **BM25 on its own improves by 72% in nDCG@3** (0.328 to 0.564), and MRR rises
  from 0.508 to 0.773. This was a bug, not a tuning choice, so the fix stays.
- **For the fused strategies the effect is mixed and small.** Hybrid gains
  nDCG@3 but loses Recall@3, and Hybrid + CE drops slightly. Each shift is one
  or two questions out of 22. A stronger BM25 now pulls its own top hits into
  the fused list, and on this corpus dense retrieval is usually already right.

## 2. Candidate pool and RRF constant

nDCG@3 on the tuning half / the confirmation half.

**Hybrid (RRF):**

| Pool | k=10 | k=30 | k=60 | k=100 |
|---:|---:|---:|---:|---:|
| 5 | 0.625 / 0.695 | 0.625 / 0.695 | 0.625 / 0.695 | 0.625 / 0.695 |
| 8 | 0.583 / 0.716 | 0.583 / 0.716 | **0.583 / 0.716 (default)** | 0.583 / 0.716 |
| 10 | 0.609 / 0.695 | 0.604 / 0.695 | 0.604 / 0.695 | 0.604 / 0.695 |
| 15 | 0.609 / 0.689 | 0.583 / 0.716 | 0.583 / 0.716 | 0.583 / 0.716 |
| 20 | 0.609 / 0.711 | 0.583 / 0.668 | 0.583 / 0.668 | 0.583 / 0.668 |

**Hybrid + Cross-Encoder:**

| Pool | k=10 | k=30 | k=60 | k=100 |
|---:|---:|---:|---:|---:|
| 5 | 0.588 / 0.652 | 0.588 / 0.652 | 0.588 / 0.652 | 0.588 / 0.652 |
| 8 | 0.594 / 0.652 | 0.594 / 0.652 | **0.594 / 0.652 (default)** | 0.594 / 0.652 |
| 10 | 0.594 / 0.652 | 0.594 / 0.652 | 0.594 / 0.652 | 0.594 / 0.652 |
| 15 | 0.636 / 0.652 | 0.636 / 0.652 | 0.636 / 0.652 | 0.636 / 0.652 |
| 20 | 0.636 / 0.620 | 0.636 / 0.620 | 0.636 / 0.620 | 0.636 / 0.620 |

**Result: the defaults stay (pool 8, RRF k = 60).**

- Hybrid's best tuning-half setting (pool 5, k = 10) scores lower than the
  default on the held-out half (0.695 vs 0.716).
- Hybrid + CE's best (pool 15) only ties the default on the held-out half
  (0.652), and pool 20 is worse. A larger pool also means more cross-encoder
  calls, so more latency, for no measured gain.
- The RRF constant barely matters here. With only two lists of at most 20
  items, k changes the order of the fused scores only when the two retrievers
  disagree near the top.

## 3. Distance cutoff: not tuned

The 0.8 L2 gate never fires on this benchmark: relevant chunks sit around
0.4-0.5. Every benchmark question is answerable, so tuning the gate here
would only push it upward. Tuning it properly needs off-topic questions
labelled "no answer", which the dataset does not have; see the Limitations
section of the README.
