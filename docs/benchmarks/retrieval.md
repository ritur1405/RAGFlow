# Retrieval benchmark: Dense vs BM25 vs Hybrid

This is the plan's Week 4 deliverable. It compares all five retrieval strategies
on the shared, page-labelled benchmark, with the Week 5 retrieval metrics.

**Run date:** 2026-10-21. **Corpus:** the BEIR paper (24 pages, 222 cleaned
chunks).

Raw results: [retrieval-k3.json](retrieval-k3.json), [retrieval-k5.json](retrieval-k5.json).
To reproduce, use the **Experiments** page, "Benchmark every retriever on the
shared dataset", or:

```bash
curl -X POST localhost:8000/api/experiments/benchmark -H "Content-Type: application/json" -d '{"top_k": 3}'
```

**Cost:** 22 query embeddings for the first run, then 0, because the
embeddings are cached in-process. **No generation calls.** The whole benchmark
stays within the Gemini free tier.

## Results at K=3

K=3 is the number of chunks the generator receives.

| Strategy | Recall@3 | Precision@3 | MRR | nDCG@3 | Retrieval latency* |
|---|---:|---:|---:|---:|---:|
| Baseline (dense, no gate) | **0.864** | **0.606** | **0.894** | **0.655** | 4 ms |
| Dense (distance gate 0.8) | **0.864** | **0.606** | **0.894** | **0.655** | 2 ms |
| BM25 | 0.508 | 0.303 | 0.508 | 0.328 | 13 ms |
| Hybrid (Dense + BM25, RRF) | 0.848 | **0.606** | 0.803 | 0.625 | 15 ms |
| Hybrid + Cross-Encoder | **0.864** | **0.606** | 0.879 | 0.644 | 294 ms |

\* Retrieval latency excludes the query embedding (about 420 ms per question
through the Gemini API), which every strategy except BM25 also pays.

### nDCG@3 by question type

| Type (n) | Dense | BM25 | Hybrid | Hybrid + CE |
|---|---:|---:|---:|---:|
| factual (7) | 0.529 | 0.310 | **0.614** | **0.614** |
| keyword (9) | **0.771** | 0.346 | 0.646 | 0.693 |
| paraphrase (6) | **0.628** | 0.323 | 0.607 | 0.607 |

## Results at K=5

| Strategy | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Baseline | 0.902 | **0.545** | **0.894** | **0.601** |
| Dense | 0.902 | **0.545** | **0.894** | **0.601** |
| BM25 | 0.697 | 0.264 | 0.560 | 0.294 |
| Hybrid (RRF) | 0.924 | 0.464 | 0.823 | 0.521 |
| Hybrid + Cross-Encoder | **0.939** | **0.545** | 0.879 | 0.592 |

## Findings

1. **Dense retrieval is the strongest single strategy here.** It has the
   highest rank quality at both cut-offs (MRR 0.894, nDCG@3 0.655). Baseline
   and Dense score the same because the 0.8 L2 distance gate never fires on
   this corpus: relevant chunks sit around 0.4–0.5. The gate's value is
   rejecting off-topic questions, not reranking.
2. **BM25 is weak, and the cause is in our tokeniser rather than in BM25.**
   - BM25 often fills its top 3 with pages 23, 16 and 17: the BEIR examples
     table and the reference pages. These are long, term-dense chunks, so they
     match many query words.
   - The tokeniser is `text.lower().split()`. Punctuation stays attached
     (`bm25?` never matches `bm25,`), and stopwords such as "what", "the" and
     "does" count as matches.
   - On a 222-chunk corpus, IDF does little to suppress stopwords.
   - This is the main target for Week 6 retrieval optimisation.
3. **Hybrid trades rank quality for coverage.**
   - At K=5 hybrid reaches the highest Recall@5 of the first-stage
     strategies, 0.924 against 0.902 for dense: fusion surfaces relevant
     pages that dense alone misses (questions 8, 11, 16, 20, 21).
   - But BM25's noisy rankings push some good dense hits down (questions 2,
     4, 12, 13, 17), so MRR drops from 0.894 to 0.803 at K=3.
   - Hybrid is best on *factual* questions (nDCG@3 0.614 vs 0.529), and worst
     against dense on *keyword* questions. That is the opposite of the usual
     expectation, and it again points at the tokeniser: exact-term matching is
     BM25's whole job, and it is failing at it here.
4. **The cross-encoder recovers most of what fusion loses.** It reorders the
   8-chunk hybrid pool, raising MRR from 0.803 to 0.879 and nDCG@3 from 0.625
   to 0.644, and it gives the best Recall@5 overall (0.939). It costs about
   290 ms per query on CPU. That is still under the 420 ms query embedding,
   and far below LLM generation.
5. **Latency ranking:** pgvector search (2–4 ms) < BM25 over 222 chunks
   (13 ms; it rebuilds the in-memory index every query) ≈ hybrid (15 ms) ≪
   cross-encoder (~290 ms). The query embedding (~420 ms) dominates every
   dense-based strategy end to end.

## Methodology

**Dataset.** 22 questions about the BEIR paper
([backend/benchmarks/ragops_benchmark.json](../../backend/benchmarks/ragops_benchmark.json)):

- 7 factual: the question restates the paper's wording
- 9 keyword: the question hinges on exact names and numbers
- 6 paraphrased: the question shares little vocabulary with the text

Each question is labelled with the 1–3 PDF pages that answer it, and every
label was checked against the extracted page text.

**Relevance.** A retrieved chunk is relevant if it comes from a labelled page
of the labelled file. Labels are page-level because chunk IDs change on every
re-upload while page numbers do not. One consequence: a chunk from the right
page that doesn't contain the answer still counts as relevant, so the
metrics slightly favour strategies that stay on the right page.

**Metrics** (`backend/app/eval_retrieval.py`), computed on the top K chunks:

- **Precision@K**: relevant chunks in the top K, divided by K. A strategy that
  returns fewer than K chunks is not rewarded for it.
- **Recall@K**: labelled pages hit, divided by min(K, number of labelled
  pages). This is BEIR's capped recall.
- **MRR**: 1 / rank of the first relevant chunk.
- **nDCG@K**: binary gain, with an ideal DCG of K relevant chunks.

Every metric is the mean over all 22 questions.

**Strategies** (`backend/app/rag_pipelines.py`). Each was run exactly as it is
used in the product:

- **Baseline:** top-K by L2 distance.
- **Dense:** top-8 by L2, filtered at distance 0.8, keep top-K.
- **BM25:** Okapi over all chunks.
- **Hybrid:** RRF (k=60) of dense top-8 and BM25 top-8.
- **Hybrid + CE:** hybrid top-8 re-scored by `ms-marco-MiniLM-L-6-v2`.

**Controls.**
- **Same input for every strategy:** each question is embedded once, with
  the same model and `RETRIEVAL_QUERY` task type, and that vector is shared
  by every strategy, so they differ only in the retrieval algorithm.
- **One strategy at a time:** strategies run sequentially, so latencies don't
  contend with each other.
- **Repeatable:** runs are deterministic. Repeated runs gave identical
  metrics, with no sampling and no LLM involved.

**Known limitations.**
- **Small sample:** 22 questions over one document can rank the strategies,
  but it can't claim statistical significance. A single question moves a
  mean by about 0.045.
- **One labeller:** the question set was written by the same people who built
  the system.
- **No tuning yet:** all parameters are the defaults. Week 6 tunes them with a
  held-out split.
