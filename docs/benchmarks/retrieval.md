# Retrieval benchmark: Dense vs BM25 vs Hybrid

This is the plan's Week 4 deliverable. It compares all five retrieval strategies
on the shared, page-labelled benchmark, with the Week 5 retrieval metrics.

**Run date:** 2026-10-22 (Week 6; first run 2026-10-21). **Corpus:** the BEIR paper (24 pages, 222 cleaned
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

K=3 is the number of chunks the generator receives. Week 6 numbers, after the
BM25 tokeniser fix ([tuning.md](tuning.md)).

| Strategy | Recall@3 | Precision@3 | MRR | nDCG@3 | Retrieval latency* |
|---|---:|---:|---:|---:|---:|
| Baseline (dense, no gate) | **0.864** | 0.606 | **0.894** | **0.655** | 6 ms |
| Dense (distance gate 0.8) | **0.864** | 0.606 | **0.894** | **0.655** | 3 ms |
| BM25 | 0.765 | 0.530 | 0.773 | 0.564 | 20 ms |
| Hybrid (Dense + BM25, RRF) | 0.788 | **0.621** | 0.803 | 0.649 | 22 ms |
| Hybrid + Cross-Encoder | 0.841 | 0.576 | 0.879 | 0.623 | 0.3–1.3 s |

\* Retrieval latency excludes the query embedding (about 0.4–0.5 s per
question through the Gemini API), which every strategy except BM25 also pays.
The cross-encoder runs on CPU, so its time depends on the host: 294 ms on an
idle machine in Week 4, 0.56–1.34 s in repeated Week 6 runs on a laptop.

### nDCG@3 by question type

| Type (n) | Dense | BM25 | Hybrid | Hybrid + CE |
|---|---:|---:|---:|---:|
| factual (7) | 0.529 | 0.638 | 0.605 | **0.647** |
| keyword (9) | **0.771** | 0.601 | 0.731 | 0.641 |
| paraphrase (6) | **0.628** | 0.422 | 0.578 | 0.568 |

## Results at K=5

| Strategy | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Baseline | 0.902 | **0.545** | **0.894** | **0.601** |
| Dense | 0.902 | **0.545** | **0.894** | **0.601** |
| BM25 | 0.803 | 0.409 | 0.773 | 0.471 |
| Hybrid (RRF) | 0.939 | 0.518 | 0.835 | 0.571 |
| Hybrid + Cross-Encoder | **0.955** | 0.536 | 0.879 | 0.582 |

## Findings

1. **Dense retrieval is the strongest single strategy here.** It has the
   highest rank quality at both cut-offs (MRR 0.894, nDCG@3 0.655). Baseline
   and Dense score the same because the 0.8 L2 distance gate never fires on
   this corpus: relevant chunks sit around 0.4–0.5. The gate's value is
   rejecting off-topic questions, not reranking.
2. **BM25 was weak because of our tokeniser, and fixing it made BM25 a real
   contender.**
   - Week 4 used `text.lower().split()`: punctuation stayed attached
     (`bm25?` never matched `bm25,`), and stopwords such as "what" and "the"
     matched almost every chunk, so long, term-dense pages (the examples
     table, the references) filled its top 3.
   - With lowercase words, punctuation stripped and stopwords dropped, BM25's
     nDCG@3 rose from 0.328 to 0.564 and MRR from 0.508 to 0.773.
   - BM25 is now the best single retriever on *factual* questions (0.638 vs
     dense 0.529), where the question reuses the paper's wording. It is still
     weakest on *paraphrase* questions, as expected.
3. **Hybrid trades rank quality for coverage.**
   - At K=5 hybrid reaches Recall@5 0.939 against 0.902 for dense: fusion
     surfaces relevant pages that dense alone misses.
   - At K=3 BM25's hits displace some good dense hits, so Recall@3 drops from
     0.864 to 0.788 and MRR from 0.894 to 0.803. Hybrid has the best
     Precision@3 (0.621).
4. **The cross-encoder restores ranking.** Re-scoring the 8-chunk hybrid pool
   raises MRR from 0.803 to 0.879 and gives the best Recall@5 overall (0.955).
   It is the best strategy on factual questions (nDCG@3 0.647). It is the only
   strategy whose cost is noticeable: hundreds of milliseconds per query on
   CPU, comparable to the query embedding and far below LLM generation.
5. **Latency ranking:** pgvector search (3–6 ms) < BM25 over 222 chunks
   (20 ms; it rebuilds the in-memory index every query) ≈ hybrid (22 ms) ≪
   cross-encoder (0.3–1.3 s). The query embedding (~0.45 s) dominates every
   dense-based strategy apart from the reranker.

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
- **BM25:** Okapi over all chunks, tokenised as lowercase words without
  punctuation or stopwords.
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
- **Tuning:** the candidate pool and RRF constant were grid-searched in
  Week 6 with a held-out split, and the defaults held; see
  [tuning.md](tuning.md).
