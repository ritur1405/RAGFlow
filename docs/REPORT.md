# RAGOps: project report

A summary of what was built against the
[project plan](RAGOps_2_Person_Project_Plan.pdf), how the pieces fit together,
how to demo it, and the questions the plan expects us to answer in interviews.
For setup and API details see the [README](../README.md).

## 1. Deliverables against the plan

| Week | Plan deliverable | Delivered |
|---|---|---|
| 1 | Architecture, repository skeleton, DB connection | FastAPI + SQLAlchemy + pgvector, CRUD endpoints, CORS, exception handling (#2) |
| 2 | PDF → chunks → embeddings → pgvector; upload UI | Extract → clean → chunk → embed pipeline (#3, #4, cleaning wired into upload in the finalization PR); Documents page (#18) |
| 3 | Upload → retrieve → generate → answer + citations; query page | Dense RAG with distance gate and `[Source N]` citations (#5, #11); Ask page (#18) |
| 4 | Dense vs BM25 vs Hybrid comparison; experiment API, config storage, comparison UI | `retrieve_*` strategies incl. BM25 and RRF hybrid, `/api/experiments`, saved configs, Experiments page (#14); cross-encoder reranking (#15) |
| 5 | Recall@K, Precision@K, MRR, NDCG; faithfulness, relevance, citation accuracy; evaluation dashboard | Retrieval metrics on page labels and a 22-question labelled benchmark (#16); LLM-judge generation metrics and dashboard (#12, extended in #16) |
| 6 | Final benchmarks, graphs, Docker, deployment, README, demo flow | Docker Compose stack (#12), CI (#13), rate-limit handling (#17), benchmark runner with charts, README, deployment guide and this report (finalization PR) |

Things the plan said **not** to build were not built:

- no microservices
- no Kubernetes
- no fine-tuning
- no complex auth (one shared-key header guards the endpoint that spends credits)
- no orchestration framework

## 2. Architecture in one paragraph

A React app talks to one FastAPI service, which stores everything in one
PostgreSQL database with pgvector.

- **Ingestion:** a PDF becomes cleaned, page-tagged 500-character chunks with
  768-d Gemini embeddings.
- **Retrieval:** five interchangeable strategies, each of which returns ranked
  `(chunk, score)` pairs:
  - **Baseline**: dense top-K
  - **Dense**: top-K behind a distance gate
  - **BM25**
  - **Hybrid**: RRF fusion of dense and BM25
  - **Hybrid + cross-encoder rerank**
- **Generation:** an answer from only those chunks, with citations.
- **Evaluation:** each configuration runs over a shared, page-labelled question
  set. Retrieval is scored with IR metrics, and the answer is scored by an LLM
  judge whose reasoning is stored.
- **Two dashboards:** the Experiments page compares retrievers on one query, and
  the Evaluation page compares whole configurations on the benchmark.

## 3. Results

The benchmark runs the 22 BEIR-paper questions through Baseline, Dense, Hybrid
and Hybrid + Cross-Encoder:

```bash
python scripts/run_benchmark.py
```

It writes tables to [benchmarks/results.md](benchmarks/results.md) and charts
to `docs/benchmarks/`, which the README's results section links to.

**What to look for when reading them:**

- **By question type.** Keyword questions (exact names and numbers, such as
  "k=0.9, b=0.4" or "Hole@10") should favour configurations that include BM25.
  Paraphrased questions should favour dense retrieval. Hybrid should be the most
  consistent across all three types, which is the reason to fuse.
- **Rank quality vs coverage.** The cross-encoder can reorder chunks only within
  the hybrid pool, so it can raise MRR and nDCG without changing which pages
  are reachable. Recall@3 can still move, because reranking decides which three
  of the eight candidates survive.
- **Generation follows retrieval.** Context relevance and faithfulness should
  track retrieval quality.
- **Latency.**
  - Dense and hybrid latency is dominated by the query embedding call.
  - BM25 alone is about 20–40 ms.
  - The cross-encoder adds tens of milliseconds on CPU. It is far cheaper here
    than in the BEIR paper, because it scores 8 candidates instead of 100.

## 4. Demo flow (about 5 minutes)

1. **Home → Documents.** Upload `backend/benchmarks/beir-2104.08663v4.pdf`, then
   point out the 222 chunks across 24 pages. Mention the cleaning step and why
   chunks are split per page: every chunk keeps an exact page number, which
   citations and the evaluation labels rely on.
2. **Ask.** "What k and b parameters were used for Anserini BM25?" The answer is
   k=0.9 and b=0.4. Click the `[1]` citation to jump to the page-5 chunk. Then
   ask about something the paper doesn't contain, to show the fixed "cannot find"
   answer instead of a guess.
3. **Experiments.** Compare two queries:
   - "What k and b parameters were used for Anserini BM25?": every strategy
     ranks page 5 first.
   - "Why do some dense retrievers favour short documents while others favour
     long ones?": dense finds the labelled page 21, while BM25 drifts to pages
     3, 4 and 7. Hybrid keeps page 21 and adds BM25's page 7, which is the other
     labelled page.

   Point at the agreement matrix and the latency chart.
4. **Evaluation.** Open a completed benchmark run:
   - the comparison table, with retrieval and generation side by side
   - the two charts
   - one question expanded, showing the page hits and the judge's reasoning for
     each score
5. Close on the quality-vs-latency chart in the README.

## 5. Interview questions (plan §9)

**Why hybrid retrieval?** BM25 and dense retrieval fail differently.

- BM25 matches exact tokens: rare names, numbers and parameter values. It is
  blind to paraphrase, which is the lexical gap the BEIR paper describes.
- Dense embeddings match meaning, but can rank a near-synonym above the exact
  term, and can miss rare identifiers.

Reciprocal Rank Fusion combines the two using only ranks. That sidesteps
calibrating L2 distances against BM25 scores, which live on unrelated scales.
A chunk ranked well by both lists wins.

**When does reranking help?** When the right chunk is in the candidate pool but
not at the top. A cross-encoder reads the question and the chunk together, so
it catches relevance that independent embeddings miss. It cannot recover a
chunk that first-stage retrieval never found, and it costs one model pass per
candidate. That is why it re-scores the 8-chunk hybrid pool and not the whole
corpus. The BEIR paper's BM25+CE was its best zero-shot system, and also its
slowest: 6.1 s per query on CPU when re-ranking 100 hits.

**Recall@K vs MRR?**

- **Recall@K**: did the answer-bearing pages appear in the top K at all? It
  measures coverage, which bounds what the generator can possibly say.
- **MRR**: how high is the *first* relevant chunk? It measures rank quality.
- **nDCG**: credits every relevant chunk, discounted by position.

We use capped recall (BEIR Appendix G), so a question with more relevant pages
than K can still reach 1.0.

**How was faithfulness evaluated?** With an LLM judge (`gemma-4-31b-it`) at
temperature 0. It is a different model family from the generator
(`gemini-3.5-flash-lite`), so the judge isn't grading its own model's style.
One call scores all four metrics, each against its own rubric.

- **Inputs:** the question, the exact retrieved context and the answer.
- **Task:** break the answer into claims, check each claim against the context,
  and score the grounded fraction from 0 to 1, with written reasoning.
- **Guardrails:** the judge never sees the reference answer when scoring
  faithfulness, and a judge failure fails the run instead of recording a
  default score. Citation accuracy adds a deterministic check: citing a chunk
  that was never retrieved caps the score at 0.2.

**What latency/quality trade-off appeared?** The answer comes from the
quality-vs-latency chart. Structurally:

- BM25 is roughly 20× faster than dense, because it needs no embedding call.
- Hybrid costs about the same as dense, since the embedding call dominates.
- The cross-encoder adds little on a small pool.
- In this setup, generation dwarfs every retrieval cost.

**Which configuration performed best and why?** See
[benchmarks/results.md](benchmarks/results.md). The question-type breakdown
explains *why*: compare each configuration's nDCG on keyword versus
paraphrased questions.

## 6. Engineering decisions worth mentioning

- **Page-level relevance labels.** Labels survive re-chunking and re-uploading,
  unlike chunk IDs, and they were verified against the extracted text.
- **Fail loudly.** A failed reranker, pipeline or judge marks the run failed. It
  never falls back to a different configuration under the same label, and
  never fills in a score.
- **Quota-aware retries.** Per-minute 429s wait for the server's suggested
  delay. Daily quotas fail immediately with a clear 503. This was found by
  running the system end to end on a free-tier key.
- **Local, deterministic reranker.** ONNX through fastembed, baked into the
  image, loaded at startup so the first query isn't penalised.
- **CI from week 6.** CI was added after branch merges had silently dropped a
  week of work and left `main`-bound branches failing at import. It caught a
  missing test dependency on its first run.

## 7. Resume bullets

- Built a full-stack RAG experimentation platform (React, FastAPI, PostgreSQL +
  pgvector, Docker) that ingests PDFs and compares dense, BM25, hybrid (RRF) and
  cross-encoder-reranked retrieval on a shared labelled benchmark.
- Implemented retrieval evaluation (Recall@K, Precision@K, MRR, nDCG) on
  page-level relevance labels, plus LLM-as-judge generation metrics
  (faithfulness, answer/context relevance, citation accuracy) with stored
  reasoning and a deterministic citation guardrail.
- Designed retrieval as interchangeable ranked retrievers behind one registry,
  powering both an interactive side-by-side comparison UI and an end-to-end
  benchmark runner that produces tables and quality-vs-latency charts.
- Hardened the system for production use: quota-aware retry handling, honest
  error mapping, startup model warm-up, idempotent schema upgrades, about 200
  hermetic tests and GitHub Actions CI.
