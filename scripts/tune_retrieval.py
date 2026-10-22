"""Grid-searches the hybrid retrieval settings on the shared benchmark.

    docker exec -i ragflow_backend python - < scripts/tune_retrieval.py

Runs inside the backend container, which already has the database, the
benchmark and the app code. Costs one query embedding per question (22) and
no generation calls, so it stays well inside the Gemini free tier.

Method: settings are chosen on the odd-numbered questions and confirmed on the
even-numbered ones. A default is worth changing only if the winner also beats
it on the held-out half; otherwise the gain is noise from 11 questions.
Prints Markdown tables for docs/benchmarks/tuning.md.
"""

import app.rag_pipelines as rp
from app.benchmark import load_benchmark
from app.retrieval_benchmark import aggregate, query_embeddings, run_strategy

K = 3
POOLS = (5, 8, 10, 15, 20)
RRF_KS = (10, 30, 60, 100)

questions = load_benchmark()["questions"]
tune, confirm = questions[0::2], questions[1::2]  # questions 1,3,5... and 2,4,6...
vectors, _ = query_embeddings([q["question"] for q in questions])
defaults = (rp.RERANK_CANDIDATE_POOL, rp.RRF_K)


def scores(strategy, qs):
    return aggregate(run_strategy(strategy, qs, K, vectors))["overall"]


def fmt(x):
    return f"{x:.3f}"


# 1. The tokeniser fix, on all 22 questions at the default settings.
print(f"## Tokeniser (all {len(questions)} questions, K={K})\n")
print("| Strategy | Tokeniser | Recall@3 | MRR | nDCG@3 |\n|---|---|---:|---:|---:|")
new_tokenize = rp._tokenize
for strategy in ("bm25", "hybrid", "hybrid_reranker"):
    for name, tokenize in (("whitespace split (old)", lambda t: t.lower().split()), ("words, no stopwords (new)", new_tokenize)):
        rp._tokenize = tokenize
        s = scores(strategy, questions)
        print(f"| {strategy} | {name} | {fmt(s['recall_at_k'])} | {fmt(s['reciprocal_rank'])} | {fmt(s['ndcg_at_k'])} |")
rp._tokenize = new_tokenize

# 2. Candidate pool x RRF constant, nDCG@3 on each half.
for strategy in ("hybrid", "hybrid_reranker"):
    print(f"\n## {strategy}: candidate pool x RRF k (nDCG@3, tune / confirm)\n")
    print("| Pool | " + " | ".join(f"k={k}" for k in RRF_KS) + " |\n|---:|" + "---:|" * len(RRF_KS))
    results = {}
    for pool in POOLS:
        cells = []
        for k in RRF_KS:
            rp.RERANK_CANDIDATE_POOL, rp.RRF_K = pool, k
            results[(pool, k)] = (scores(strategy, tune)["ndcg_at_k"], scores(strategy, confirm)["ndcg_at_k"])
            mark = " (default)" if (pool, k) == defaults else ""
            cells.append(f"{fmt(results[(pool, k)][0])} / {fmt(results[(pool, k)][1])}{mark}")
        print(f"| {pool} | " + " | ".join(cells) + " |")
    # Ties go to the default, so a setting is only changed when it actually wins.
    best = max(results, key=lambda key: (results[key][0], key == defaults))
    beats = results[best][0] > results[defaults][0] and results[best][1] > results[defaults][1]
    print(f"\nBest on tune: pool={best[0]}, k={best[1]} "
          f"({fmt(results[best][0])} vs default {fmt(results[defaults][0])}); "
          f"confirm {fmt(results[best][1])} vs {fmt(results[defaults][1])}: "
          f"{'change the default' if beats else 'keep the default'}.")
rp.RERANK_CANDIDATE_POOL, rp.RRF_K = defaults
