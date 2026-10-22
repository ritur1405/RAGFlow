# Benchmark results: beir-paper-v1

**Coverage:** 22 of 22 questions scored for every configuration. Last updated 2026-10-22T19:07:09+00:00.

Generation model `gemini-3.5-flash-lite`; judge `gemma-4-31b-it` (one call scores all four metrics, 0-1). Retrieval metrics at K=3 against page labels. Latency is the RAG pipeline (retrieval + generation), excluding judging. Best value per column in bold.

## Generation quality and latency

| Configuration | n | Faithfulness | Answer relevance | Context relevance | Citation accuracy | Overall (judge) | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline RAG | 22 | 0.955 | 0.773 | 0.597 | 0.955 | 0.847 | **5718.977** ms |
| Dense RAG | 22 | 0.955 | **0.809** | **0.605** | 0.955 | 0.856 | 7814.549 ms |
| Hybrid RAG (Dense + BM25) | 22 | **1.000** | 0.682 | 0.594 | 0.991 | 0.853 | 14782.652 ms |
| Hybrid + Cross-Encoder Reranker | 22 | **1.000** | 0.727 | 0.603 | **1.000** | **0.866** | 5791.340 ms |

## Retrieval quality (the same questions, K=3)

| Configuration | Recall@3 | Precision@3 | MRR | nDCG@3 |
|---|---:|---:|---:|---:|
| Baseline RAG | **0.864** | 0.606 | **0.894** | **0.655** |
| Dense RAG | **0.864** | 0.606 | **0.894** | **0.655** |
| Hybrid RAG (Dense + BM25) | 0.788 | **0.621** | 0.803 | 0.649 |
| Hybrid + Cross-Encoder Reranker | 0.841 | 0.576 | 0.879 | 0.623 |

## Overall judge score by question type

| Question type | Baseline RAG | Dense RAG | Hybrid RAG (Dense + BM25) | Hybrid + Cross-Encoder Reranker |
|---|---:|---:|---:|---:|
| factual | 0.773 (n=7) | 0.782 (n=7) | 0.840 (n=7) | 0.871 (n=7) |
| keyword | 0.868 (n=9) | 0.868 (n=9) | 0.852 (n=9) | 0.848 (n=9) |
| paraphrase | 0.900 (n=6) | 0.922 (n=6) | 0.869 (n=6) | 0.887 (n=6) |
