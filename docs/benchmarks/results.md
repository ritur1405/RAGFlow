# Benchmark results: beir-paper-v1

**Coverage:** 22 of 22 questions scored for every configuration. Last updated 2026-10-21T20:33:40+00:00.

Generation model `gemini-3.5-flash-lite`; judge `gemma-4-31b-it` (one call scores all four metrics, 0-1). Retrieval metrics at K=3 against page labels. Latency is the RAG pipeline (retrieval + generation), excluding judging. Best value per column in bold.

## Generation quality and latency

| Configuration | n | Faithfulness | Answer relevance | Context relevance | Citation accuracy | Overall (judge) | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline RAG | 22 | 0.955 | **0.818** | 0.605 | 0.955 | 0.857 | 1361.682 ms |
| Dense RAG | 22 | 0.955 | 0.773 | 0.610 | 0.955 | 0.849 | **1300.297** ms |
| Hybrid RAG (Dense + BM25) | 22 | **1.000** | 0.659 | 0.602 | 0.991 | 0.850 | 1383.450 ms |
| Hybrid + Cross-Encoder Reranker | 22 | **1.000** | 0.773 | **0.636** | **1.000** | **0.882** | 2487.205 ms |

## Retrieval quality (the same questions, K=3)

| Configuration | Recall@3 | Precision@3 | MRR | nDCG@3 |
|---|---:|---:|---:|---:|
| Baseline RAG | **0.864** | **0.606** | **0.894** | **0.655** |
| Dense RAG | **0.864** | **0.606** | **0.894** | **0.655** |
| Hybrid RAG (Dense + BM25) | 0.848 | **0.606** | 0.803 | 0.625 |
| Hybrid + Cross-Encoder Reranker | **0.864** | **0.606** | 0.879 | 0.644 |

## Overall judge score by question type

| Question type | Baseline RAG | Dense RAG | Hybrid RAG (Dense + BM25) | Hybrid + Cross-Encoder Reranker |
|---|---:|---:|---:|---:|
| factual | 0.782 (n=7) | 0.782 (n=7) | 0.846 (n=7) | 0.871 (n=7) |
| keyword | 0.868 (n=9) | 0.868 (n=9) | 0.841 (n=9) | 0.867 (n=9) |
| paraphrase | 0.929 (n=6) | 0.899 (n=6) | 0.868 (n=6) | 0.917 (n=6) |
