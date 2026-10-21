# Benchmark dataset

`ragops_benchmark.json` is the shared evaluation set every RAG configuration is
measured against. It holds 22 questions about the BEIR paper. Each question
has a reference answer and the PDF pages that contain the answer:

| Field | Used for |
|---|---|
| `question` | Sent to each RAG configuration |
| `ground_truth` | Stored with the result, for reading the judge's verdicts |
| `source_file`, `relevant_pages` | Retrieval metrics. A retrieved chunk counts as relevant when it comes from one of these pages of this file. |
| `type` | `factual`, `keyword` or `paraphrase`, for slicing the results |

Labels are page numbers rather than chunk IDs, because chunk IDs change every
time the PDF is re-uploaded. Page numbers are 1-based, matching
`process_pdf` in `app/services.py`.

## Source document

`beir-2104.08663v4.pdf` is included unmodified:

> Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna
> Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
> Information Retrieval Models.* arXiv:2104.08663v4, 2021.
> https://arxiv.org/abs/2104.08663v4

It is distributed under the
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
licence, as stated on its arXiv page. The question set is derived from it and
is shared under the same licence.

The paper was chosen because its subject matches what this project measures:
it compares BM25, dense retrievers and cross-encoder re-ranking, which makes
the benchmark questions meaningful to discuss as well as to answer.

SHA-256: `1925755aeb627123d1a63c66f02e5aa2432f7e92751a02bea7c8c3b6bbb6ec33`

## Running it

Upload the PDF on the Documents page (or `POST /documents/upload/`), then run
`scripts/run_benchmark.py`. It posts this file to `/api/eval/run` and writes
the results to `docs/benchmarks/`.
