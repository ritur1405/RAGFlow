"""
eval_rag.py — RAGFlow Ragas evaluation harness.

Benchmarks the RAG pipeline on three core metrics:
  - faithfulness:        is the answer actually supported by the retrieved context?
  - context_recall:      did retrieval surface all the info needed to answer?
  - context_precision:   are the relevant chunks ranked near the top?

NOTE ON RAGAS VERSIONS:
Ragas's public API has changed more than once through 2025-2026. This script
targets the current class-based metric API (Faithfulness(), LLMContextRecall(),
LLMContextPrecisionWithReference()). If your installed version instead exposes
the older functional metrics (ragas.metrics.faithfulness / context_recall /
context_precision as pre-built singletons), the import block below will fall
back to those automatically. Run `pip show ragas` and check
https://docs.ragas.io if you hit an ImportError this script doesn't catch.

RAGFlow uses Gemini everywhere else, so the Ragas judge LLM/embeddings are
wired to Gemini too (via langchain-google-genai) instead of defaulting to
OpenAI, which Ragas normally expects. Install with:
    pip install ragas datasets langchain-google-genai --break-system-packages
"""

import os
import sys

from datasets import Dataset

# ---------------------------------------------------------------------------
# Judge LLM / embeddings setup (Gemini, matching the rest of RAGFlow)
# ---------------------------------------------------------------------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    _api_key = os.environ.get("GEMINI_API_KEY")
    if not _api_key:
        print("ERROR: GEMINI_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=_api_key)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=_api_key)
    )
except ImportError as e:
    print(
        "Missing dependency for Gemini-backed evaluation. "
        "Run: pip install langchain-google-genai --break-system-packages\n"
        f"Original error: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Metric imports — current class-based API, falling back to older functional
# API if your installed ragas version predates it.
# ---------------------------------------------------------------------------
try:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextRecall,
        LLMContextPrecisionWithReference,
    )

    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    context_recall_metric = LLMContextRecall(llm=evaluator_llm)
    context_precision_metric = LLMContextPrecisionWithReference(llm=evaluator_llm)
    _USING_LEGACY_API = False

except ImportError:
    # Older ragas versions expose these as pre-built singleton metrics instead
    # of classes you instantiate yourself.
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_recall, context_precision

    faithfulness_metric = faithfulness
    context_recall_metric = context_recall
    context_precision_metric = context_precision
    _USING_LEGACY_API = True


# ---------------------------------------------------------------------------
# Dummy benchmark dataset
#
# Each row: a question actually answerable from the retrieved context,
# the RAG-generated answer, the chunks that were retrieved for it, and a
# human-written reference answer (ground_truth) used by context_recall to
# check whether the *right* information was retrieved at all.
#
# Replace this with real (question, answer, contexts, ground_truth) tuples
# pulled from your own /query/ endpoint + Supabase once you're ready to
# benchmark against real ingested documents rather than this smoke-test set.
# ---------------------------------------------------------------------------
DUMMY_EVAL_DATA = {
    "question": [
        "What parser does the AST-based chunking system use?",
        "How does the search engine combine vector search and BM25?",
        "What is the RRF smoothing constant used in the example?",
    ],
    "answer": [
        "The system uses Python's native ast module (via ast.NodeVisitor) to "
        "extract complete function and class definitions as chunks.",
        "It queries ChromaDB (dense) and BM25 (sparse) in parallel and merges "
        "their ranked results using Reciprocal Rank Fusion (RRF).",
        "The smoothing constant k is set to 60 in the example RRF implementation.",
    ],
    "contexts": [
        [
            "Instead of splitting text arbitrarily by character length, the "
            "parser traverses the source code using Python's built-in ast "
            "module. Using ast.NodeVisitor, it targets FunctionDef and "
            "ClassDef nodes, extracting standalone, syntactically valid code blocks."
        ],
        [
            "To achieve high precision, search_engine.py queries both "
            "retrievers in parallel and merges their rankings using "
            "Reciprocal Rank Fusion (RRF)."
        ],
        [
            "Where k is a smoothing constant set to 60.",
            "def reciprocal_rank_fusion(dense_results, sparse_results, k=60):",
        ],
    ],
    "ground_truth": [
        "Python's built-in ast module, using ast.NodeVisitor to find "
        "FunctionDef and ClassDef nodes.",
        "By running dense (ChromaDB) and sparse (BM25) search in parallel, "
        "then fusing the two rankings with Reciprocal Rank Fusion.",
        "60.",
    ],
}


def build_dataset(data: dict = DUMMY_EVAL_DATA) -> Dataset:
    """Converts a plain dict of eval rows into a HuggingFace Dataset, the
    format ragas.evaluate() expects."""
    return Dataset.from_dict(data)


def run_evaluation(dataset: Dataset) -> dict:
    """Runs faithfulness, context_recall, and context_precision over the
    given dataset and returns the aggregate scores as a plain dict."""
    metrics = [faithfulness_metric, context_recall_metric, context_precision_metric]

    if _USING_LEGACY_API:
        result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)
    else:
        result = evaluate(dataset=dataset, metrics=metrics)

    return result


def main():
    print("Building evaluation dataset (dummy benchmark rows)...")
    dataset = build_dataset()

    print(f"Running Ragas evaluation ({'legacy' if _USING_LEGACY_API else 'current'} API)...")
    result = run_evaluation(dataset)

    print("\n=== Ragas Evaluation Results ===")
    print(result)

    try:
        df = result.to_pandas()
        print("\n--- Per-row breakdown ---")
        print(df.to_string(index=False))
    except Exception:
        pass  # Not all ragas versions expose to_pandas() the same way; the summary above still printed.


if __name__ == "__main__":
    main()