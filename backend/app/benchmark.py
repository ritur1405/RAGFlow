"""The shared benchmark dataset.

One page-labelled question set (backend/benchmarks/ragops_benchmark.json) is
used by both the end-to-end evaluation (/api/eval) and the retrieval-only
benchmark (/api/experiments/benchmark), as the plan requires: one shared
evaluation dataset so experiments are comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "ragops_benchmark.json"


def load_benchmark() -> dict:
    """Returns the dataset as stored. Raises FileNotFoundError if it is missing."""
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
