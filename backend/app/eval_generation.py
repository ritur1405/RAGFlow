"""LLM-as-a-judge generation evaluation.

Scores a single (query, answer, context) triple on four generation-quality
metrics:

    1. Faithfulness       - is the answer strictly grounded in the retrieved context?
    2. Answer Relevance   - does the answer actually address the user's query?
    3. Context Relevance  - how much of the retrieved context is useful vs. noise?
    4. Citation Accuracy  - do the cited chunk IDs actually support the claims made?

Every score is normalized to [0.0, 1.0] and carries short step-by-step
reasoning so a result can be audited in the dashboard rather than taken on
trust.

One judge call per answer. All four metrics are scored by a single prompt
with a rubric per metric, so judging a benchmark answer costs one request
instead of four. On the Gemini free tier, where daily requests are capped per
model, that is the difference between the benchmark fitting in one day or
not. A deterministic check still caps citation accuracy when the answer cites
a chunk that was never retrieved, regardless of the judge's score.

The judge runs on its own model (GEMINI_JUDGE_MODEL, default gemma-4-31b-it):
a different model family from the answer generator, so the judge is not
grading its own model's style, and a separate per-model free-tier quota.

Failure policy: this module never invents a score. If the judge cannot run --
no API key, SDK missing, or the call fails after retries -- it raises. A run
that could not be judged is recorded as failed, not as a run that happened to
score well. See JudgeUnavailableError and JudgeCallError.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from google.genai import errors as genai_errors
from google.genai import types

from app.config import JUDGE_MODEL
from app.quota import is_daily_quota

# Reuse the single client initialised in services.py rather than creating a
# second one, matching how app/llm.py sources its client.
from app.services import client as _client

logger = logging.getLogger("ragops.eval_generation")

# Kept under this name for callers and logs that already refer to it.
GEMINI_JUDGE_MODEL = JUDGE_MODEL

# Retries for malformed or failed judge responses. Rate limits and transient
# server errors are already retried inside the shared client (app/services.py).
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.5

METRICS = ("faithfulness", "answer_relevance", "context_relevance", "citation_accuracy")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class JudgeError(RuntimeError):
    """Base class for judge failures."""


class JudgeUnavailableError(JudgeError):
    """The judge is not configured and therefore cannot score anything."""


class JudgeCallError(JudgeError):
    """The judge was configured but the call failed after every retry."""


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ContextChunk:
    """A single retrieved chunk passed into the generation step.

    ``chunk_id`` is a string so it compares directly against the citation
    markers extracted from the generated answer.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeScore:
    """Normalized output of one judge metric."""

    metric: str
    score: float                  # normalized 0.0 - 1.0
    reasoning: str                # short step-by-step justification
    raw_response: Optional[dict] = None
    latency_ms: Optional[float] = None


@dataclass
class GenerationEvalResult:
    """Aggregate of all four metrics for a single (query, answer, context) triple."""

    faithfulness: JudgeScore
    answer_relevance: JudgeScore
    context_relevance: JudgeScore
    citation_accuracy: JudgeScore
    overall_score: float
    total_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "faithfulness": _score_to_dict(self.faithfulness),
            "answer_relevance": _score_to_dict(self.answer_relevance),
            "context_relevance": _score_to_dict(self.context_relevance),
            "citation_accuracy": _score_to_dict(self.citation_accuracy),
            "overall_score": round(self.overall_score, 4),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


def _score_to_dict(s: JudgeScore) -> dict:
    return {
        "metric": s.metric,
        "score": round(s.score, 4) if s.score is not None else None,
        "reasoning": s.reasoning,
        "latency_ms": round(s.latency_ms, 2) if s.latency_ms else None,
    }


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def _extract_json(raw_text: str) -> dict:
    """Parses the judge's JSON, tolerating code fences or prose around it.

    Models without a JSON response mode (e.g. Gemma) sometimes wrap the object
    in ```json fences or add a sentence before it, so the outermost {...} is
    parsed when the whole text is not valid JSON.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def judge_available() -> bool:
    """True when a judge client is configured. Callers may pre-flight with this."""
    return _client is not None


def _format_context(context_chunks: list[ContextChunk]) -> str:
    if not context_chunks:
        return "(no context chunks were retrieved)"
    return "\n\n".join(
        f"[chunk_id: {c.chunk_id}]\n{c.text.strip()}" for c in context_chunks
    )


def extract_citation_ids(answer: str) -> list[str]:
    """Best-effort extraction of citation markers like [12], [chunk_3], (chunk-7).

    Returns ids in first-seen order with duplicates removed.
    """
    pattern = r"\[([A-Za-z0-9_\-]+)\]|\(([A-Za-z0-9_\-]*chunk[A-Za-z0-9_\-]*)\)"
    matches = re.findall(pattern, answer, flags=re.IGNORECASE)

    seen: set[str] = set()
    unique_ids: list[str] = []
    for a, b in matches:
        val = a or b
        if val and val not in seen:
            seen.add(val)
            unique_ids.append(val)
    return unique_ids


# --------------------------------------------------------------------------- #
# The judge prompt: four rubrics, one call
# --------------------------------------------------------------------------- #

_JUDGE_INSTRUCTIONS = """You are a strict, impartial evaluator for a Retrieval-Augmented \
Generation (RAG) system. Score one generated answer on four independent metrics, each from \
0.0 to 1.0. Judge each metric on its own rubric; do not let one metric influence another.

1. faithfulness: is every factual claim in the answer supported by the RETRIEVED CONTEXT?
   Break the answer into claims and check each against the context. Outside knowledge does
   not count as support, even if true.
   1.0 all claims supported | 0.5-0.9 minor unsupported details | 0.1-0.4 several unsupported
   claims | 0.0 substantially fabricated or contradicts the context.
   A refusal that states the context does not contain the answer makes no claims: score 1.0.

2. answer_relevance: does the answer directly and completely address the QUERY?
   Ignore correctness here; judge only whether it answers the question that was asked.
   1.0 direct and complete | 0.5-0.9 partial, indirect or padded | 0.1-0.4 tangential |
   0.0 does not address the query (including a refusal).

3. context_relevance: how much of the RETRIEVED CONTEXT is useful for answering the QUERY?
   Judge each chunk as useful or noise.
   1.0 nearly all useful | 0.5-0.9 mostly useful | 0.1-0.4 mostly noise | 0.0 none useful.

4. citation_accuracy: the answer cites chunks inline as [chunk_id]. Does each cited chunk
   support the claim it is attached to, and is every claim that needs support cited?
   1.0 all citations correct and complete | 0.5-0.9 minor mismatches or omissions |
   0.1-0.4 many wrong or missing | 0.0 absent, fabricated or systematically wrong.
   A refusal with no claims and no citations: score 1.0.

Respond with ONLY a JSON object of exactly this shape, and nothing else:
{"faithfulness": {"score": <float>, "reasoning": "<2-4 short numbered steps>"},
 "answer_relevance": {"score": <float>, "reasoning": "<...>"},
 "context_relevance": {"score": <float>, "reasoning": "<...>"},
 "citation_accuracy": {"score": <float>, "reasoning": "<...>"}}"""

_JUDGE_TEMPLATE = """QUERY:
{query}

RETRIEVED CONTEXT (with chunk IDs):
{context}

GENERATED ANSWER:
{answer}

CITED CHUNK IDS EXTRACTED FROM THE ANSWER:
{cited_ids}"""


def _supports_system_and_json(model: str) -> bool:
    """Gemma models on the Gemini API reject system instructions and JSON mode."""
    return not model.lower().startswith("gemma")


def _judge_request(prompt: str) -> tuple[str, types.GenerateContentConfig]:
    if _supports_system_and_json(GEMINI_JUDGE_MODEL):
        return prompt, types.GenerateContentConfig(
            system_instruction=_JUDGE_INSTRUCTIONS,
            temperature=0.0,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )
    # Instructions travel in the prompt itself; _extract_json copes with any
    # fences or prose the model adds around the JSON.
    return f"{_JUDGE_INSTRUCTIONS}\n\n{prompt}", types.GenerateContentConfig(
        temperature=0.0, max_output_tokens=2048
    )


def _parse_scores(parsed: dict) -> dict[str, tuple[float, str]]:
    """Validates that all four metrics are present. Raises ValueError otherwise."""
    scores = {}
    for metric in METRICS:
        entry = parsed.get(metric)
        if not isinstance(entry, dict) or "score" not in entry:
            raise ValueError(f"judge response is missing metric '{metric}'")
        reasoning = str(entry.get("reasoning", "")).strip() or "No reasoning provided by judge."
        scores[metric] = (_clamp01(entry["score"]), reasoning)
    return scores


async def _call_judge(prompt: str) -> tuple[dict[str, tuple[float, str]], dict, float]:
    """Runs the single judge call, retrying malformed or failed responses.

    Raises rather than returning a placeholder score: a metric that could not
    be measured must not look like a metric that scored well.
    """
    if _client is None:
        raise JudgeUnavailableError(
            "Gemini judge is not configured (no client). Set GEMINI_API_KEY to "
            "run evaluations."
        )

    contents, config = _judge_request(prompt)
    start = time.perf_counter()
    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await asyncio.to_thread(
                _client.models.generate_content,
                model=GEMINI_JUDGE_MODEL,
                contents=contents,
                config=config,
            )
            parsed = _extract_json(response.text or "")
            scores = _parse_scores(parsed)
            return scores, parsed, (time.perf_counter() - start) * 1000
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
            if isinstance(exc, genai_errors.APIError) and is_daily_quota(exc):
                # Retrying cannot succeed until the quota resets.
                raise JudgeUnavailableError(
                    f"Gemini daily quota exhausted for the judge model "
                    f"({GEMINI_JUDGE_MODEL}): {exc.message}"
                ) from exc
            if isinstance(exc, genai_errors.APIError):
                # The shared client already retried 429s and 5xx; retrying here
                # would multiply those requests against the same quota.
                raise JudgeCallError(f"Judge call failed: {exc}") from exc
            last_error = exc
            logger.warning(
                "Judge call failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc
            )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    logger.error("Judge call failed after %d attempts", _MAX_RETRIES, exc_info=last_error)
    raise JudgeCallError(
        f"Judge failed after {_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

_METRIC_WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevance": 0.2,
    "context_relevance": 0.2,
    "citation_accuracy": 0.25,
}


async def run_generation_eval(
    query: str,
    answer: str,
    context_chunks: list[ContextChunk],
    cited_chunk_ids: Optional[list[str]] = None,
) -> GenerationEvalResult:
    """Scores one generated answer on all four metrics with a single judge call.

    Raises JudgeUnavailableError / JudgeCallError if the judge cannot produce
    all four scores. Partial results are not returned.
    """
    if cited_chunk_ids is None:
        cited_chunk_ids = extract_citation_ids(answer)

    prompt = _JUDGE_TEMPLATE.format(
        query=query,
        context=_format_context(context_chunks),
        answer=answer,
        cited_ids=", ".join(cited_chunk_ids) if cited_chunk_ids else "(none found)",
    )
    scores, raw, latency_ms = await _call_judge(prompt)

    results = {
        metric: JudgeScore(
            metric=metric, score=score, reasoning=reasoning, raw_response=raw, latency_ms=latency_ms
        )
        for metric, (score, reasoning) in scores.items()
    }

    # Deterministic guardrail: a citation pointing at a chunk that was never
    # retrieved is a hard error regardless of how the judge scored it.
    valid_ids = {c.chunk_id for c in context_chunks}
    fabricated = [cid for cid in cited_chunk_ids if cid not in valid_ids]
    if fabricated:
        citation = results["citation_accuracy"]
        citation.score = min(citation.score, 0.2)
        citation.reasoning += (
            f" [Deterministic check] Fabricated chunk ID(s) not present in retrieved "
            f"context: {fabricated}. Score capped at 0.2."
        )

    overall = sum(results[m].score * _METRIC_WEIGHTS[m] for m in METRICS)

    return GenerationEvalResult(
        faithfulness=results["faithfulness"],
        answer_relevance=results["answer_relevance"],
        context_relevance=results["context_relevance"],
        citation_accuracy=results["citation_accuracy"],
        overall_score=overall,
        total_latency_ms=latency_ms,
    )
