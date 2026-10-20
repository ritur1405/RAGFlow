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

Failure policy: this module never invents a score. If the judge cannot run --
no API key, SDK missing, or the call fails after retries -- it raises. A run
that could not be judged is recorded as failed, not as a run that happened to
score well. See JudgeUnavailableError and JudgeCallError.

Environment variables:
    GEMINI_API_KEY      - required (already required by app.services).
    GEMINI_JUDGE_MODEL  - optional. Defaults to app.config.LLM_MODEL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from google.genai import types

from app.config import LLM_MODEL

# Reuse the single client initialised in services.py rather than creating a
# second one, matching how app/llm.py sources its client.
from app.services import client as _client

logger = logging.getLogger("ragops.eval_generation")

# The judge may deliberately differ from the answer-generation model.
GEMINI_JUDGE_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", LLM_MODEL)

# Retry policy for transient Gemini errors (rate limits, 5xx, timeouts).
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.5


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
    """Normalized output of a single LLM-as-a-judge metric call."""

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
# Core Gemini call helper
# --------------------------------------------------------------------------- #


def _extract_json(raw_text: str) -> dict:
    """Strips markdown code fences and parses raw JSON response."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def judge_available() -> bool:
    """True when a judge client is configured. Callers may pre-flight with this."""
    return _client is not None


async def _call_gemini_judge(
    system_instruction: str,
    user_prompt: str,
    metric_name: str,
) -> JudgeScore:
    """Shared execution path for judge metrics.

    Runs the blocking SDK call in a worker thread and retries transient
    failures. Raises rather than returning a placeholder score: a metric that
    could not be measured must not look like a metric that scored well.
    """
    if _client is None:
        raise JudgeUnavailableError(
            "Gemini judge is not configured (no client). Set GEMINI_API_KEY to "
            "run evaluations."
        )

    start = time.perf_counter()
    last_error: Optional[Exception] = None

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=1024,
        response_mime_type="application/json",
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await asyncio.to_thread(
                _client.models.generate_content,
                model=GEMINI_JUDGE_MODEL,
                contents=user_prompt,
                config=config,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            parsed = _extract_json(response.text)
            score = _clamp01(parsed.get("score"))
            reasoning = (
                str(parsed.get("reasoning", "")).strip()
                or "No reasoning provided by judge."
            )

            return JudgeScore(
                metric=metric_name,
                score=score,
                reasoning=reasoning,
                raw_response=parsed,
                latency_ms=elapsed_ms,
            )
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
            last_error = exc
            logger.warning(
                "Judge call failed (metric=%s, attempt=%d/%d): %s",
                metric_name, attempt, _MAX_RETRIES, exc,
            )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    logger.exception(
        "Judge call for metric=%s failed after %d attempts", metric_name, _MAX_RETRIES,
        exc_info=last_error,
    )
    raise JudgeCallError(
        f"Judge metric '{metric_name}' failed after {_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def _format_context(context_chunks: list[ContextChunk]) -> str:
    if not context_chunks:
        return "(no context chunks were retrieved)"
    return "\n\n".join(
        f"[chunk_id: {c.chunk_id}]\n{c.text.strip()}" for c in context_chunks
    )


# --------------------------------------------------------------------------- #
# Metric (a): Faithfulness
# --------------------------------------------------------------------------- #

_FAITHFULNESS_SYSTEM = """You are a strict factual-consistency judge for a Retrieval-Augmented \
Generation (RAG) system. You determine whether a generated answer is fully grounded in the \
provided context, with zero hallucinated or unsupported claims.

Score guidance:
- 1.0: every factual claim in the answer is directly supported by the context.
- 0.5-0.9: mostly supported, with minor unsupported details or slight overreach.
- 0.1-0.4: several claims are not supported by the context.
- 0.0: the answer is substantially fabricated or contradicts the context.

Respond ONLY with a JSON object of the exact shape:
{"score": <float 0.0-1.0>, "reasoning": "<3-5 short numbered steps of your reasoning>"}
"""

_FAITHFULNESS_TEMPLATE = """QUERY:
{query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Step-by-step task:
1. Break the answer into its individual factual claims.
2. For each claim, check whether it is explicitly or reasonably entailed by the context.
3. Flag any claim that is unsupported, exaggerated, or contradicts the context.
4. Compute a faithfulness score reflecting the fraction of claims that are properly grounded.
Return the JSON object described in your instructions."""


async def evaluate_faithfulness(
    query: str, answer: str, context_chunks: list[ContextChunk]
) -> JudgeScore:
    """Checks whether `answer` is strictly grounded in `context_chunks`."""
    prompt = _FAITHFULNESS_TEMPLATE.format(
        query=query, context=_format_context(context_chunks), answer=answer
    )
    return await _call_gemini_judge(_FAITHFULNESS_SYSTEM, prompt, "faithfulness")


# --------------------------------------------------------------------------- #
# Metric (b): Answer Relevance
# --------------------------------------------------------------------------- #

_ANSWER_RELEVANCE_SYSTEM = """You are a judge evaluating whether a generated answer directly and \
completely addresses the user's query. You do NOT check factual correctness against context here \
- only whether the answer is on-topic, complete, and avoids padding, evasion, or answering a \
different question than the one asked.

Score guidance:
- 1.0: directly and completely answers the query.
- 0.5-0.9: answers the query but is partially incomplete, indirect, or includes irrelevant padding.
- 0.1-0.4: only tangentially related to the query.
- 0.0: does not address the query at all (e.g., "I don't know" with no attempt, or off-topic).

Respond ONLY with a JSON object of the exact shape:
{"score": <float 0.0-1.0>, "reasoning": "<3-5 short numbered steps of your reasoning>"}
"""

_ANSWER_RELEVANCE_TEMPLATE = """QUERY:
{query}

GENERATED ANSWER:
{answer}

Step-by-step task:
1. Identify the core information need expressed in the query.
2. Check whether the answer directly targets that need.
3. Note any irrelevant, evasive, or off-topic content in the answer.
4. Compute a relevance score for how directly and completely the answer addresses the query.
Return the JSON object described in your instructions."""


async def evaluate_answer_relevance(query: str, answer: str) -> JudgeScore:
    """Checks if the answer addresses the query, independent of context grounding."""
    prompt = _ANSWER_RELEVANCE_TEMPLATE.format(query=query, answer=answer)
    return await _call_gemini_judge(
        _ANSWER_RELEVANCE_SYSTEM, prompt, "answer_relevance"
    )


# --------------------------------------------------------------------------- #
# Metric (c): Context Relevance
# --------------------------------------------------------------------------- #

_CONTEXT_RELEVANCE_SYSTEM = """You are a judge evaluating retrieval quality for a RAG pipeline. \
Given a query and the chunks retrieved for it, determine what fraction of the retrieved context \
is actually useful/relevant to answering the query versus noise (off-topic, redundant, or \
tangential chunks).

Score guidance:
- 1.0: nearly all retrieved content is relevant and useful for answering the query.
- 0.5-0.9: a majority of the content is relevant, with some noisy or redundant chunks.
- 0.1-0.4: mostly noise, with only a small relevant portion.
- 0.0: none of the retrieved context is relevant to the query.

Respond ONLY with a JSON object of the exact shape:
{"score": <float 0.0-1.0>, "reasoning": "<3-5 short numbered steps of your reasoning>"}
"""

_CONTEXT_RELEVANCE_TEMPLATE = """QUERY:
{query}

RETRIEVED CONTEXT CHUNKS:
{context}

Step-by-step task:
1. For each chunk, judge whether it contains information useful for answering the query.
2. Count how many chunks are relevant vs. noise (off-topic/redundant).
3. Consider whether noisy chunks are likely to distract or mislead the generator.
4. Compute a context relevance score approximating the useful-to-noise ratio.
Return the JSON object described in your instructions."""


async def evaluate_context_relevance(
    query: str, context_chunks: list[ContextChunk]
) -> JudgeScore:
    """Evaluates the quality/noise ratio of the retrieved context for the query."""
    prompt = _CONTEXT_RELEVANCE_TEMPLATE.format(
        query=query, context=_format_context(context_chunks)
    )
    return await _call_gemini_judge(
        _CONTEXT_RELEVANCE_SYSTEM, prompt, "context_relevance"
    )


# --------------------------------------------------------------------------- #
# Metric (d): Citation Accuracy
# --------------------------------------------------------------------------- #

_CITATION_ACCURACY_SYSTEM = """You are a citation-auditing judge for a RAG system. The generated \
answer contains inline citations referencing chunk IDs (e.g. "[chunk_3]" or similar markers). You \
must verify that each cited chunk ID actually contains text that supports the specific claim it \
was attached to, and that no claim requiring support is left uncited.

Score guidance:
- 1.0: every citation correctly supports its associated claim, and no supportable claim is missing a citation.
- 0.5-0.9: most citations are accurate; minor mismatches or a small number of missing citations.
- 0.1-0.4: many citations point to chunks that do not support the claim, or citations are largely missing.
- 0.0: citations are absent, fabricated (referencing chunk IDs that don't exist), or systematically wrong.

Respond ONLY with a JSON object of the exact shape:
{"score": <float 0.0-1.0>, "reasoning": "<3-5 short numbered steps of your reasoning>", \
"mismatched_citations": [{"cited_chunk_id": "<id>", "claim": "<claim text>", "issue": "<why it does not match>"}]}
"""

_CITATION_ACCURACY_TEMPLATE = """QUERY:
{query}

RETRIEVED CONTEXT CHUNKS (with chunk IDs):
{context}

GENERATED ANSWER (with inline citations):
{answer}

CITED CHUNK IDS EXTRACTED FROM THE ANSWER:
{cited_ids}

Step-by-step task:
1. For each citation in the answer, locate the chunk with the matching chunk_id in the context.
2. Verify the cited chunk's text actually supports the specific claim next to that citation.
3. Check whether every claim that requires support (specific facts, numbers, named entities) has a citation.
4. Flag fabricated chunk IDs (cited but not present in context) as maximum-severity errors.
5. Compute a citation accuracy score reflecting the fraction of correct, well-supported citations.
Return the JSON object described in your instructions, including the mismatched_citations list \
(empty array if none)."""


async def evaluate_citation_accuracy(
    answer: str,
    context_chunks: list[ContextChunk],
    cited_chunk_ids: Optional[list[str]] = None,
    query: str = "",
) -> JudgeScore:
    """Verifies that cited chunk IDs in `answer` actually support the context text."""
    if cited_chunk_ids is None:
        cited_chunk_ids = extract_citation_ids(answer)

    prompt = _CITATION_ACCURACY_TEMPLATE.format(
        query=query,
        context=_format_context(context_chunks),
        answer=answer,
        cited_ids=", ".join(cited_chunk_ids) if cited_chunk_ids else "(none found)",
    )

    score = await _call_gemini_judge(
        _CITATION_ACCURACY_SYSTEM, prompt, "citation_accuracy"
    )

    # Deterministic guardrail: a citation pointing at a chunk that was never
    # retrieved is a hard error regardless of how the judge scored it.
    valid_ids = {c.chunk_id for c in context_chunks}
    fabricated = [cid for cid in cited_chunk_ids if cid not in valid_ids]
    if fabricated:
        score.score = min(score.score, 0.2)
        score.reasoning += (
            f" [Deterministic check] Fabricated chunk ID(s) not present in retrieved "
            f"context: {fabricated}. Score capped at 0.2."
        )

    return score


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
    """Runs all four judge metrics concurrently for a single generation sample.

    Raises JudgeUnavailableError / JudgeCallError if any metric cannot be
    measured. Partial results are not returned.
    """
    start = time.perf_counter()

    (
        faithfulness,
        answer_relevance,
        context_relevance,
        citation_accuracy,
    ) = await asyncio.gather(
        evaluate_faithfulness(query, answer, context_chunks),
        evaluate_answer_relevance(query, answer),
        evaluate_context_relevance(query, context_chunks),
        evaluate_citation_accuracy(answer, context_chunks, cited_chunk_ids, query=query),
    )

    total_latency_ms = (time.perf_counter() - start) * 1000

    overall = (
        faithfulness.score * _METRIC_WEIGHTS["faithfulness"]
        + answer_relevance.score * _METRIC_WEIGHTS["answer_relevance"]
        + context_relevance.score * _METRIC_WEIGHTS["context_relevance"]
        + citation_accuracy.score * _METRIC_WEIGHTS["citation_accuracy"]
    )

    return GenerationEvalResult(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_relevance=context_relevance,
        citation_accuracy=citation_accuracy,
        overall_score=overall,
        total_latency_ms=total_latency_ms,
    )
