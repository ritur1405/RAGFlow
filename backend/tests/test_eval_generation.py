"""Unit tests for the LLM-as-a-judge layer in app/eval_generation.py.

No live Gemini calls: the judge client or the single judge-call helper is
patched. Covers citation parsing, response normalisation, the deterministic
fabricated-citation guardrail, the weighted overall score, and the failure
policy (no client or repeated failures raise; nothing is invented).

Adapted from the Week 6 generation-evaluation tests in PR #9 to the module's
current API.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import eval_generation as eg
from app.eval_generation import ContextChunk, JudgeScore


def _score(metric: str, value: float) -> JudgeScore:
    return JudgeScore(metric=metric, score=value, reasoning=f"{metric} ok", latency_ms=5.0)


def _chunks(*ids: str) -> list[ContextChunk]:
    return [ContextChunk(chunk_id=i, text=f"text {i}") for i in ids]


class TestExtractCitationIds:
    def test_parses_inline_markers_in_order(self):
        answer = "Revenue grew 12% [4]. Headcount stayed flat [7]. Again [4]."
        assert eg.extract_citation_ids(answer) == ["4", "7"]

    def test_accepts_chunk_style_markers(self):
        assert eg.extract_citation_ids("See (chunk-7) and [chunk_3].") == ["chunk-7", "chunk_3"]

    def test_empty_when_no_markers(self):
        assert eg.extract_citation_ids("No citations here.") == []


class TestNormalisation:
    def test_extract_json_strips_code_fences(self):
        assert eg._extract_json('```json\n{"score": 0.5}\n```') == {"score": 0.5}

    @pytest.mark.parametrize(
        "raw, expected", [(0.4, 0.4), ("0.9", 0.9), (1.7, 1.0), (-2, 0.0), (None, 0.0), ("x", 0.0)]
    )
    def test_clamp01(self, raw, expected):
        assert eg._clamp01(raw) == expected


class TestJudgeCall:
    def test_no_client_raises_unavailable(self):
        with patch.object(eg, "_client", None):
            assert eg.judge_available() is False
            with pytest.raises(eg.JudgeUnavailableError):
                asyncio.run(eg._call_gemini_judge("sys", "prompt", "faithfulness"))

    def test_parses_score_and_reasoning(self):
        client = MagicMock()
        client.models.generate_content.return_value.text = (
            '{"score": 0.75, "reasoning": "1. grounded"}'
        )
        with patch.object(eg, "_client", client):
            result = asyncio.run(eg._call_gemini_judge("sys", "prompt", "faithfulness"))

        assert (result.metric, result.score, result.reasoning) == ("faithfulness", 0.75, "1. grounded")

    def test_retries_then_raises_call_error(self):
        client = MagicMock()
        client.models.generate_content.side_effect = RuntimeError("429")
        with patch.object(eg, "_client", client), \
             patch.object(eg, "_RETRY_BACKOFF_SECONDS", 0):
            with pytest.raises(eg.JudgeCallError, match="failed after 3 attempts"):
                asyncio.run(eg._call_gemini_judge("sys", "prompt", "answer_relevance"))

        assert client.models.generate_content.call_count == eg._MAX_RETRIES

    def test_recovers_from_a_transient_failure(self):
        ok = MagicMock(text='{"score": 1, "reasoning": "fine"}')
        client = MagicMock()
        client.models.generate_content.side_effect = [RuntimeError("503"), ok]
        with patch.object(eg, "_client", client), \
             patch.object(eg, "_RETRY_BACKOFF_SECONDS", 0):
            result = asyncio.run(eg._call_gemini_judge("sys", "prompt", "context_relevance"))

        assert result.score == 1.0


class TestCitationAccuracyGuardrail:
    def test_fabricated_chunk_id_caps_score(self):
        judge = AsyncMock(return_value=_score("citation_accuracy", 0.95))
        with patch.object(eg, "_call_gemini_judge", judge):
            result = asyncio.run(
                eg.evaluate_citation_accuracy("Claim [1]. Invented [99].", _chunks("1"))
            )

        assert result.score == 0.2
        assert "99" in result.reasoning

    def test_valid_citations_keep_judge_score(self):
        judge = AsyncMock(return_value=_score("citation_accuracy", 0.9))
        with patch.object(eg, "_call_gemini_judge", judge):
            result = asyncio.run(
                eg.evaluate_citation_accuracy("Claim [1].", _chunks("1", "2"), ["1"])
            )

        assert result.score == 0.9

    def test_prompt_reports_missing_citations(self):
        judge = AsyncMock(return_value=_score("citation_accuracy", 0.0))
        with patch.object(eg, "_call_gemini_judge", judge):
            asyncio.run(eg.evaluate_citation_accuracy("The sky is blue.", _chunks("1")))

        assert "(none found)" in judge.call_args.args[1]


class TestRunGenerationEval:
    def test_overall_is_the_weighted_sum(self):
        with patch.object(eg, "evaluate_faithfulness", AsyncMock(return_value=_score("f", 1.0))), \
             patch.object(eg, "evaluate_answer_relevance", AsyncMock(return_value=_score("a", 0.5))), \
             patch.object(eg, "evaluate_context_relevance", AsyncMock(return_value=_score("c", 0.5))), \
             patch.object(eg, "evaluate_citation_accuracy", AsyncMock(return_value=_score("x", 0.0))):
            result = asyncio.run(eg.run_generation_eval("Q?", "A [1].", _chunks("1")))

        # 1.0*0.35 + 0.5*0.2 + 0.5*0.2 + 0.0*0.25
        assert result.overall_score == pytest.approx(0.55)
        assert sum(eg._METRIC_WEIGHTS.values()) == pytest.approx(1.0)

    def test_one_failing_metric_fails_the_whole_sample(self):
        failing = AsyncMock(side_effect=eg.JudgeCallError("boom"))
        with patch.object(eg, "evaluate_faithfulness", failing), \
             patch.object(eg, "evaluate_answer_relevance", AsyncMock(return_value=_score("a", 1))), \
             patch.object(eg, "evaluate_context_relevance", AsyncMock(return_value=_score("c", 1))), \
             patch.object(eg, "evaluate_citation_accuracy", AsyncMock(return_value=_score("x", 1))):
            with pytest.raises(eg.JudgeCallError):
                asyncio.run(eg.run_generation_eval("Q?", "A", _chunks("1")))
