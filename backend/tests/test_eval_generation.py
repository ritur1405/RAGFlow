"""Unit tests for the LLM-as-a-judge layer in app/eval_generation.py.

No live Gemini calls: the client is a MagicMock. Covers citation parsing,
response parsing, the single-call judge (all four metrics from one request),
model-specific request shapes, the deterministic fabricated-citation
guardrail, the weighted overall score, and the failure policy (no client,
malformed responses, or repeated failures raise; nothing is invented).

Adapted from the Week 6 generation-evaluation tests in PR #9.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from app import eval_generation as eg
from app.eval_generation import ContextChunk


def _chunks(*ids: str) -> list[ContextChunk]:
    return [ContextChunk(chunk_id=i, text=f"text {i}") for i in ids]


def _judge_json(faith=0.9, ans=0.8, ctx=0.7, cite=0.6) -> str:
    return json.dumps({
        "faithfulness": {"score": faith, "reasoning": "1. grounded"},
        "answer_relevance": {"score": ans, "reasoning": "1. on topic"},
        "context_relevance": {"score": ctx, "reasoning": "1. mostly useful"},
        "citation_accuracy": {"score": cite, "reasoning": "1. cites match"},
    })


def _client(*texts):
    client = MagicMock()
    client.models.generate_content.side_effect = [
        t if isinstance(t, Exception) else MagicMock(text=t) for t in texts
    ]
    return client


def _run(client, answer="Claim [1].", chunks=None, cited=None, model="gemini-3.5-flash-lite"):
    with patch.object(eg, "_client", client), \
         patch.object(eg, "GEMINI_JUDGE_MODEL", model), \
         patch.object(eg, "_RETRY_BACKOFF_SECONDS", 0):
        return asyncio.run(eg.run_generation_eval("Q?", answer, chunks or _chunks("1"), cited))


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

    def test_extract_json_tolerates_prose_around_the_object(self):
        assert eg._extract_json('Here is my evaluation:\n{"a": {"b": 1}}\nDone.') == {"a": {"b": 1}}

    @pytest.mark.parametrize(
        "raw, expected", [(0.4, 0.4), ("0.9", 0.9), (1.7, 1.0), (-2, 0.0), (None, 0.0), ("x", 0.0)]
    )
    def test_clamp01(self, raw, expected):
        assert eg._clamp01(raw) == expected


class TestSingleCallJudge:
    def test_one_request_scores_all_four_metrics(self):
        client = _client(_judge_json())
        result = _run(client)

        assert client.models.generate_content.call_count == 1
        assert (result.faithfulness.score, result.answer_relevance.score,
                result.context_relevance.score, result.citation_accuracy.score) == (0.9, 0.8, 0.7, 0.6)
        assert result.faithfulness.reasoning == "1. grounded"
        # 0.9*.35 + 0.8*.2 + 0.7*.2 + 0.6*.25
        assert result.overall_score == pytest.approx(0.765)
        assert sum(eg._METRIC_WEIGHTS.values()) == pytest.approx(1.0)

    def test_prompt_carries_query_context_answer_and_citations(self):
        client = _client(_judge_json())
        _run(client, answer="Claim [1].", chunks=_chunks("1", "2"))

        contents = client.models.generate_content.call_args.kwargs["contents"]
        assert "Q?" in contents and "Claim [1]." in contents
        assert "[chunk_id: 2]" in contents
        assert "CITED CHUNK IDS EXTRACTED FROM THE ANSWER:\n1" in contents

    def test_gemini_judge_uses_system_instruction_and_json_mode(self):
        client = _client(_judge_json())
        _run(client, model="gemini-3.5-flash-lite")

        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-3.5-flash-lite"
        assert kwargs["config"].system_instruction == eg._JUDGE_INSTRUCTIONS
        assert kwargs["config"].response_mime_type == "application/json"

    def test_gemma_judge_gets_instructions_in_the_prompt(self):
        client = _client(_judge_json())
        _run(client, model="gemma-4-31b-it")

        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["config"].system_instruction is None
        assert kwargs["config"].response_mime_type is None
        assert kwargs["contents"].startswith(eg._JUDGE_INSTRUCTIONS)

    def test_fabricated_citation_caps_citation_accuracy(self):
        client = _client(_judge_json(cite=0.95))
        result = _run(client, answer="Claim [1]. Invented [99].")

        assert result.citation_accuracy.score == 0.2
        assert "99" in result.citation_accuracy.reasoning
        # The other metrics keep the judge's scores.
        assert result.faithfulness.score == 0.9

    def test_missing_citations_are_reported_to_the_judge(self):
        client = _client(_judge_json())
        _run(client, answer="The sky is blue.")
        assert "(none found)" in client.models.generate_content.call_args.kwargs["contents"]


class TestFailurePolicy:
    def test_no_client_raises_unavailable(self):
        with patch.object(eg, "_client", None):
            assert eg.judge_available() is False
            with pytest.raises(eg.JudgeUnavailableError):
                asyncio.run(eg.run_generation_eval("Q?", "A", _chunks("1")))

    def test_a_missing_metric_is_retried_then_fails(self):
        partial = json.dumps({"faithfulness": {"score": 1, "reasoning": "ok"}})
        client = _client(partial, partial, partial)
        with pytest.raises(eg.JudgeCallError, match="missing metric 'answer_relevance'"):
            _run(client)
        assert client.models.generate_content.call_count == eg._MAX_RETRIES

    def test_recovers_from_a_malformed_response(self):
        client = _client("not json", _judge_json())
        assert _run(client).faithfulness.score == 0.9
        assert client.models.generate_content.call_count == 2

    def test_repeated_errors_raise_call_error(self):
        client = _client(*[RuntimeError("503")] * eg._MAX_RETRIES)
        with pytest.raises(eg.JudgeCallError, match="failed after 3 attempts"):
            _run(client)
