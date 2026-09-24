"""Unit tests for generation-evaluation helpers (no live Gemini calls)."""

from unittest.mock import patch

from app.eval_generation import (
    Citation,
    evaluate_citation_accuracy,
    evaluate_context_relevance,
    evaluate_faithfulness,
    evaluate_generation,
    extract_citations,
)
from app.models import Document
from app.services import NO_ANSWER_MESSAGE


def _doc(chunk_id: int, content: str) -> Document:
    return Document(id=chunk_id, title="doc.pdf", content=content)


class TestExtractCitations:
    def test_parses_inline_markers(self):
        answer = "Revenue grew 12% [4]. Headcount stayed flat [7]."
        citations = extract_citations(answer)
        assert [(c.chunk_id, c.sentence) for c in citations] == [
            (4, "Revenue grew 12%"),
            (7, "Headcount stayed flat"),
        ]

    def test_empty_when_no_markers(self):
        assert extract_citations("No citations here.") == []


class TestFaithfulness:
    def test_no_context_is_unfaithful(self):
        result = evaluate_faithfulness("Q?", "An answer.", [])
        assert result["score"] == 0.0

    @patch("app.eval_generation._call_judge")
    def test_score_from_grounded_claims(self, mock_judge):
        mock_judge.return_value = {
            "claims": [
                {"claim": "A", "grounded": True},
                {"claim": "B", "grounded": False},
            ],
            "reasoning": "One claim invented.",
        }
        result = evaluate_faithfulness("Q?", "A then B.", [_doc(1, "A is true.")])
        assert result["score"] == 0.5
        assert "invented" in result["reasoning"]


class TestContextRelevance:
    def test_empty_context(self):
        result = evaluate_context_relevance("Q?", [])
        assert result["score"] == 0.0

    @patch("app.eval_generation._call_judge")
    def test_signal_to_noise(self, mock_judge):
        mock_judge.return_value = {
            "chunks": [
                {"chunk_id": 1, "relevant": True},
                {"chunk_id": 2, "relevant": False},
                {"chunk_id": 3, "relevant": True},
            ],
            "reasoning": "Two of three chunks help.",
        }
        result = evaluate_context_relevance("Q?", [_doc(1, "a"), _doc(2, "b"), _doc(3, "c")])
        assert result["score"] == round(2 / 3, 2)


class TestCitationAccuracy:
    def test_refusal_without_citations_is_correct(self):
        result = evaluate_citation_accuracy(NO_ANSWER_MESSAGE, [], [_doc(1, "irrelevant")])
        assert result["score"] == 1.0

    def test_missing_citations_score_zero(self):
        result = evaluate_citation_accuracy("The sky is blue.", [], [_doc(1, "sky")])
        assert result["score"] == 0.0

    def test_unknown_chunk_ids_score_zero(self):
        citations = [Citation(chunk_id=99, sentence="made up")]
        result = evaluate_citation_accuracy("made up [99].", citations, [_doc(1, "real")])
        assert result["score"] == 0.0

    @patch("app.eval_generation._call_judge")
    def test_mix_of_supported_and_invalid(self, mock_judge):
        mock_judge.return_value = {
            "verifications": [{"index": 1, "supported": True}],
            "reasoning": "Chunk 1 supports the claim.",
        }
        citations = [
            Citation(chunk_id=1, sentence="Revenue grew"),
            Citation(chunk_id=99, sentence="Aliens landed"),
        ]
        result = evaluate_citation_accuracy(
            "Revenue grew [1]. Aliens landed [99].",
            citations,
            [_doc(1, "Revenue grew 12% year over year.")],
        )
        assert result["score"] == 0.5
        assert "99" in result["reasoning"]


class TestEvaluateGeneration:
    @patch("app.eval_generation.evaluate_citation_accuracy")
    @patch("app.eval_generation.evaluate_context_relevance")
    @patch("app.eval_generation.evaluate_answer_relevance")
    @patch("app.eval_generation.evaluate_faithfulness")
    def test_overall_is_mean_of_four_metrics(
        self, mock_faith, mock_rel, mock_ctx, mock_cite
    ):
        mock_faith.return_value = {"score": 1.0, "reasoning": "ok"}
        mock_rel.return_value = {"score": 0.5, "reasoning": "ok"}
        mock_ctx.return_value = {"score": 0.5, "reasoning": "ok"}
        mock_cite.return_value = {"score": 0.0, "reasoning": "ok"}

        result = evaluate_generation("Q?", "A [1].", [_doc(1, "A")])
        assert result["overall_score"] == 0.5
        assert "faithfulness" in result
        assert isinstance(result["citations"], list)
