"""Tests for the retrieval metrics in app/eval_retrieval.py, against hand-computed values."""

import math

import pytest

from app import eval_retrieval as er
from app.models import Document


def _doc(page, file_name="paper.pdf"):
    return Document(id=page * 100, title="t", content="c", page_number=page, file_name=file_name)


class TestRelevance:
    def test_page_and_file_must_both_match(self):
        docs = [_doc(5), _doc(5, "other.pdf"), _doc(6)]
        assert er.relevance(docs, "paper.pdf", [5]) == [True, False, False]

    def test_file_match_is_case_insensitive(self):
        assert er.relevance([_doc(5, "Paper.PDF")], "paper.pdf", [5]) == [True]

    def test_no_source_file_means_any_file(self):
        assert er.relevance([_doc(5, "other.pdf")], None, [5]) == [True]


class TestMetrics:
    def test_precision_divides_by_k_not_by_retrieved(self):
        # A retriever returning fewer than k chunks is not rewarded for it.
        assert er.precision_at_k([True], 3) == pytest.approx(1 / 3)
        assert er.precision_at_k([True, False, True], 3) == pytest.approx(2 / 3)

    def test_recall_is_capped_by_k(self):
        # 2 of 2 labelled pages found.
        assert er.recall_at_k([5, 7], [5, 7], k=3) == 1.0
        # 1 of 2 labelled pages found.
        assert er.recall_at_k([5, 5], [5, 7], k=3) == 0.5
        # 4 labelled pages but only k=3 slots: denominator is capped at 3.
        assert er.recall_at_k([1, 2, 3], [1, 2, 3, 4], k=3) == 1.0

    def test_reciprocal_rank(self):
        assert er.reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
        assert er.reciprocal_rank([True, True]) == 1.0
        assert er.reciprocal_rank([False, False]) == 0.0
        assert er.reciprocal_rank([]) == 0.0

    def test_ndcg(self):
        assert er.ndcg_at_k([True, True, True], 3) == pytest.approx(1.0)
        assert er.ndcg_at_k([False, False, False], 3) == 0.0
        idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        # Relevant only at rank 2.
        assert er.ndcg_at_k([False, True, False], 3) == pytest.approx((1 / math.log2(3)) / idcg)
        # Rank order matters: the same hit at rank 1 scores higher.
        assert er.ndcg_at_k([True, False, False], 3) > er.ndcg_at_k([False, False, True], 3)

    def test_ndcg_with_known_relevant_count(self):
        assert er.ndcg_at_k([True, False, False], 3, num_relevant=1) == pytest.approx(1.0)


class TestScoreRetrieval:
    def test_unlabelled_question_has_no_scores(self):
        assert er.score_retrieval([_doc(1)], "paper.pdf", [], k=3) is None

    def test_full_example(self):
        docs = [_doc(9), _doc(5), _doc(5)]
        scores = er.score_retrieval(docs, "paper.pdf", [5, 7], k=3)

        assert scores.k == 3
        assert scores.precision_at_k == pytest.approx(2 / 3)
        assert scores.recall_at_k == pytest.approx(1 / 2)
        assert scores.reciprocal_rank == pytest.approx(1 / 2)
        idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        assert scores.ndcg_at_k == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / idcg)

    def test_only_top_k_count(self):
        docs = [_doc(1), _doc(2), _doc(3), _doc(5)]
        assert er.score_retrieval(docs, "paper.pdf", [5], k=3).reciprocal_rank == 0.0
