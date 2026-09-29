"""Tests for the four RAG configurations in app/rag_pipelines.py.

No live Gemini calls and no database: retrieval, generation and the reranker
are all patched, so these tests assert the wiring and ranking logic of each
configuration rather than model behaviour.
"""

from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest

from app import rag_pipelines as rp
from app.config import NO_ANSWER_MESSAGE
from app.models import Document

# Mirrors the row shape returned by _load_all_documents' select().
Row = namedtuple("Row", "id title content page_number chunk_index")


def _doc(doc_id: int, content: str) -> Document:
    return Document(
        id=doc_id, title="doc.pdf", content=content, page_number=1, chunk_index=doc_id
    )


def _row(doc_id: int, content: str) -> Row:
    return Row(id=doc_id, title="doc.pdf", content=content, page_number=1, chunk_index=doc_id)


@pytest.fixture
def fake_generation():
    """Patches the LLM so generation returns a citing answer without a network call."""
    with patch.object(rp, "llm_generate_answer") as gen:
        gen.return_value = "Revenue grew. [1] Costs fell. [2]"
        yield gen


class TestRegistry:
    def test_registry_keys_match_api_enum(self):
        """The registry is keyed by the API's config values, so no translation is needed."""
        from app.routers.eval import RAGConfig

        assert set(rp.PIPELINE_REGISTRY) == {c.value for c in RAGConfig}

    def test_every_config_is_callable(self):
        assert all(callable(fn) for fn in rp.PIPELINE_REGISTRY.values())


class TestBaselineRag:
    def test_uses_plain_top_k_without_filtering(self, fake_generation):
        db = MagicMock()
        with patch.object(rp, "generate_embedding", return_value=[0.1] * 768) as emb, \
             patch.object(rp, "search_documents") as search:
            # Second doc is beyond the distance gate; baseline must keep it anyway.
            search.return_value = [(_doc(1, "a"), 0.1), (_doc(2, "b"), 5.0)]

            result = rp.run_baseline_rag(db, "How did revenue change?")

        emb.assert_called_once()
        assert search.call_args.kwargs["top_k"] == rp.FINAL_TOP_K
        assert [d.id for d in result.context_docs] == [1, 2]
        assert result.retrieved_chunk_ids == ["1", "2"]
        assert result.cited_chunk_ids == ["1", "2"]
        assert result.latency_ms >= 0


class TestDenseRag:
    def test_applies_distance_gate_and_sorts(self, fake_generation):
        db = MagicMock()
        with patch.object(rp, "generate_embedding", return_value=[0.1] * 768), \
             patch.object(rp, "search_documents") as search:
            search.return_value = [
                (_doc(1, "far"), 0.9),    # beyond MAX_RELEVANT_DISTANCE (0.8) -> dropped
                (_doc(2, "near"), 0.2),
                (_doc(3, "mid"), 0.5),
            ]

            result = rp.run_dense_rag(db, "q")

        assert [d.id for d in result.context_docs] == [2, 3]

    def test_pulls_wider_candidate_pool_than_baseline(self, fake_generation):
        db = MagicMock()
        with patch.object(rp, "generate_embedding", return_value=[0.1] * 768), \
             patch.object(rp, "search_documents") as search:
            search.return_value = []
            rp.run_dense_rag(db, "q")

        assert search.call_args.kwargs["top_k"] == rp.RERANK_CANDIDATE_POOL

    def test_no_surviving_context_returns_no_answer_message(self):
        db = MagicMock()
        with patch.object(rp, "generate_embedding", return_value=[0.1] * 768), \
             patch.object(rp, "search_documents", return_value=[(_doc(1, "x"), 9.9)]), \
             patch.object(rp, "llm_generate_answer") as gen:
            result = rp.run_dense_rag(db, "q")

        # The LLM is never called when nothing was retrieved.
        gen.assert_not_called()
        assert result.answer == NO_ANSWER_MESSAGE
        assert result.context_docs == []
        assert result.cited_chunk_ids == []


class TestBm25AndFusion:
    def test_bm25_ranks_by_term_overlap_and_drops_zero_scores(self):
        rows = [
            _row(1, "revenue grew strongly this quarter"),
            _row(2, "unrelated boilerplate about parking"),
            _row(3, "revenue revenue revenue"),
        ]
        ranked = rp._bm25_rank(rows, "revenue", top_k=5)

        assert [d.id for d in ranked] == [3, 1]  # doc 2 scores 0 and is excluded
        assert all(isinstance(d, Document) for d in ranked)

    def test_bm25_empty_corpus(self):
        assert rp._bm25_rank([], "anything", top_k=5) == []

    def test_rrf_rewards_documents_ranked_by_both_retrievers(self):
        dense = [_doc(1, "a"), _doc(2, "b")]
        bm25 = [_doc(3, "c"), _doc(1, "a")]

        fused = rp._reciprocal_rank_fusion([dense, bm25], top_k=3)

        # Doc 1 appears in both lists, so its fused score is highest.
        assert fused[0].id == 1
        assert {d.id for d in fused} == {1, 2, 3}

    def test_rrf_respects_top_k(self):
        lists = [[_doc(i, str(i)) for i in range(1, 6)]]
        assert len(rp._reciprocal_rank_fusion(lists, top_k=2)) == 2


class TestHybridRag:
    def test_fuses_dense_and_bm25(self, fake_generation):
        db = MagicMock()
        # The BM25 corpus needs documents that do *not* contain the query term,
        # otherwise the term's IDF is zero and no document scores above 0.
        corpus = [
            _row(2, "revenue grew strongly"),
            _row(3, "parking policy details"),
            _row(4, "office furniture inventory"),
        ]
        with patch.object(rp, "generate_embedding", return_value=[0.1] * 768), \
             patch.object(rp, "search_documents", return_value=[(_doc(1, "a"), 0.1)]), \
             patch.object(rp, "_load_all_documents", return_value=corpus):
            result = rp.run_hybrid_rag(db, "revenue")

        # Doc 1 from dense, doc 2 from BM25: the fusion contains both retrievers.
        assert {d.id for d in result.context_docs} == {1, 2}


class TestHybridRerankRag:
    def test_reorders_by_llm_relevance_scores(self, fake_generation):
        db = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"scores": [{"chunk_id": 1, "relevance": 2}, {"chunk_id": 2, "relevance": 9}]}'
        client = MagicMock()
        client.models.generate_content.return_value = mock_response

        with patch.object(rp, "_hybrid_candidates", return_value=[_doc(1, "a"), _doc(2, "b")]), \
             patch.object(rp, "gemini_client", client):
            result = rp.run_hybrid_rerank_rag(db, "q")

        # Doc 2 scored higher, so it must come first.
        assert [d.id for d in result.context_docs] == [2, 1]

    def test_reranker_failure_raises_instead_of_degrading_to_hybrid(self, fake_generation):
        """A failed reranker must not silently produce plain-hybrid results.

        Those results would be stored under the "hybrid_reranker" label while
        actually being hybrid, which misrepresents the benchmark.
        """
        db = MagicMock()
        client = MagicMock()
        client.models.generate_content.side_effect = RuntimeError("rate limited")

        with patch.object(rp, "_hybrid_candidates", return_value=[_doc(1, "a")]), \
             patch.object(rp, "gemini_client", client):
            with pytest.raises(RuntimeError, match="LLM reranker failed"):
                rp.run_hybrid_rerank_rag(db, "q")

    def test_malformed_rerank_json_raises(self, fake_generation):
        db = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "not json at all"
        client = MagicMock()
        client.models.generate_content.return_value = mock_response

        with patch.object(rp, "_hybrid_candidates", return_value=[_doc(1, "a")]), \
             patch.object(rp, "gemini_client", client):
            with pytest.raises(RuntimeError, match="LLM reranker failed"):
                rp.run_hybrid_rerank_rag(db, "q")

    def test_empty_candidate_pool_short_circuits(self, fake_generation):
        db = MagicMock()
        with patch.object(rp, "_hybrid_candidates", return_value=[]), \
             patch.object(rp, "gemini_client") as client:
            result = rp.run_hybrid_rerank_rag(db, "q")

        client.models.generate_content.assert_not_called()
        assert result.answer == NO_ANSWER_MESSAGE
