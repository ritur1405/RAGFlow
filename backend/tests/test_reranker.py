"""Tests for the cross-encoder reranker in app/reranker.py.

The ONNX model is replaced with a stub, so these tests need no download and
cover ordering, truncation and the failure policy.
"""

from unittest.mock import MagicMock, patch

import pytest

from app import reranker
from app.models import Document


def _doc(doc_id: int) -> Document:
    return Document(id=doc_id, title="t", content=f"chunk {doc_id}")


def _encoder(scores):
    enc = MagicMock()
    enc.rerank.return_value = iter(scores)
    return enc


def test_orders_by_logit_and_truncates():
    docs = [_doc(1), _doc(2), _doc(3)]
    with patch.object(reranker, "_get_encoder", return_value=_encoder([-2.0, 5.5, 0.1])) as get:
        ranked = reranker.rerank("q", docs, top_k=2)

    assert [(d.id, s) for d, s in ranked] == [(2, 5.5), (3, 0.1)]
    get.return_value.rerank.assert_called_once_with("q", ["chunk 1", "chunk 2", "chunk 3"])


def test_empty_pool_does_not_load_the_model():
    with patch.object(reranker, "_get_encoder") as get:
        assert reranker.rerank("q", [], top_k=3) == []
    get.assert_not_called()


def test_model_errors_raise():
    enc = MagicMock()
    enc.rerank.side_effect = OSError("model file missing")
    with patch.object(reranker, "_get_encoder", return_value=enc):
        with pytest.raises(RuntimeError, match="Cross-encoder reranker failed: model file missing"):
            reranker.rerank("q", [_doc(1)], top_k=1)


def test_score_count_mismatch_raises():
    with patch.object(reranker, "_get_encoder", return_value=_encoder([1.0])):
        with pytest.raises(RuntimeError, match="1 scores for 2 candidates"):
            reranker.rerank("q", [_doc(1), _doc(2)], top_k=2)


def test_encoder_is_loaded_once(monkeypatch):
    monkeypatch.setattr(reranker, "_encoder", None)
    fake_cls = MagicMock()
    with patch("fastembed.rerank.cross_encoder.TextCrossEncoder", fake_cls):
        first = reranker._get_encoder()
        second = reranker._get_encoder()

    assert first is second
    fake_cls.assert_called_once_with(model_name=reranker.RERANKER_MODEL)
    monkeypatch.setattr(reranker, "_encoder", None)
