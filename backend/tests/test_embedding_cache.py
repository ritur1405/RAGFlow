"""Query embeddings are cached so repeated questions cost no extra Gemini requests."""

from unittest.mock import MagicMock, patch

from app import services
from app.config import EMBED_TASK_QUERY


def _response(values):
    return MagicMock(embeddings=[MagicMock(values=values)])


def test_repeated_query_is_embedded_once():
    services._cached_query_embedding.cache_clear()
    with patch.object(services.client.models, "embed_content", return_value=_response([0.1, 0.2])) as embed:
        first = services.generate_embedding("same question?", task_type=EMBED_TASK_QUERY)
        second = services.generate_embedding("same question?", task_type=EMBED_TASK_QUERY)
        services.generate_embedding("other question?", task_type=EMBED_TASK_QUERY)

    assert first == second == [0.1, 0.2]
    assert embed.call_count == 2
    # Callers get their own list, so mutating one can't corrupt the cache.
    first.append(9.9)
    assert services._cached_query_embedding(services.EMBEDDING_MODEL, "same question?") == (0.1, 0.2)
    services._cached_query_embedding.cache_clear()


def test_document_embeddings_are_not_cached():
    with patch.object(services.client.models, "embed_content", return_value=_response([0.3])) as embed:
        services.generate_embedding("chunk text")
        services.generate_embedding("chunk text")
    assert embed.call_count == 2
