"""Unit tests for the query API endpoints.

Validates that endpoints return the correct schema, including
the question, answer, and fully-populated source metadata.
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models import Document
from app.services import NO_ANSWER_MESSAGE

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_doc(id: int, content: str, **kwargs) -> Document:
    """Creates a mock SQLAlchemy Document object."""
    doc = MagicMock(spec=Document)
    doc.id = id
    doc.title = kwargs.get("title", f"Mock Doc {id}")
    doc.content = content
    doc.file_name = kwargs.get("file_name")
    doc.page_number = kwargs.get("page_number")
    doc.chunk_index = kwargs.get("chunk_index")
    return doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("app.main.search_documents")
@patch("app.main.generate_answer")
@patch("app.main.generate_embedding")
def test_query_with_sources_returns_full_metadata(mock_embed, mock_generate, mock_search):
    """Verifies that a successful dense RAG query populates all expected fields."""
    # Setup mocks
    mock_embed.return_value = [0.1] * 768
    
    mock_doc = _make_mock_doc(
        id=42,
        content="This is the source text.",
        file_name="report.pdf",
        page_number=3,
        chunk_index=7,
    )
    # search_documents returns list[tuple[Document, float]] (Document, distance)
    mock_search.return_value = [(mock_doc, 0.15)]
    
    mock_generate.return_value = "This is the generated answer based on [Source 1]."

    # Execute request
    response = client.post("/query/", json={"question": "What is the answer?"})
    
    # Assert HTTP success
    assert response.status_code == 200
    data = response.json()

    # Assert top-level structure
    assert data["question"] == "What is the answer?"
    assert data["answer"] == "This is the generated answer based on [Source 1]."
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 1

    # Assert source metadata fields (verifying the DocumentChunkOut schema)
    source = data["sources"][0]
    assert source["id"] == 42
    assert source["content"] == "This is the source text."
    assert source["file_name"] == "report.pdf"
    assert source["page_number"] == 3
    assert source["chunk_index"] == 7
    # 1 - 0.15 = 0.85
    assert source["relevance_score"] == 0.85


@patch("app.main.search_documents")
@patch("app.main.generate_embedding")
def test_query_with_no_relevant_documents(mock_embed, mock_search):
    """Verifies that an empty retrieval safely returns the fallback message and empty sources."""
    # Setup mocks
    mock_embed.return_value = [0.1] * 768
    
    # Simulate DB returning no relevant documents
    mock_search.return_value = []

    # Execute request
    response = client.post("/query/", json={"question": "What is the answer?"})
    
    # Assert HTTP success
    assert response.status_code == 200
    data = response.json()

    # Assert fallback behavior
    assert data["question"] == "What is the answer?"
    assert data["answer"] == NO_ANSWER_MESSAGE
    assert data["sources"] == []


@patch("app.main.search_documents")
@patch("app.main.generate_embedding")
def test_query_distance_above_threshold_returns_no_answer(mock_embed, mock_search):
    """Verifies that results exceeding MAX_RELEVANT_DISTANCE trigger the fallback."""
    from app.main import MAX_RELEVANT_DISTANCE
    
    # Setup mocks
    mock_embed.return_value = [0.1] * 768
    
    mock_doc = _make_mock_doc(id=1, content="Irrelevant text")
    # Distance is greater than the threshold
    bad_distance = MAX_RELEVANT_DISTANCE + 0.1
    mock_search.return_value = [(mock_doc, bad_distance)]

    # Execute request
    response = client.post("/query/", json={"question": "What is the answer?"})
    
    assert response.status_code == 200
    data = response.json()

    # Assert fallback behavior
    assert data["answer"] == NO_ANSWER_MESSAGE
    assert data["sources"] == []
