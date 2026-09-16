"""End-to-End mocked pipeline tests for Week 3.

Verifies the integration between the API layer, retrieval logic, context
building, and LLM prompting without making real database or API calls.
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models import Document
from app.services import NO_ANSWER_MESSAGE

client = TestClient(app)

def _make_mock_doc(id: int, content: str, title: str = "Mock Title", **kwargs) -> Document:
    doc = MagicMock(spec=Document)
    doc.id = id
    doc.title = title
    doc.content = content
    doc.file_name = kwargs.get("file_name")
    doc.page_number = kwargs.get("page_number")
    doc.chunk_index = kwargs.get("chunk_index")
    return doc


class TestPipelineMocked:

    @patch("app.main.generate_embedding")
    @patch("app.main.search_documents")
    @patch("app.services.client.models.generate_content")
    def test_successful_retrieval_pipeline(
        self, mock_generate_content, mock_search, mock_embed
    ):
        """Verifies the complete dense retrieval pipeline success path."""
        # 1. Setup mocks
        mock_embed.return_value = [0.1] * 768
        
        mock_doc1 = _make_mock_doc(1, "First chunk of information.", file_name="doc.pdf", page_number=1, chunk_index=0)
        mock_doc2 = _make_mock_doc(2, "Second chunk of information.", file_name="doc.pdf", page_number=1, chunk_index=1)
        
        mock_search.return_value = [(mock_doc1, 0.1), (mock_doc2, 0.2)]
        
        # Mock the LLM returning an answer
        mock_llm_response = MagicMock()
        mock_llm_response.text = "Based on the documents, here is the answer."
        mock_generate_content.return_value = mock_llm_response

        # Execute
        question = "What is the information?"
        top_k = 5
        response = client.post("/query/", json={"question": question, "top_k": top_k})

        # --- Verifications ---
        
        assert response.status_code == 200
        data = response.json()

        # 1. Query is embedded as RETRIEVAL_QUERY
        mock_embed.assert_called_once_with(question, task_type="RETRIEVAL_QUERY")

        # 2. top_k is passed correctly to search
        mock_search.assert_called_once()
        search_args = mock_search.call_args.kwargs
        assert search_args["top_k"] == top_k

        # 3 & 4. LLM receives the question and formatted context
        mock_generate_content.assert_called_once()
        prompt_sent_to_llm = mock_generate_content.call_args.kwargs["contents"]
        
        # Verify the prompt has the question
        assert question in prompt_sent_to_llm
        # Verify the prompt has the formatted chunks (context)
        assert "First chunk of information." in prompt_sent_to_llm
        assert "Second chunk of information." in prompt_sent_to_llm
        assert 'File: "doc.pdf", Page 1, Chunk 0' in prompt_sent_to_llm

        # 5. Returned sources match the retrieved chunks
        assert len(data["sources"]) == 2
        assert data["sources"][0]["id"] == 1
        assert data["sources"][0]["relevance_score"] == 0.9  # 1 - 0.1
        assert data["sources"][1]["id"] == 2
        assert data["sources"][1]["relevance_score"] == 0.8  # 1 - 0.2

    @patch("app.main.generate_embedding")
    @patch("app.main.search_documents")
    @patch("app.services.client.models.generate_content")
    def test_empty_retrieval_pipeline(
        self, mock_generate_content, mock_search, mock_embed
    ):
        """Verifies empty retrieval returns a safe response without calling LLM."""
        mock_embed.return_value = [0.1] * 768
        mock_search.return_value = []

        response = client.post("/query/", json={"question": "What is the information?"})
        
        assert response.status_code == 200
        data = response.json()

        # 6. Empty retrieval returns a safe response
        assert data["answer"] == NO_ANSWER_MESSAGE
        assert data["sources"] == []
        
        # Ensure LLM was never called
        mock_generate_content.assert_not_called()

    def test_invalid_top_k_rejected(self):
        """Verifies validation logic in FastAPI/Pydantic schemas."""
        # 7. Invalid top_k is rejected
        
        # top_k too low
        response_low = client.post("/query/", json={"question": "What?", "top_k": 0})
        assert response_low.status_code == 422
        
        # top_k too high
        response_high = client.post("/query/", json={"question": "What?", "top_k": 100})
        assert response_high.status_code == 422
        
        # top_k is not an int
        response_type = client.post("/query/", json={"question": "What?", "top_k": "five"})
        assert response_type.status_code == 422
