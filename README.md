# RAGOps

## 2-Person Resume-Level Project Plan

**Goal:** Build a focused RAG experimentation and evaluation platform that is strong enough for a resume and technical interviews, without turning the project into a startup-scale platform.

## 1. Project Objective

RAGOps will allow users to upload documents, build a searchable knowledge base, ask questions, and compare different RAG retrieval configurations. The main differentiator is measurable comparison of retrieval quality rather than simply building another PDF chatbot.

**Core comparison:** Baseline RAG → Dense Retrieval → Hybrid Retrieval → Hybrid + Reranker

## 2. Final MVP

| Feature | Required |
|---|---|
| PDF document upload | Yes |
| Text extraction + cleaning | Yes |
| Chunking | Yes |
| Embeddings | Yes |
| PostgreSQL + pgvector | Yes |
| Dense retrieval | Yes |
| BM25 / keyword retrieval | Yes |
| Hybrid retrieval | Yes |
| Reranking | Yes |
| LLM answer generation | Yes |
| Citations / source chunks | Yes |
| Retrieval + generation evaluation | Yes |
| Experiment comparison dashboard | Yes |
| Docker + deployment | Final stage |

## 3. Team Structure

Both members work across ML/RAG and development. Each week has a lead owner, but the second member reviews and contributes to the same deliverable. This prevents the project from becoming two disconnected halves.

| Member A — Primary Strength | Member B — Primary Strength |
|---|---|
| RAG and retrieval | FastAPI |
| Embeddings | PostgreSQL + pgvector |
| BM25 | React |
| Hybrid retrieval | API integration |
| Reranking | Dashboard |
| Evaluation | Docker/deployment |
| Experiment analysis | |

**Cross-training:** ~30% development contribution  
**Cross-training:** ~30% ML/RAG contribution

## 4. Target Architecture

React → FastAPI → PostgreSQL + pgvector  
Ingestion: PDF → extraction → cleaning → chunking → embeddings  
Retrieval: Dense / BM25 / Hybrid → optional Reranker  
Generation: LLM + source citations  
Evaluation: Retrieval metrics + generation metrics  
Dashboard: Compare experiments and visualize results

## 6. Recommended Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React | Upload, query and dashboard |
| Backend | FastAPI | REST APIs and orchestration |
| Database | PostgreSQL + pgvector | Metadata + vector storage |
| RAG | LangChain or LlamaIndex | Pipeline integration |
| Embeddings | One strong embedding model | Semantic representation |
| Retrieval | Vector search + BM25 | Dense, keyword and hybrid |
| Reranking | Cross-encoder | Improve context ordering |
| LLM | One API/cloud LLM | Answer generation |
| Deployment | Docker + suitable cloud | Reproducible deployment |

## 7. Final Benchmark

Run the same evaluation dataset through four configurations:

| Configuration | Purpose |
|---|---|
| Baseline RAG | Reference point |
| Dense RAG | Semantic retrieval performance |
| Hybrid RAG | Dense + keyword retrieval |
| Hybrid + Reranker | Best-quality retrieval candidate |

**Metrics:** Recall@K, Precision@K, MRR/NDCG, faithfulness, answer/context relevance, citation accuracy and latency.

## 8. What NOT to Build

- Microservices architecture
- Kubernetes
- Custom LLM training/fine-tuning
- Complex authentication and role systems
- Large-scale distributed vector infrastructure
- Over-engineered experiment orchestration
- Huge failure taxonomy

**Target:** a technically credible, measurable and deployable student project — not a startup platform.

## 9. Resume Positioning

**Suggested positioning:** Built a full-stack RAG experimentation platform that ingests documents, supports dense/BM25/hybrid retrieval with reranking, evaluates retrieval and generation quality, and visualizes benchmark results through a web dashboard.

**Interview topics:** Why hybrid retrieval? When does reranking help? Recall@K vs MRR? How was faithfulness evaluated? What latency/quality trade-off appeared? Which configuration performed best and why?

## 10. Working Rules

- Commit small, understandable changes.
- Use GitHub issues/tasks so ownership is clear.
- A feature is complete only after end-to-end integration and testing.
- Both members must understand the complete RAG pipeline.
- Use one shared evaluation dataset for comparable experiments.
- Prioritize working functionality over UI polish.
- If a feature does not improve learning, resume value or the final demo, cut it.

## 11. Week 3 — Dense RAG MVP (Member A)

### Pipeline Architecture

```
User Query
    ↓
Query Embedding (gemini-embedding-001, RETRIEVAL_QUERY)
    ↓
Dense Vector Search (PostgreSQL + pgvector, L2 distance)
    ↓
Top-K Relevant Chunks (filtered by distance threshold)
    ↓
Context Construction (numbered, source-attributed)
    ↓
Prompt Construction (grounded QA with citation instructions)
    ↓
LLM Generation (gemini-3.6-flash)
    ↓
Answer + Source Citations
```

### Module Structure

| Module | Purpose |
|---|---|
| `backend/app/config.py` | Centralized RAG configuration (env-var driven) |
| `backend/app/retrieval.py` | Dense retrieval: query → embedding → pgvector → `RetrievedChunk` |
| `backend/app/context.py` | Context + prompt construction (pure functions) |
| `backend/app/llm.py` | LLM generation wrapper |
| `backend/app/rag_pipeline.py` | Orchestrator — single entry point `query_rag()` |

### Dense Retrieval

The retrieval layer converts the user query into a 768-dimensional embedding using `gemini-embedding-001` with `task_type="RETRIEVAL_QUERY"` (matching the `RETRIEVAL_DOCUMENT` type used during ingestion). It then performs an L2 distance search on PostgreSQL using pgvector's `<->` operator.

### Top-K and Distance Threshold

- **Top-K** (default 5): The number of nearest chunks retrieved from pgvector. Configurable via the `RAG_TOP_K` environment variable.
- **Distance threshold** (default 0.8): Chunks with L2 distance above this value are discarded as irrelevant. Configurable via `RAG_MAX_DISTANCE`.

### Context Construction

Retrieved chunks are formatted as numbered, source-attributed blocks:

```
[Source 1 | report.pdf | Page 3 | Chunk 7]
<chunk text>

[Source 2 | report.pdf | Page 4 | Chunk 8]
<chunk text>
```

This format preserves source identity and lets the LLM reference specific sources in its answer.

### Prompt Design

The prompt instructs the LLM to:
1. Answer using ONLY the provided context
2. Not use outside knowledge or speculate
3. Respond with a fixed message if info is not available
4. Reference source identifiers (e.g. `[Source 1]`) for traceability
5. Provide concise, useful answers

### LLM Generation

Uses `gemini-3.6-flash` via the `google-genai` client. The model name is configurable via the `LLM_MODEL` environment variable.

### Citation Handling

Every citation in the response maps 1-to-1 to an actual chunk retrieved from PostgreSQL. The `RAGResult` includes a `sources` list where each `SourceChunk` contains: `chunk_id`, `document_name`, `text`, `page_number`, `chunk_index`, and `score`.

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | (required) | PostgreSQL connection string |
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `RAG_TOP_K` | `5` | Number of chunks to retrieve |
| `RAG_MAX_DISTANCE` | `0.8` | L2 distance threshold for relevance |
| `LLM_MODEL` | `gemini-3.6-flash` | LLM model name |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model name |
| `EMBEDDING_DIM` | `768` | Embedding dimensionality |

### Testing

```bash
cd backend
python -m pytest tests/ -v
```

All Week 3 tests use mocks — no paid API calls required.

### Integration Example (for Member B — Query API)

```python
from sqlalchemy.orm import Session
from app.rag_pipeline import query_rag

def handle_query(db: Session, user_question: str):
    result = query_rag(db, user_question)

    return {
        "answer": result.answer,
        "sources": [
            {
                "chunk_id": src.chunk_id,
                "document": src.document_name,
                "text": src.text,
                "page": src.page_number,
                "chunk_index": src.chunk_index,
                "score": src.score,
            }
            for src in result.sources
        ],
    }
```

