# RAGFlow — Retrieval Evaluation Dashboard

An end-to-end Retrieval-Augmented Generation (RAG) platform featuring dense vector search, BM25 keyword matching, and hybrid retrieval comparisons.

## Setup & Prerequisites

### Environment Variables
Create a `.env` file inside the `backend/` directory:

```env
DATABASE_URL=postgresql://postgres.[PROJECT_ID]:[PASSWORD]@[aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres](https://aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres)

Backend:
cd backend
.\.venv\Scripts\uvicorn app.main:app --reload --port 8000
API Documentation available at: http://localhost:8000/docs

Frontend:
cd frontend
npm run dev
Dashboard available at: http://localhost:5173